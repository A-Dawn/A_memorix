"""Optional OpenTelemetry metrics and tracing for the gRPC boundary."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterator

import logging

import grpc

from a_memorix.config import ObservabilityConfig


@dataclass
class RpcObservation:
    method: str
    status: str = "OK"


class ObservabilityRuntime:
    def __init__(self, config: ObservabilityConfig) -> None:
        self.config = config
        self._logger = logging.getLogger("a_memorix.rpc")
        self._meter_provider: Any = None
        self._tracer_provider: Any = None
        self._metrics_server: Any = None
        self._requests: Any = None
        self._duration: Any = None
        self._active: Any = None
        self._tracer: Any = None
        if config.metrics_port is not None:
            self._configure_metrics()
        if config.otlp_endpoint:
            self._configure_tracing()

    @property
    def interceptors(self) -> tuple[grpc.aio.ServerInterceptor, ...]:
        return (RpcObservabilityInterceptor(self),)

    @contextmanager
    def observe_rpc(self, method: str) -> Iterator[RpcObservation]:
        started = perf_counter()
        observation = RpcObservation(method=method)
        attributes = {"rpc.system": "grpc", "rpc.method": method}
        if self._active is not None:
            self._active.add(1, attributes)
        span_context = (
            self._tracer.start_as_current_span(method, attributes=attributes)
            if self._tracer is not None
            else _null_context()
        )
        try:
            with span_context as span:
                try:
                    yield observation
                except BaseException as exc:
                    observation.status = _exception_status(exc)
                    if span is not None:
                        span.record_exception(exc)
                    raise
                finally:
                    if span is not None:
                        span.set_attribute("rpc.grpc.status_code", observation.status)
        finally:
            duration_ms = (perf_counter() - started) * 1000
            completed_attributes = {
                **attributes,
                "rpc.grpc.status_code": observation.status,
            }
            if self._active is not None:
                self._active.add(-1, attributes)
                self._requests.add(1, completed_attributes)
                self._duration.record(duration_ms, completed_attributes)
            if self.config.access_log:
                self._logger.info(
                    "gRPC request completed",
                    extra={
                        "rpc_method": method,
                        "rpc_status": observation.status,
                        "duration_ms": round(duration_ms, 3),
                    },
                )

    def shutdown(self) -> None:
        if self._metrics_server is not None:
            self._metrics_server.shutdown()
            self._metrics_server.server_close()
            self._metrics_server = None
        if self._meter_provider is not None:
            self._meter_provider.shutdown()
            self._meter_provider = None
        if self._tracer_provider is not None:
            self._tracer_provider.shutdown()
            self._tracer_provider = None

    def _configure_metrics(self) -> None:
        try:
            from opentelemetry.exporter.prometheus import PrometheusMetricReader
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.resources import Resource
            from prometheus_client import start_http_server
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "metrics require the 'observability' extra: "
                "pip install 'a-memorix[observability]'"
            ) from exc
        reader = PrometheusMetricReader()
        self._meter_provider = MeterProvider(
            metric_readers=(reader,),
            resource=Resource.create({"service.name": self.config.service_name}),
        )
        meter = self._meter_provider.get_meter("a_memorix.server")
        self._requests = meter.create_counter(
            "a_memorix.rpc.requests",
            description="Completed gRPC requests",
        )
        self._duration = meter.create_histogram(
            "a_memorix.rpc.duration",
            unit="ms",
            description="gRPC request duration",
        )
        self._active = meter.create_up_down_counter(
            "a_memorix.rpc.active",
            description="Active gRPC requests",
        )
        self._metrics_server, _ = start_http_server(
            self.config.metrics_port,
            addr=self.config.metrics_host,
        )

    def _configure_tracing(self) -> None:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "tracing requires the 'observability' extra: "
                "pip install 'a-memorix[observability]'"
            ) from exc
        self._tracer_provider = TracerProvider(
            resource=Resource.create({"service.name": self.config.service_name}),
            sampler=ParentBased(TraceIdRatioBased(self.config.trace_sample_ratio)),
        )
        exporter = OTLPSpanExporter(
            endpoint=self.config.otlp_endpoint,
            insecure=self.config.otlp_insecure,
        )
        self._tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
        self._tracer = self._tracer_provider.get_tracer("a_memorix.server")


class RpcObservabilityInterceptor(grpc.aio.ServerInterceptor):
    def __init__(self, runtime: ObservabilityRuntime) -> None:
        self._runtime = runtime

    async def intercept_service(self, continuation: Any, handler_call_details: Any) -> Any:
        handler = await continuation(handler_call_details)
        if handler is None or handler.unary_unary is None:
            return handler
        method = str(handler_call_details.method).removeprefix("/")

        async def observed(request: Any, context: grpc.aio.ServicerContext) -> Any:
            with self._runtime.observe_rpc(method) as observation:
                try:
                    return await handler.unary_unary(request, context)
                finally:
                    code = context.code()
                    if code is not None:
                        observation.status = code.name

        return grpc.unary_unary_rpc_method_handler(
            observed,
            request_deserializer=handler.request_deserializer,
            response_serializer=handler.response_serializer,
        )


@contextmanager
def _null_context() -> Iterator[None]:
    yield None


def _exception_status(error: BaseException) -> str:
    if isinstance(error, grpc.RpcError):
        code = error.code()
        if code is not None:
            return code.name
    return "UNKNOWN"
