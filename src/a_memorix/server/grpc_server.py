"""Lifecycle wrapper for the A_memorix grpc.aio server."""

from __future__ import annotations

from ipaddress import ip_address
from typing import TYPE_CHECKING

import os

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from a_memorix.engine import AMemorixEngine

from .auth import GrpcAuthPolicy
from .services import register_services

if TYPE_CHECKING:
    from a_memorix.observability import ObservabilityRuntime


_HEALTH_SERVICES = (
    "",
    "a_memorix.api.v1.NamespaceService",
    "a_memorix.api.v1.BackupService",
    "a_memorix.api.v1.AuthService",
    "a_memorix.api.v1.MemoryService",
    "a_memorix.api.v1.JobService",
)


class AMemorixGrpcServer:
    def __init__(
        self,
        engine: AMemorixEngine,
        *,
        host: str = "127.0.0.1",
        port: int = 50051,
        admin_token: str | None = None,
        allow_unauthenticated: bool = False,
        credentials: grpc.ServerCredentials | None = None,
        manage_engine_lifecycle: bool = True,
        maximum_message_bytes: int = 16 * 1024 * 1024,
        observability: "ObservabilityRuntime | None" = None,
    ) -> None:
        if not (0 <= int(port) <= 65535):
            raise ValueError("port must be between 0 and 65535")
        if allow_unauthenticated and not _is_loopback(host):
            raise ValueError("unauthenticated gRPC may only listen on a loopback address")
        self._engine = engine
        self._host = host
        self._requested_port = int(port)
        self._credentials = credentials
        self._manage_engine_lifecycle = manage_engine_lifecycle
        self._observability = observability
        resolved_admin_token = (
            os.environ.get("A_MEMORIX_ADMIN_TOKEN", "")
            if admin_token is None
            else admin_token
        )
        self._auth = GrpcAuthPolicy(
            engine,
            admin_token=resolved_admin_token,
            allow_unauthenticated=allow_unauthenticated,
        )
        self._options = (
            ("grpc.max_receive_message_length", maximum_message_bytes),
            ("grpc.max_send_message_length", maximum_message_bytes),
        )
        self._server: grpc.aio.Server | None = None
        self._health: health.aio.HealthServicer | None = None
        self._bound_port: int | None = None

    @property
    def bound_port(self) -> int:
        if self._bound_port is None:
            raise RuntimeError("gRPC server has not been started")
        return self._bound_port

    @property
    def target(self) -> str:
        return _address(self._host, self.bound_port)

    async def start(self) -> None:
        if self._server is not None:
            return
        if self._manage_engine_lifecycle:
            await self._engine.initialize()
        interceptors = (
            self._observability.interceptors
            if self._observability is not None
            else ()
        )
        server = grpc.aio.server(options=self._options, interceptors=interceptors)
        register_services(server, self._engine, self._auth)
        health_service = health.aio.HealthServicer()
        health_pb2_grpc.add_HealthServicer_to_server(health_service, server)
        address = _address(self._host, self._requested_port)
        try:
            if self._credentials is None:
                bound_port = server.add_insecure_port(address)
            else:
                bound_port = server.add_secure_port(address, self._credentials)
            if bound_port == 0:
                raise RuntimeError(f"gRPC failed to bind {address}")
            await server.start()
            for service_name in _HEALTH_SERVICES:
                await health_service.set(service_name, health_pb2.HealthCheckResponse.SERVING)
        except BaseException:
            await server.stop(0)
            if self._manage_engine_lifecycle:
                await self._engine.shutdown()
            if self._observability is not None:
                self._observability.shutdown()
            raise
        self._server = server
        self._health = health_service
        self._bound_port = bound_port

    async def stop(self, grace: float = 5.0) -> None:
        server = self._server
        if server is None:
            return
        self._server = None
        self._bound_port = None
        health_service = self._health
        self._health = None
        try:
            if health_service is not None:
                await health_service.enter_graceful_shutdown()
            await server.stop(grace)
        finally:
            if self._manage_engine_lifecycle:
                await self._engine.shutdown()
            if self._observability is not None:
                self._observability.shutdown()

    async def wait_for_termination(self) -> None:
        if self._server is None:
            raise RuntimeError("gRPC server has not been started")
        await self._server.wait_for_termination()

    async def __aenter__(self) -> "AMemorixGrpcServer":
        await self.start()
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        await self.stop()


def _is_loopback(host: str) -> bool:
    normalized = str(host or "").strip().strip("[]")
    if normalized.casefold() == "localhost":
        return True
    try:
        return ip_address(normalized).is_loopback
    except ValueError:
        return False


def _address(host: str, port: int) -> str:
    normalized = str(host or "").strip()
    if ":" in normalized and not normalized.startswith("["):
        normalized = f"[{normalized}]"
    return f"{normalized}:{port}"
