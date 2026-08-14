"""Build runtime providers from the service configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from a_memorix.config import ProvidersConfig, read_secret
from a_memorix.contracts import NamespaceConfig, ProviderReference
from a_memorix.ports import NamespaceHostPorts

from .openai_compatible import (
    OpenAICompatibleEmbeddingProvider,
    OpenAICompatibleLLMProvider,
)


@dataclass(frozen=True)
class ConfiguredProviders:
    """Provider instances and matching non-secret runtime configuration."""

    config: ProvidersConfig
    embedding: OpenAICompatibleEmbeddingProvider | None
    llm: OpenAICompatibleLLMProvider | None

    def host_ports(self, _namespace: object) -> NamespaceHostPorts:
        return NamespaceHostPorts(
            embedding_provider=self.embedding,
            llm_provider=self.llm,
        )

    def runtime_config(self, _namespace: object) -> Mapping[str, object]:
        embedding = self.config.embedding
        return {
            "embedding": {
                "dimension": embedding.dimension,
                "batch_size": embedding.batch_size,
                "max_concurrent": embedding.max_concurrent,
                "model_name": embedding.model or "auto",
                "dimension_request_mode": embedding.dimension_request_mode,
                "retry": {"max_attempts": 1},
            }
        }

    def namespace_config(self) -> NamespaceConfig:
        return NamespaceConfig(
            embedding=_provider_reference(
                self.config.embedding.provider,
                self.config.embedding.model,
                self.config.embedding.api_key_file,
                "A_MEMORIX_EMBEDDING_API_KEY",
            )
            if self.embedding is not None
            else None,
            llm=_provider_reference(
                self.config.llm.provider,
                self.config.llm.model,
                self.config.llm.api_key_file,
                "A_MEMORIX_LLM_API_KEY",
            )
            if self.llm is not None
            else None,
        )

    async def probe_standard(self, *, probe_llm: bool = True) -> None:
        if self.embedding is None:
            raise RuntimeError(
                "standard MCP mode requires an Embedding endpoint and model"
            )
        if self.llm is None:
            raise RuntimeError("standard MCP mode requires an LLM endpoint and model")
        requested_dimension = (
            self.config.embedding.dimension
            if self.config.embedding.dimension_request_mode == "always"
            else None
        )
        observed_dimension = await self.embedding.probe(
            dimensions=requested_dimension
        )
        configured_dimension = self.config.embedding.dimension
        if observed_dimension != configured_dimension:
            raise RuntimeError(
                "Embedding dimension mismatch: "
                f"configured={configured_dimension}, observed={observed_dimension}"
            )
        if probe_llm:
            await self.llm.probe()


def build_configured_providers(config: ProvidersConfig) -> ConfiguredProviders:
    embedding_config = config.embedding
    llm_config = config.llm
    embedding = None
    if embedding_config.configured:
        embedding = OpenAICompatibleEmbeddingProvider(
            endpoint=embedding_config.endpoint,
            api_key=read_secret(
                environment_name="A_MEMORIX_EMBEDDING_API_KEY",
                file_path=embedding_config.api_key_file,
            ),
            model=embedding_config.model,
            timeout_seconds=embedding_config.timeout_seconds,
            max_attempts=embedding_config.max_attempts,
            retry_delay_seconds=embedding_config.retry_delay_seconds,
            retry_max_delay_seconds=embedding_config.retry_max_delay_seconds,
            retry_backoff_multiplier=embedding_config.retry_backoff_multiplier,
            max_concurrent=embedding_config.max_concurrent,
        )
    llm = None
    if llm_config.configured:
        llm = OpenAICompatibleLLMProvider(
            endpoint=llm_config.endpoint,
            api_key=read_secret(
                environment_name="A_MEMORIX_LLM_API_KEY",
                file_path=llm_config.api_key_file,
            ),
            model=llm_config.model,
            timeout_seconds=llm_config.timeout_seconds,
            max_attempts=llm_config.max_attempts,
            retry_delay_seconds=llm_config.retry_delay_seconds,
            retry_max_delay_seconds=llm_config.retry_max_delay_seconds,
            retry_backoff_multiplier=llm_config.retry_backoff_multiplier,
            max_concurrent=llm_config.max_concurrent,
            max_tokens=llm_config.max_tokens,
            temperature=llm_config.temperature,
            enable_thinking=llm_config.enable_thinking,
            thinking_mode=llm_config.thinking_mode,
        )
    return ConfiguredProviders(config=config, embedding=embedding, llm=llm)


def _provider_reference(
    provider_id: str,
    model_id: str,
    api_key_file: object,
    environment_name: str,
) -> ProviderReference:
    secret_ref = (
        "config:api_key_file"
        if api_key_file is not None
        else f"env:{environment_name}"
    )
    return ProviderReference(
        provider_id=provider_id,
        model_id=model_id,
        secret_ref=secret_ref,
    )
