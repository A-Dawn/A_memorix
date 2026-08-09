"""Bearer authentication and namespace authorization for gRPC services."""

from __future__ import annotations

from dataclasses import dataclass

import secrets

import grpc

from a_memorix.contracts import ForbiddenError, UnauthorizedError
from a_memorix.engine import AMemorixEngine


@dataclass(frozen=True)
class AuthPrincipal:
    principal_id: str
    namespace_id: str | None = None
    is_admin: bool = False


class GrpcAuthPolicy:
    def __init__(
        self,
        engine: AMemorixEngine,
        *,
        admin_token: str,
        allow_unauthenticated: bool = False,
    ) -> None:
        token = str(admin_token or "").strip()
        if not allow_unauthenticated and len(token) < 32:
            raise ValueError("admin_token must contain at least 32 characters")
        self._engine = engine
        self._admin_token = token
        self._allow_unauthenticated = allow_unauthenticated

    def require_admin(self, context: grpc.aio.ServicerContext) -> AuthPrincipal:
        if self._allow_unauthenticated:
            return AuthPrincipal(principal_id="anonymous", is_admin=True)
        token = self._bearer_token(context)
        if secrets.compare_digest(token, self._admin_token):
            return AuthPrincipal(principal_id="admin", is_admin=True)
        api_key = self._engine.authenticate_api_key(token)
        if api_key is not None:
            raise ForbiddenError("administrator credentials are required")
        raise UnauthorizedError("invalid bearer token")

    def require_namespace(
        self,
        context: grpc.aio.ServicerContext,
        namespace_id: str,
    ) -> AuthPrincipal:
        if self._allow_unauthenticated:
            return AuthPrincipal(
                principal_id="anonymous",
                namespace_id=namespace_id,
            )
        token = self._bearer_token(context)
        if secrets.compare_digest(token, self._admin_token):
            return AuthPrincipal(
                principal_id="admin",
                namespace_id=namespace_id,
                is_admin=True,
            )
        api_key = self._engine.authenticate_api_key(token)
        if api_key is None:
            raise UnauthorizedError("invalid bearer token")
        if not secrets.compare_digest(api_key.namespace_id, namespace_id):
            raise ForbiddenError(
                "API key is not authorized for this namespace",
                details={
                    "requested_namespace_id": namespace_id,
                    "authorized_namespace_id": api_key.namespace_id,
                },
            )
        return AuthPrincipal(
            principal_id=f"api-key:{api_key.key_id}",
            namespace_id=api_key.namespace_id,
        )

    @staticmethod
    def _bearer_token(context: grpc.aio.ServicerContext) -> str:
        values = [
            str(item.value)
            for item in context.invocation_metadata()
            if item.key.lower() == "authorization"
        ]
        if len(values) != 1:
            raise UnauthorizedError("exactly one Authorization header is required")
        scheme, separator, token = values[0].partition(" ")
        if not separator or scheme.casefold() != "bearer" or not token.strip():
            raise UnauthorizedError("Authorization must use the Bearer scheme")
        return token.strip()
