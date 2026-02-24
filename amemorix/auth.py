"""Bearer token auth middleware."""

from __future__ import annotations

from typing import Iterable, Optional, Set

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from .common.logging import get_logger

logger = get_logger("A_Memorix.Auth")

WRITE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
PUBLIC_PATHS = {"/healthz", "/readyz"}


def _extract_bearer(auth_header: str) -> Optional[str]:
    if not auth_header:
        return None
    parts = auth_header.strip().split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    token = parts[1].strip()
    return token or None


def _token_set(values: Iterable[str]) -> Set[str]:
    return {str(v).strip() for v in values if str(v).strip()}


class BearerAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        ctx = getattr(request.app.state, "context", None)
        if ctx is None:
            return await call_next(request)

        enabled = bool(ctx.get_config("auth.enabled", True))
        if not enabled:
            return await call_next(request)

        path = request.url.path
        if path in PUBLIC_PATHS:
            return await call_next(request)

        protect_read = bool(ctx.get_config("auth.protect_read_endpoints", False))
        requires_auth = request.method.upper() in WRITE_METHODS or protect_read
        if not requires_auth:
            return await call_next(request)

        write_tokens = _token_set(ctx.get_config("auth.write_tokens", []) or [])
        read_tokens = _token_set(ctx.get_config("auth.read_tokens", []) or [])
        allow_tokens = write_tokens if request.method.upper() in WRITE_METHODS else (read_tokens or write_tokens)

        if not allow_tokens:
            logger.warning("Auth enabled but no tokens configured.")
            return JSONResponse(
                status_code=503,
                content={"detail": "Authentication tokens are not configured."},
            )

        token = _extract_bearer(request.headers.get("Authorization", ""))
        if token is None or token not in allow_tokens:
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

        return await call_next(request)

