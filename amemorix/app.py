"""FastAPI app factory for standalone runtime."""

from __future__ import annotations

from fastapi import FastAPI

from server import MemorixServer

from .auth import BearerAuthMiddleware
from .bootstrap import build_context
from .common.logging import get_logger, setup_logging
from .routers.v1_router import router as v1_router
from .settings import AppSettings
from .task_manager import TaskManager

logger = get_logger("A_Memorix.App")


def create_app(*, settings: AppSettings) -> FastAPI:
    setup_logging("DEBUG" if bool(settings.get("advanced.debug", False)) else "INFO")

    context = build_context(settings)
    compat = MemorixServer(
        plugin_instance=context,
        host=settings.host,
        port=settings.port,
    )
    app = compat.app
    app.state.context = context
    app.state.task_manager = TaskManager(context)

    # Auth is applied globally. It enforces writes by default.
    app.add_middleware(BearerAuthMiddleware)
    app.include_router(v1_router)

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz():
        ctx = app.state.context
        return {
            "ready": all(
                [
                    ctx.metadata_store is not None,
                    ctx.graph_store is not None,
                    ctx.vector_store is not None,
                    ctx.retriever is not None,
                ]
            )
        }

    @app.on_event("shutdown")
    async def _shutdown():
        try:
            await app.state.task_manager.stop()
        except Exception as exc:
            logger.warning("Task manager stop failed: %s", exc)
        try:
            await app.state.context.close()
        except Exception as exc:
            logger.warning("Shutdown close failed: %s", exc)

    @app.on_event("startup")
    async def _startup():
        try:
            await app.state.task_manager.start()
        except Exception as exc:
            logger.warning("Task manager start failed: %s", exc)

    return app
