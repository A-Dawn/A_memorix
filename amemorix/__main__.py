"""CLI entrypoint."""

from __future__ import annotations

import argparse
import sys
from typing import Optional

import uvicorn

from .settings import AppSettings


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="amemorix",
        description="A_Memorix API-specialized standalone service",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Run API service")
    serve.add_argument("--config", type=str, default=None, help="Path to config.toml")
    serve.add_argument("--host", type=str, default=None, help="Override listen host")
    serve.add_argument("--port", type=int, default=None, help="Override listen port")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    if args.command != "serve":
        return 2

    settings = AppSettings.load(args.config)
    if args.host:
        settings.config.setdefault("server", {})["host"] = args.host
    if args.port:
        settings.config.setdefault("server", {})["port"] = int(args.port)
    settings.config.setdefault("server", {})["workers"] = 1

    from .app import create_app

    app = create_app(settings=settings)
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        workers=1,
        log_level="debug" if bool(settings.get("advanced.debug", False)) else "info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
