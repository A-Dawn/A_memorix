"""Run the A_memorix gRPC service."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os

from a_memorix.engine import AMemorixEngine

from .grpc_server import AMemorixGrpcServer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the A_memorix gRPC service")
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("A_MEMORIX_DATA_DIR", "./data"),
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("A_MEMORIX_GRPC_HOST", "127.0.0.1"),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("A_MEMORIX_GRPC_PORT", "50051")),
    )
    parser.add_argument(
        "--allow-unauthenticated",
        action="store_true",
        help="allow anonymous access; valid only on a loopback listener",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(_serve(args))
    except KeyboardInterrupt:
        pass


async def _serve(args: argparse.Namespace) -> None:
    engine = AMemorixEngine(data_dir=args.data_dir)
    server = AMemorixGrpcServer(
        engine,
        host=args.host,
        port=args.port,
        allow_unauthenticated=args.allow_unauthenticated,
    )
    await server.start()
    logging.getLogger(__name__).info("gRPC listening on %s", server.target)
    try:
        await server.wait_for_termination()
    finally:
        await server.stop()


if __name__ == "__main__":
    main()
