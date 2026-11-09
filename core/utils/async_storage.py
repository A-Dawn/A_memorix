"""将存储 I/O 移出事件循环，并在取消时等待线程结束后再释放资源。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TypeVar

from src.common.logger import get_logger

_Result = TypeVar("_Result")
logger = get_logger("A_Memorix.AsyncStorage")


async def run_storage_io(operation: Callable[[], _Result]) -> _Result:
    return await wait_for_storage_task(asyncio.create_task(asyncio.to_thread(operation)))


async def wait_for_storage_task(worker: asyncio.Task[_Result]) -> _Result:
    cancellation = None
    while not worker.done():
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError as exc:
            # Python 不能强制终止工作线程；提前返回会使 close 与初始化并发。
            cancellation = exc
        except Exception:
            break
    if cancellation is not None:
        if not worker.cancelled() and worker.exception() is not None:
            error = worker.exception()
            logger.error("存储操作在取消期间失败", exc_info=(type(error), error, error.__traceback__))
        raise cancellation
    return worker.result()
