"""A single background event loop for running the async RAG pipeline.

Streamlit re-executes the script on every interaction, often from a fresh
thread, and some of those threads already have (or forbid) a running loop.
`asyncio.run` per call would create and destroy a loop each time, which breaks
LangChain's cached async HTTP clients ("Event loop is closed"). One long-lived
loop on a daemon thread avoids both problems: every coroutine runs on the same
loop, and the Streamlit script thread only ever blocks on a plain
`concurrent.futures.Future`.
"""
from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future
from typing import Any, Coroutine


class AsyncBridge:
    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="rag-async-bridge", daemon=True)
        self._thread.start()
        self._ready.wait()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.call_soon(self._ready.set)
        self._loop.run_forever()

    def submit(self, coro: Coroutine[Any, Any, Any]) -> Future:
        """Schedule `coro` on the background loop; returns a concurrent Future."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def run(self, coro: Coroutine[Any, Any, Any], timeout: float | None = None) -> Any:
        """Run `coro` to completion from sync code."""
        return self.submit(coro).result(timeout)

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
