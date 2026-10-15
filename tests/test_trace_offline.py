"""Offline tests for the trace layer (no models, no network).  pytest tests/"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util, types  # noqa: E402

# Load rag/trace.py without importing the heavy `rag` package.
_spec = importlib.util.spec_from_file_location(
    "trace_mod", os.path.join(os.path.dirname(__file__), "..", "rag", "trace.py"))
trace = importlib.util.module_from_spec(_spec)
sys.modules["trace_mod"] = trace  # dataclasses needs the module registered
_spec.loader.exec_module(trace)


def test_recorder_and_store(tmp_path):
    updates = []
    rec = trace.TraceRecorder.new("q", "s1", on_update=lambda t: updates.append(len(t.steps)))
    rec.declare("router", "plan", "retrieve")
    with rec.step("router", "q") as s:
        s.output = {"decision": "chat"}
    rec.skip_many(["plan", "retrieve"], "chat route")
    try:
        with rec.step("answer", "q"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    rec.finish(answer="x")
    t = rec.trace
    assert t.step("router").status == "ok" and t.step("router").duration_ms is not None
    assert t.step("plan").status == "skipped"
    assert t.step("answer").status == "failed" and "boom" in t.step("answer").error
    assert updates, "listener never called"

    store = trace.TraceStore(str(tmp_path / "t.db"))
    store.save(t)
    (loaded,) = store.load_session("s1")
    assert loaded.to_dict() == t.to_dict()
    json.dumps(loaded.to_dict())
    assert store.latest_session_id() == "s1"
    store.delete_session("s1")
    assert store.load_session("s1") == []


def test_cancelled_step_is_marked():
    rec = trace.TraceRecorder.new("q", "s")

    async def run():
        async def work():
            with rec.step("plan"):
                await asyncio.sleep(10)
        task = asyncio.create_task(work())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(run())
    assert rec.trace.step("plan").status == "cancelled"
