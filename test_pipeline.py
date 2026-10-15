"""Live smoke test: one retrieve turn and one chat turn, asserting the trace shape.

    python test_pipeline.py

Needs API keys in .env and at least one indexed document (index via the app or
`python -m rag`). Traces are written to a throw-away SQLite file, not traces.db.
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from rag import TraceStore, TurnTrace, list_indexed, respond, sync_documents

RETRIEVE_STEPS = {"router", "plan", "pool", "rerank", "answer"}
SKIPPED_ON_CHAT = {"plan", "retrieve", "pool", "rerank"}


def check_common(t: TurnTrace) -> None:
    assert t.turn_id and t.session_id and t.query and t.timestamp
    assert t.route in {"retrieve", "chat"}, t.route
    assert t.answer.strip(), "empty answer"
    assert t.total_ms and t.total_ms > 0
    assert t.model_used, "model_used not recorded"
    names = [s.name for s in t.steps]
    assert len(names) == len(set(names)), f"duplicate steps: {names}"
    for s in t.steps:
        assert s.status in {"ok", "skipped", "failed", "cancelled"}, (s.name, s.status)
        json.dumps(s.output)  # must be JSON-serialisable
        if s.status in {"ok", "failed"}:
            assert s.duration_ms is not None and s.duration_ms >= 0, s.name
    assert t.step("router").status == "ok"
    assert t.step("answer").status == "ok"


def check_retrieve(t: TurnTrace) -> None:
    check_common(t)
    assert t.route == "retrieve", f"router chose {t.route!r} for a document question"
    names = {s.name for s in t.steps}
    assert RETRIEVE_STEPS <= names, names
    variants = [s for s in t.steps if s.name.startswith("retrieve:")]
    assert any(s.name == "retrieve:original" for s in variants)
    assert all("hits" in s.output for s in variants if s.status == "ok")
    assert t.step("pool").output["sent_to_rerank"] >= 0
    assert t.documents, "no documents recorded (is anything indexed?)"
    for rank, d in enumerate(t.documents, 1):
        assert set(d) >= {"rank", "file_name", "score", "snippet", "found_by"}
        assert d["rank"] == rank and d["found_by"] and isinstance(d["found_by"], list)


def check_chat(t: TurnTrace) -> None:
    check_common(t)
    assert t.route == "chat", f"router chose {t.route!r} for small talk"
    for n in SKIPPED_ON_CHAT:
        s = t.step(n)
        assert s is not None and s.status in {"skipped", "cancelled"}, (n, s and s.status)
    assert not t.documents


async def main() -> int:
    load_dotenv()
    sync_documents()
    if not list_indexed():
        print("No documents indexed; add some to ./document first.")
        return 2
    store = TraceStore(str(Path(tempfile.mkdtemp()) / "test_traces.db"))
    sid = "smoke-test"

    _, r1 = await respond("What are the main topics covered in the documents?", sid, store=store)
    check_retrieve(r1.trace)
    print("retrieve turn OK:", [(s.name, s.status) for s in r1.trace.steps])

    _, r2 = await respond("Hello! How are you today?", sid, store=store)
    check_chat(r2.trace)
    print("chat turn OK:    ", [(s.name, s.status) for s in r2.trace.steps])

    saved = store.load_session(sid)
    assert [t.turn_id for t in saved] == [r1.trace.turn_id, r2.trace.turn_id]
    assert saved[0].to_dict() == r1.trace.to_dict(), "persisted trace differs from live trace"
    print("persistence round-trip OK")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
