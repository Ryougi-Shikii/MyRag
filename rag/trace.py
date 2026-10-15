"""Observability layer: step/turn traces, a live recorder, and SQLite persistence.

The pipeline never talks to the UI directly. It writes into a `TraceRecorder`;
the recorder mutates a `TurnTrace` and calls an optional `on_update(trace)`
callback after every change, which is how the Streamlit graph animates a turn
while it runs. Finished turns are persisted by `TraceStore`.
"""
from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterator

logger = logging.getLogger(__name__)

# Step statuses. The first four are the persisted outcomes; pending/running
# only exist while a turn is in flight.
PENDING, RUNNING, OK, SKIPPED, FAILED, CANCELLED = (
    "pending", "running", "ok", "skipped", "failed", "cancelled",
)
FINAL_STATUSES = {OK, SKIPPED, FAILED, CANCELLED}


def short(text: Any, limit: int = 300) -> str:
    """Collapse whitespace and truncate for display in `StepTrace.input`."""
    s = " ".join(str(text).split())
    return s if len(s) <= limit else s[: limit - 1] + "…"


# ================================================================ dataclasses
@dataclass
class StepTrace:
    name: str
    status: str = PENDING
    started_at: float | None = None      # epoch seconds
    duration_ms: float | None = None
    input: str = ""
    output: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StepTrace":
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class TurnTrace:
    turn_id: str
    session_id: str
    query: str
    timestamp: str                        # ISO-8601, UTC
    route: str = ""                       # "retrieve" | "chat" | "" (not decided yet)
    steps: list[StepTrace] = field(default_factory=list)
    answer: str = ""
    reasoning: str = ""
    model_used: str = ""
    documents: list[dict] = field(default_factory=list)  # {rank,file_name,score,snippet,found_by}
    total_ms: float | None = None
    error: str | None = None

    def step(self, name: str) -> StepTrace | None:
        return next((s for s in self.steps if s.name == name), None)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TurnTrace":
        d = dict(d)
        steps = [StepTrace.from_dict(s) for s in d.pop("steps", [])]
        known = cls.__dataclass_fields__
        return cls(steps=steps, **{k: v for k, v in d.items() if k in known})


# ================================================================ recorder
class TraceRecorder:
    """Mutates one `TurnTrace` and notifies a listener after every change."""

    def __init__(
        self,
        trace: TurnTrace,
        on_update: Callable[[TurnTrace], None] | None = None,
    ):
        self.trace = trace
        self._on_update = on_update
        self._t0 = time.perf_counter()
        self._clock: dict[int, float] = {}

    @classmethod
    def new(
        cls,
        query: str,
        session_id: str,
        on_update: Callable[[TurnTrace], None] | None = None,
    ) -> "TraceRecorder":
        trace = TurnTrace(
            turn_id=uuid.uuid4().hex,
            session_id=session_id,
            query=query,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        )
        return cls(trace, on_update)

    # -------------------------------------------------------------- events
    def emit(self) -> None:
        if self._on_update is None:
            return
        try:
            self._on_update(self.trace)
        except Exception:  # a broken listener must never break the pipeline
            logger.exception("trace listener failed")

    # -------------------------------------------------------------- steps
    def get(self, name: str) -> StepTrace | None:
        return self.trace.step(name)

    def declare(self, *names: str) -> None:
        """Pre-register steps as pending so a live graph can show them."""
        for n in names:
            if self.get(n) is None:
                self.trace.steps.append(StepTrace(name=n))

    def expand(self, placeholder: str, names: list[str]) -> None:
        """Replace a pending placeholder step by concrete pending steps."""
        steps = self.trace.steps
        idx = next((i for i, s in enumerate(steps) if s.name == placeholder), None)
        fresh = [StepTrace(name=n) for n in names if self.get(n) is None]
        if idx is None:
            steps.extend(fresh)
        else:
            steps[idx:idx + 1] = fresh
        self.emit()

    def start(self, name: str, input: Any = "") -> StepTrace:
        s = self.get(name)
        if s is None:
            s = StepTrace(name=name)
            self.trace.steps.append(s)
        s.status = RUNNING
        s.started_at = time.time()
        s.input = short(input)
        self._clock[id(s)] = time.perf_counter()
        self.emit()
        return s

    def end(self, s: StepTrace, status: str | None = None) -> None:
        if status:
            s.status = status
        elif s.status == RUNNING:
            s.status = OK
        t0 = self._clock.pop(id(s), None)
        if t0 is not None:
            s.duration_ms = round((time.perf_counter() - t0) * 1000, 1)
        self.emit()

    @contextlib.contextmanager
    def step(self, name: str, input: Any = "") -> Iterator[StepTrace]:
        """Time a step. Code inside may set `s.output`, or override `s.status`
        (e.g. "failed" for a handled error) before leaving the block."""
        s = self.start(name, input)
        try:
            yield s
        except BaseException as exc:
            # asyncio.CancelledError is a BaseException, hence the wide catch.
            if isinstance(exc, Exception):
                tb = traceback.extract_tb(exc.__traceback__)
                where = f" (at {tb[-1].filename.rsplit('/', 1)[-1]}:{tb[-1].lineno} in {tb[-1].name})" if tb else ""
                s.error = f"{type(exc).__name__}: {exc}{where}"
                self.end(s, FAILED)
            else:
                self.end(s, CANCELLED)
            raise
        else:
            self.end(s)

    def skip(self, name: str, reason: str) -> None:
        s = self.get(name)
        if s is None:
            s = StepTrace(name=name)
            self.trace.steps.append(s)
        s.status = SKIPPED
        s.output = {**s.output, "reason": reason}
        self.emit()

    def skip_many(self, names: list[str], reason: str) -> None:
        for n in names:
            self.skip(n, reason)

    def fail_running(self, exc: BaseException) -> None:
        """Close every step still running when a turn dies."""
        cancelled = not isinstance(exc, Exception)
        for s in self.trace.steps:
            if s.status == RUNNING:
                if not cancelled:
                    s.error = f"{type(exc).__name__}: {exc}"
                self.end(s, CANCELLED if cancelled else FAILED)

    # -------------------------------------------------------------- turn
    def finish(self, answer: str = "", reasoning: str = "", error: str | None = None) -> None:
        t = self.trace
        t.answer = answer or t.answer
        t.reasoning = reasoning or t.reasoning
        t.error = error
        t.total_ms = round((time.perf_counter() - self._t0) * 1000, 1)
        self.emit()


# ================================================================ persistence
class TraceStore:
    """SQLite store for finished turns, keyed by session_id."""

    def __init__(self, path: str | None = None):
        if path is None:
            from .config import TRACES_DB  # lazy: keeps this module import-light
            path = TRACES_DB
        self.path = str(path)
        self._init_lock = threading.Lock()
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init(self) -> None:
        with self._init_lock, contextlib.closing(self._connect()) as c, c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS turns ("
                " turn_id TEXT PRIMARY KEY,"
                " session_id TEXT NOT NULL,"
                " ts TEXT NOT NULL,"
                " payload TEXT NOT NULL)"
            )
            c.execute("CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, ts)")

    def save(self, trace: TurnTrace) -> None:
        payload = json.dumps(trace.to_dict(), default=str)
        with contextlib.closing(self._connect()) as c, c:
            c.execute(
                "INSERT OR REPLACE INTO turns(turn_id, session_id, ts, payload) VALUES (?,?,?,?)",
                (trace.turn_id, trace.session_id, trace.timestamp, payload),
            )

    def load_session(self, session_id: str, limit: int | None = None) -> list[TurnTrace]:
        """Turns of a session, oldest first. `limit` keeps the most recent N."""
        sql = "SELECT payload FROM turns WHERE session_id=? ORDER BY ts DESC, rowid DESC"
        args: list[Any] = [session_id]
        if limit:
            sql += " LIMIT ?"
            args.append(limit)
        with contextlib.closing(self._connect()) as c:
            rows = c.execute(sql, args).fetchall()
        return [TurnTrace.from_dict(json.loads(r[0])) for r in reversed(rows)]

    def latest_session_id(self) -> str | None:
        with contextlib.closing(self._connect()) as c:
            row = c.execute("SELECT session_id FROM turns ORDER BY ts DESC, rowid DESC LIMIT 1").fetchone()
        return row[0] if row else None

    def delete_session(self, session_id: str) -> None:
        with contextlib.closing(self._connect()) as c, c:
            c.execute("DELETE FROM turns WHERE session_id=?", (session_id,))


_default_store: TraceStore | None = None
_default_lock = threading.Lock()


def default_store() -> TraceStore:
    global _default_store
    with _default_lock:
        if _default_store is None:
            _default_store = TraceStore()
        return _default_store
