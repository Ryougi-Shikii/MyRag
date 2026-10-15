"""Streamlit front end for the RAG package.  Run with:  streamlit run app.py"""
from __future__ import annotations

import logging
import threading
import time
import types
import uuid
from typing import Any

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="Document Q&A", page_icon="📜", layout="wide")

from async_bridge import AsyncBridge  # noqa: E402
from graph_view import graph_html  # noqa: E402
from rag import (  # noqa: E402
    UPLOAD_TYPES, RetrieverFactory, TurnTrace, default_store, delete_document,
    forget_session, list_indexed, respond, save_uploads, sync_documents,
)
from rag.config import (  # noqa: E402
    GRAPH_MAX_TURNS, UI_POLL_SECONDS, get_agent, get_embeddings, get_reranker,
    get_translation_agent,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
log = logging.getLogger("app")

GRAPH_HEIGHT = 820
TURN_LOAD_LIMIT = 60
STATUS_ICON = {"pending": "⚪", "running": "🟡", "ok": "✅", "skipped": "⏭️", "failed": "❌", "cancelled": "⛔"}
STEP_TITLES = {
    "router": "Router", "plan": "Plan", "retrieve": "Retrieve", "pool": "Pool & dedupe",
    "rerank": "Rerank", "answer": "Answer",
}


# ================================================================ cached resources
@st.cache_resource
def get_bridge() -> AsyncBridge:
    """One background event loop for the whole server process."""
    return AsyncBridge()


@st.cache_resource
def get_index_state() -> dict:
    """Process-wide index version; bumped after every (re-)index or delete."""
    return {"version": 0}


@st.cache_resource(show_spinner="Loading embedding model, reranker and agents…")
def load_resources(index_version: int) -> types.SimpleNamespace:
    """Models, agents and the retriever. The version argument is the cache key,
    so bumping it (see `bump_index`) rebuilds the retriever over the new index."""
    get_embeddings()
    get_reranker()
    get_agent()
    get_translation_agent()
    return types.SimpleNamespace(factory=RetrieverFactory.cached(), version=index_version)


def bump_index() -> None:
    get_index_state()["version"] += 1
    load_resources.clear()


# ================================================================ helpers
def step_title(name: str) -> str:
    if name.startswith("retrieve:"):
        v = name[9:]
        return "Retrieve · " + ("HyDE" if v == "hyde" else v.replace("expansion_", "expand "))
    return STEP_TITLES.get(name, name)


def fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n} B"


def fmt_ms(ms: float | None) -> str:
    if ms is None:
        return ""
    return f"{ms:.0f} ms" if ms < 1000 else f"{ms / 1000:.1f} s"


class TurnJob:
    """Thread-safe mailbox between the pipeline (event-loop thread) and the UI.

    Streamlit elements may only be touched from the script thread, so the
    pipeline callbacks just store a snapshot here and the script polls it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._trace: dict | None = None
        self._text = ""

    def on_update(self, trace: TurnTrace) -> None:
        snap = trace.to_dict()
        with self._lock:
            self._trace = snap

    def on_token(self, piece: str) -> None:
        with self._lock:
            self._text += piece

    def read(self) -> tuple[dict | None, str]:
        with self._lock:
            return self._trace, self._text


class IndexJob:
    """Background indexing run with a progress mailbox."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.fraction = 0.0
        self.message = "Starting…"
        self.log: list[str] = []
        self.failed: list[str] = []
        self.done = False
        self.changed = False
        self.error: str | None = None
        self.handled = False

    def callback(self, ev: dict) -> None:
        with self._lock:
            self.fraction = ev["fraction"]
            self.message = ev["message"] or ev["stage"]
            line = f"{ev['stage']}: {ev['message']}".strip()
            if not self.log or self.log[-1] != line:
                self.log.append(line)
            if ev["stage"] == "failed":
                self.failed.append(ev["message"])

    def run(self) -> None:
        try:
            self.changed = sync_documents(progress_cb=self.callback)
        except Exception as exc:
            log.exception("Indexing failed")
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.done = True

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "fraction": self.fraction, "message": self.message, "log": list(self.log),
                "failed": list(self.failed), "done": self.done, "error": self.error,
                "changed": self.changed,
            }


def history_messages(turns: list[TurnTrace]) -> list[dict]:
    """Rebuild the chat transcript from persisted traces (survives restarts)."""
    msgs: list[dict] = []
    for t in turns:
        msgs.append({"role": "user", "content": t.query, "trace": None})
        text = t.answer or (f"⚠️ This turn failed: {t.error}" if t.error else "")
        msgs.append({"role": "assistant", "content": text, "trace": t.to_dict()})
    return msgs


def render_details(trace: dict | None, show_reasoning: bool) -> None:
    """Expander under an answer: route/model line, sources, optional reasoning."""
    if not trace:
        return
    docs = trace.get("documents") or []
    reasoning = trace.get("reasoning") if show_reasoning else ""
    if not docs and not reasoning:
        return
    route = {"retrieve": "🔎 retrieve", "chat": "💬 chat"}.get(trace.get("route", ""), "")
    label = "Sources & reasoning" if show_reasoning else "Sources"
    with st.expander(label):
        meta = " · ".join(x for x in [route, trace.get("model_used"), fmt_ms(trace.get("total_ms"))] if x)
        if meta:
            st.caption(meta)
        for d in docs:
            score = f" · score {d['score']:.3f}" if d.get("score") is not None else ""
            found = f" · found by {', '.join(d['found_by'])}" if d.get("found_by") else ""
            st.markdown(f"**#{d['rank']} {d['file_name']}**{score}{found}")
            st.caption(d.get("snippet", ""))
        if reasoning:
            st.markdown("**Reasoning**")
            st.text(reasoning)


def steps_markdown(trace: dict | None) -> str:
    if not trace:
        return "…"
    lines = []
    for s in trace["steps"]:
        extra = f" — {fmt_ms(s['duration_ms'])}" if s.get("duration_ms") is not None else ""
        lines.append(f"{STATUS_ICON.get(s['status'], '•')} {step_title(s['name'])}{extra}")
    return "  \n".join(lines)


def trace_signature(trace: dict | None) -> tuple:
    if not trace:
        return ()
    return (trace["route"], trace["error"], tuple((s["name"], s["status"]) for s in trace["steps"]))


def draw_graph(slot, turns: list[dict], selected: str | None, live: bool) -> None:
    html = graph_html(turns, selected, live=live, max_turns=GRAPH_MAX_TURNS)
    with slot.container():
        if hasattr(st, "iframe"):   # components.v1.html is deprecated in recent Streamlit
            st.iframe(html, height=GRAPH_HEIGHT)
        else:
            components.html(html, height=GRAPH_HEIGHT, scrolling=False)


# ================================================================ session state
ss = st.session_state
store = default_store()
bridge = get_bridge()

if "session_id" not in ss:
    ss.session_id = store.latest_session_id() or uuid.uuid4().hex[:12]
    ss.focus_turn = None
    ss.uploader_key = 0

try:
    resources = load_resources(get_index_state()["version"])
except Exception as exc:  # e.g. missing API key: show it instead of a stack trace
    log.exception("Could not initialise the pipeline")
    st.error(f"Could not initialise the RAG pipeline: {type(exc).__name__}: {exc}")
    st.info("Check your `.env` (GROQ_API_KEY / GOOGLE_API_KEY) and that the models can be downloaded.")
    st.stop()

index_job: IndexJob | None = ss.get("index_job")
indexing = index_job is not None and not index_job.done

# ================================================================ sidebar
with st.sidebar:
    st.header("📚 Documents")
    files = st.file_uploader(
        "Add documents", type=UPLOAD_TYPES, accept_multiple_files=True,
        key=f"uploader_{ss.uploader_key}", disabled=indexing,
    )
    if st.button("Index documents", type="primary", use_container_width=True, disabled=indexing):
        if files:
            save_uploads(files)
            ss.uploader_key += 1          # clears the uploader
        job = IndexJob()
        ss.index_job = job
        threading.Thread(target=job.run, name="indexer", daemon=True).start()
        st.rerun()

    @st.fragment(run_every=0.5 if indexing else None)
    def index_progress() -> None:
        job: IndexJob | None = st.session_state.get("index_job")
        if job is None:
            return
        snap = job.snapshot()
        if snap["done"] and not job.handled:
            job.handled = True
            if snap["changed"]:
                bump_index()              # rebuild retriever over the new index
            st.rerun()                    # full rerun: refresh list, re-enable chat
        if not snap["done"]:
            with st.status("Indexing…", expanded=True):
                st.progress(snap["fraction"], text=snap["message"])
                for line in snap["log"][-5:]:
                    st.caption(line)
        elif snap["error"]:
            st.error(f"Indexing failed: {snap['error']}")
        else:
            if snap["failed"]:
                st.warning("Some files failed (they'll be retried next time):\n\n- " + "\n- ".join(snap["failed"]))
            st.success("Index updated." if snap["changed"] else "Index already up to date.")

    index_progress()

    docs = list_indexed()
    if docs:
        st.caption(f"{len(docs)} document(s)")
    else:
        st.caption("No documents yet. Upload some and press **Index documents**.")
    for d in docs:
        c1, c2 = st.columns([5, 1])
        badge = {"indexed": "", "pending": " · ⏳ not indexed", "changed": " · ⚠️ changed"}[d["status"]]
        chunks = f" · {d['chunks']} chunks" if d["chunks"] is not None else ""
        c1.markdown(f"**{d['name']}**  \n<small>{d['type']} · {fmt_size(d['size'])}{chunks}{badge}</small>",
                    unsafe_allow_html=True)
        if c2.button("🗑", key=f"del_{d['name']}", help=f"Delete {d['name']}", disabled=indexing):
            with st.spinner(f"Deleting {d['name']}…"):
                delete_document(d["name"])
                bump_index()
            st.rerun()

    st.divider()
    st.header("💬 Chat")
    b1, b2 = st.columns(2)
    if b1.button("New chat", use_container_width=True, disabled=indexing):
        ss.session_id = uuid.uuid4().hex[:12]
        ss.focus_turn = None
        st.rerun()
    if b2.button("Clear history", use_container_width=True, disabled=indexing):
        store.delete_session(ss.session_id)
        forget_session(ss.session_id)
        ss.focus_turn = None
        st.rerun()
    st.toggle("Show reasoning", key="show_reasoning", value=False)

show_reasoning = bool(ss.get("show_reasoning"))

# ================================================================ main area
turns = store.load_session(ss.session_id, limit=TURN_LOAD_LIMIT)
turn_dicts = [t.to_dict() for t in turns]

st.title("📜 Document Q&A")
chat_tab, graph_tab = st.tabs(["Chat", "Pipeline Graph"])

with chat_tab:
    if not docs:
        st.info("No documents are indexed yet; chat works, but answers won't draw on your files.")
    for m in history_messages(turns):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                render_details(m["trace"], show_reasoning)

with graph_tab:
    if turn_dicts:
        ids = [t["turn_id"] for t in turn_dicts]
        focus = ss.focus_turn if ss.focus_turn in ids else ids[-1]
        options = list(reversed(ids))
        labels = {
            t["turn_id"]: ("Current chat" if i == len(ids) - 1 else f"#{i + 1}") + " · " + t["query"][:60]
            for i, t in enumerate(turn_dicts)
        }
        picker_key = f"picker-{ss.session_id}-{len(ids)}-{focus}"

        def _on_pick(key: str = picker_key) -> None:
            st.session_state.focus_turn = st.session_state[key]

        st.selectbox(
            "Expanded chat (you can also click a chat node in the graph)",
            options, index=options.index(focus), format_func=labels.get,
            key=picker_key, on_change=_on_pick,
        )
    else:
        focus = None
    graph_slot = st.empty()
    draw_graph(graph_slot, turn_dicts, focus, live=False)


# ================================================================ run a turn
prompt = st.chat_input(
    "Indexing in progress…" if indexing else "Ask about your documents…", disabled=indexing,
)

if prompt:
    job = TurnJob()
    with chat_tab:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            answer_slot = st.empty()
            status = st.status("Thinking…", expanded=False)
            with status:
                steps_slot = st.empty()
            future = bridge.submit(respond(
                prompt, ss.session_id, factory=resources.factory,
                on_update=job.on_update, on_token=job.on_token,
            ))

            last_sig: tuple = ()
            last_text = ""
            while not future.done():
                snap, text = job.read()
                sig = trace_signature(snap)
                if sig != last_sig and snap:
                    last_sig = sig
                    running = next((s for s in snap["steps"] if s["status"] == "running"), None)
                    status.update(label=(f"{step_title(running['name'])}…" if running else "Thinking…"))
                    steps_slot.markdown(steps_markdown(snap))
                    draw_graph(graph_slot, turn_dicts + [snap], snap["turn_id"], live=True)
                if text != last_text:
                    last_text = text
                    answer_slot.markdown(text + " ▌")
                time.sleep(UI_POLL_SECONDS)

            snap, text = job.read()
            error = None
            try:
                _, reply = future.result()
                answer_slot.markdown(reply.text)
                status.update(label=f"Done in {fmt_ms(reply.trace.total_ms)}", state="complete", expanded=False)
            except Exception as exc:  # the failed step is already in the trace/graph
                log.exception("Turn failed")
                error = f"{type(exc).__name__}: {exc}"
                answer_slot.error(f"This turn failed: {error}")
                status.update(label="Failed", state="error", expanded=True)
                steps_slot.markdown(steps_markdown(snap))
            if snap:
                draw_graph(graph_slot, turn_dicts + [snap], snap["turn_id"], live=False)
                ss.focus_turn = snap["turn_id"]
    time.sleep(0.4)  # let the user see the final state before the rerun redraws from SQLite
    st.rerun()
