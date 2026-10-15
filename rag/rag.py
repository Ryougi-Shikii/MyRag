"""Retrieval chain, query translation, and the top-level Q&A pipeline."""
from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import logging
import pickle
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage


from .config import (
    BM25_CACHE_FILE, ENSEMBLE_WEIGHTS, HISTORY_CHAR_LIMIT, HISTORY_TURNS,
    MAX_RERANK_INPUT, PRIMARY_CHAT_MODEL, RERANK_TOP_N, RERANKER_MODEL,
    TRACE_SNIPPET_CHARS, get_agent, get_reranker,
    get_translation_agent, get_vector_store, ENSEMBLE_K, HYDE_K,
)
from .trace import TraceRecorder, TraceStore, TurnTrace, default_store, short

logger = logging.getLogger(__name__)


# ================================================================ result type
@dataclass
class Reply:
    """What a turn produced. Printing is the caller's job."""
    text: str
    reasoning: str = ""
    documents: list[Document] = field(default_factory=list)
    trace: TurnTrace | None = None


# ================================================================ retriever
class RetrieverFactory:
    """BM25 + vector ensemble, cross-encoder reranked. Cached per vector store."""

    _instance_cache: dict = {}

    def __init__(self, vector_store: Chroma | None = None):
        self.vector_store = vector_store if vector_store is not None else get_vector_store()
        self._base_ensemble = None
        self._compressor = None

    @classmethod
    def cached(cls, vector_store: Chroma | None = None) -> "RetrieverFactory":
        key = id(vector_store) if vector_store is not None else "default"
        if key not in cls._instance_cache:
            cls._instance_cache[key] = cls(vector_store)
        return cls._instance_cache[key]

    def get_vector_retriever(self):
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": ENSEMBLE_K},
        )

    def get_base_ensemble(self):
        if self._base_ensemble is None:
            vector_retriever = self.get_vector_retriever()
            bm25 = self._get_bm25_retriever()
            if bm25 is not None:
                self._base_ensemble = EnsembleRetriever(
                    retrievers=[bm25, vector_retriever],
                    weights=ENSEMBLE_WEIGHTS,
                )
            else:
                self._base_ensemble = vector_retriever
        return self._base_ensemble

    def get_compressor(self):
        if self._compressor is None:
            self._compressor = CrossEncoderReranker(
                model=get_reranker(), top_n=RERANK_TOP_N,
            )
        return self._compressor

    def _get_bm25_retriever(self):
        if Path(BM25_CACHE_FILE).exists():
            try:
                with open(BM25_CACHE_FILE, "rb") as f:
                    return pickle.load(f)
            except Exception:
                logger.warning("BM25 cache unreadable; rebuilding", exc_info=True)

        stored = self.vector_store.get(include=["documents", "metadatas"])
        # Blank chunks (e.g. an image with no OCR text) have no tokens; rank_bm25
        # divides by the corpus/vocabulary size and raises ZeroDivisionError
        # when nothing is left, so drop them and fall back to vector-only.
        docs = [
            Document(page_content=d, metadata=m)
            for d, m in zip(stored["documents"], stored["metadatas"])
            if d and d.strip()
        ]
        if not docs:
            return None
        retriever = BM25Retriever.from_documents(docs)
        retriever.k = ENSEMBLE_K
        try:
            with open(BM25_CACHE_FILE, "wb") as f:
                pickle.dump(retriever, f)
        except Exception:
            logger.warning("Could not persist BM25 cache", exc_info=True)
        return retriever


def clear_retriever_cache() -> None:
    RetrieverFactory._instance_cache.clear()


# ================================================================ agent turn
def _reasoning_text(block: dict) -> str:
    """Extract the reasoning payload from a reasoning-type content block.

    Different providers name the field differently (`reasoning`, `summary`,
    `text`), and `summary` can be a list of sub-blocks. Be liberal.
    """
    for key in ("reasoning", "text", "content", "summary"):
        value = block.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    t = item.get("text") or item.get("content")
                    if isinstance(t, str):
                        parts.append(t)
            if parts:
                return "".join(parts)
    return ""


def _extract_blocks(msg) -> tuple[str, str]:
    """Return (reasoning, text) from an assistant message.

    Reasoning models (gpt-oss-120b, deepseek-r1, ...) prepend a
    {"type": "reasoning", ...} block to `content_blocks`; the actual answer
    is a later {"type": "text", ...} block. Non-reasoning models only emit
    the text block.
    """
    reasoning_chunks: list[str] = []
    text = ""

    blocks = getattr(msg, "content_blocks", None) or []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "reasoning":
            chunk = _reasoning_text(block)
            if chunk:
                reasoning_chunks.append(chunk)
        elif btype == "text" and not text:
            text = block.get("text", "")

    # Fallbacks for providers that don't use content_blocks.
    if not text:
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "".join(
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )

    return "\n\n".join(reasoning_chunks).strip(), text


def _model_name(msg) -> str:
    """Which model produced `msg`, from provider response metadata."""
    meta = getattr(msg, "response_metadata", None) or {}
    for key in ("model_name", "model", "model_id"):
        value = meta.get(key)
        if value:
            return str(value)
    return ""


def _chunk_text(chunk) -> str:
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    return "".join(
        b.get("text", "") for b in content
        if isinstance(b, dict) and b.get("type") == "text"
    )


def _thread_id(session_id: str) -> str:
    return f"chat-{session_id}"


def forget_session(session_id: str) -> None:
    """Drop the in-memory conversation thread of a session (best effort)."""
    saver = getattr(get_agent(), "checkpointer", None)
    try:
        saver.delete_thread(_thread_id(session_id))
    except Exception:
        logger.warning("Could not delete thread for %s", session_id, exc_info=True)


async def _mark_retrieval(session_id: str, n_docs: int) -> None:
    """Append a system note so future turns know this turn used retrieval.

    The retrieved chunks themselves are NOT written to the thread; only this
    marker is. `_recent_history` ignores system messages, so the marker never
    shows up in the planner/router transcript, but the chat agent sees it.
    """
    await get_agent().aupdate_state(
        {"configurable": {"thread_id": _thread_id(session_id)}},
        {"messages": [SystemMessage(
            content=(
                f"[retrieval-note] The previous answer was produced using "
                f"{n_docs} retrieved document chunk(s) from the local document "
                "store. Those chunks were injected for that single model call "
                "and are NOT retained in this conversation."
            )
        )]},
    )


async def _invoke_full(
        agent,
        prompt: str,
        thread_id: str | None = None,
        context: dict | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> tuple[str, str, str]:
    """Run one turn and return (reasoning, text, model_name). Never prints.

    With `on_token`, the answer is streamed through the agent's "messages"
    stream mode and the final message is then read back from the checkpoint.
    If streaming fails before the first token we fall back to a plain call.
    """
    config = {"configurable": {"thread_id": thread_id}} if thread_id else None
    kwargs = {"context": context} if context is not None else {}
    payload = {"messages": [{"role": "user", "content": prompt}]}

    if on_token is not None and config is not None:
        emitted = False
        try:
            async for chunk, meta in agent.astream(
                payload, config=config, stream_mode="messages", **kwargs,
            ):
                if meta.get("langgraph_node") != "model":
                    continue
                piece = _chunk_text(chunk)
                if piece:
                    emitted = True
                    on_token(piece)
            state = await agent.aget_state(config)
            msg = state.values["messages"][-1]
            reasoning, text = _extract_blocks(msg)
            return reasoning, text, _model_name(msg)
        except Exception:
            if emitted:
                raise
            logger.warning("Streaming failed before first token; retrying without it", exc_info=True)

    out = await agent.ainvoke(payload, config=config, **kwargs)
    msg = out["messages"][-1]
    reasoning, text = _extract_blocks(msg)
    return reasoning, text, _model_name(msg)


async def _invoke(
        agent,
        prompt: str,
        thread_id: str | None = None,
        context: dict | None = None,
    ) -> tuple[str, str]:
    """Run one turn and return (reasoning, text). Never prints."""
    reasoning, text, _ = await _invoke_full(agent, prompt, thread_id, context)
    return reasoning, text


async def _recent_history(session_id: str) -> str:
    """Last few turns of the session thread, as a plain transcript.

    Read-only: the query-side agent never writes to the thread, so its
    rewrite / routing prompts can't pollute the conversation memory.
    """
    state = await get_agent().aget_state(
        {"configurable": {"thread_id": _thread_id(session_id)}}
    )
    messages = (state.values or {}).get("messages", [])

    lines: list[str] = []
    for msg in messages[-2 * HISTORY_TURNS:]:
        if msg.type not in ("human", "ai"):
            continue
        text = _extract_blocks(msg)[1].strip()
        if not text:
            continue
        if len(text) > HISTORY_CHAR_LIMIT:
            text = text[:HISTORY_CHAR_LIMIT] + "..."
        lines.append(f"{'User' if msg.type == 'human' else 'Assistant'}: {text}")
    return "\n".join(lines)


def _history_block(history: str) -> str:
    return f"Conversation so far:\n{history}\n\n" if history else ""


# ================================================================ query translation
_PLAN_PROMPT = (
    "You are a query planner for a technical document retrieval system. "
    "{history}"
    "Given the user's latest question, return ONE JSON object with exactly "
    "these keys:\n"
    '  "rewrite": a single, self-contained search query (resolve pronouns and '
    "conversational filler using the conversation above if needed),\n"
    '  "expansions": a list of {n} alternative search queries covering '
    "different angles (definition, usage, examples, edge cases),\n"
    '  "hyde": a short, factual, encyclopedic paragraph (80-150 words) that '
    "would be the PERFECT answer, written like a textbook passage, without "
    "referencing the question.\n"
    "Output ONLY the JSON object.\n\n"
    "User question: {query}"
)


def _parse_plan(raw: str) -> tuple[str, list[str], str]:
    """Parse the planner output into (rewrite, expansions, hyde). Raises ValueError."""
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in planner output")
    data = json.loads(raw[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("planner output is not a JSON object")

    rewrite = str(data.get("rewrite") or "").strip()
    hyde = str(data.get("hyde") or "").strip()
    raw_expansions = data.get("expansions")
    expansions = (
        [str(e).strip() for e in raw_expansions if str(e).strip()]
        if isinstance(raw_expansions, list) else []
    )
    return rewrite, expansions, hyde


async def _translate_query(
        agent, query: str, history: str = "", expansion_n: int = 3,
        rec: TraceRecorder | None = None,
    ) -> dict:
    """Rewrite + expansions + HyDE in ONE call.

    A malformed answer degrades to original-query-only retrieval rather than
    failing the turn (the step is recorded as "failed" so the graph shows it);
    LLM/transport errors still propagate.
    """
    rec = rec or TraceRecorder.new(query, "-")
    prompt = _PLAN_PROMPT.format(
        history=_history_block(history), n=expansion_n, query=query,
    )
    with rec.step("plan", query) as st:
        _, raw = await _invoke(agent, prompt)
        try:
            rewrite, expansions, hyde = _parse_plan(raw)
        except ValueError as exc:  # includes json.JSONDecodeError
            logger.warning("Query planner returned unparseable output; using original query only")
            st.status = "failed"
            st.error = f"unparseable planner output ({exc}); used original query only"
            st.output = {"fallback": "original-query-only", "raw": raw[:500], "variants": ["original"]}
            return {"original": query, "rewrite": "", "hyde": ""}

        variants = {"original": query, "rewrite": rewrite, "hyde": hyde}
        variants.update({f"expansion_{i+1}": e for i, e in enumerate(expansions[:expansion_n])})
        st.output = {
            "rewrite": rewrite,
            "expansions": expansions[:expansion_n],
            "hyde": hyde,
            "variants": [k for k, v in variants.items() if v],
        }
    logger.debug("Translated query into %d variants", len(variants))
    return variants


# ================================================================ pipeline
def _dedupe_docs(docs: list[Document]) -> list[Document]:
    seen, out = set(), []
    for d in docs:
        key = d.page_content.strip()
        if key and key not in seen:
            seen.add(key)
            out.append(d)
    return out


def _interleave(batches: list[list[Document]]) -> list[Document]:
    """Round-robin the per-variant result lists (rank 1 of each, then rank 2...).

    Concatenating them and cutting at MAX_RERANK_INPUT always dropped the last
    batch (HyDE); interleaving gives every variant a fair share of the budget.
    """
    return [d for group in itertools.zip_longest(*batches) for d in group if d is not None]


def _source_name(doc: Document) -> str:
    return Path(doc.metadata.get("source_file", "Unknown")).name


def _snippet(text: str) -> str:
    return short(text, TRACE_SNIPPET_CHARS)


async def _score_pairs(docs: list[Document], query: str) -> list[float | None]:
    """Real cross-encoder scores for display (cheap: only the kept chunks)."""
    if not docs:
        return []
    try:
        scores = await asyncio.to_thread(
            get_reranker().score, [(query, d.page_content) for d in docs],
        )
        return [round(float(s), 4) for s in scores]
    except Exception:
        logger.warning("Could not score reranked documents", exc_info=True)
        return [None] * len(docs)


def _answer_output(model: str, text: str, n_ctx: int) -> dict:
    primary = PRIMARY_CHAT_MODEL.split("/")[-1]
    return {
        "model": model or "unknown",
        "fallback_used": bool(model) and primary not in model,
        "chars": len(text),
        "context_chunks": n_ctx,
    }


# ================================================================ routing
async def classify(
        query: str, history: str | None = None, session_id: str = "test",
        rec: TraceRecorder | None = None,
    ) -> str:
    """Return 'retrieve' or 'chat'. One cheap LLM call."""
    rec = rec or TraceRecorder.new(query, session_id)
    if history is None:
        history = await _recent_history(session_id)
    prompt = (
        "You are a router inside a document Q&A assistant. Decide whether the "
        "user's latest message needs a lookup in a technical document store, or "
        "whether it is conversational (greeting, smalltalk, meta-question, or a "
        "follow-up answerable from prior turns alone).\n\n"
        "Reply with exactly one word: retrieve or chat.\n\n"
        f"{_history_block(history)}"
        f"Message: {query}"
    )
    with rec.step("router", query) as st:
        _, raw = await _invoke(get_translation_agent(), prompt)
        decision = raw.strip().lower()
        route = "retrieve" if decision.startswith("retrieve") else "chat"
        st.output = {"decision": route, "raw": raw.strip()[:80]}
    return route


# ================================================================ chat
async def chat(
        query: str, session_id: str = "test",
        rec: TraceRecorder | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> Reply:
    """Conversational reply. No retrieval, but keeps session memory."""
    rec = rec or TraceRecorder.new(query, session_id)
    rec.trace.route = rec.trace.route or "chat"
    with rec.step("answer", query) as st:
        reasoning, text, model = await _invoke_full(
            get_agent(), query, thread_id=_thread_id(session_id), on_token=on_token,
        )
        rec.trace.model_used = model
        st.output = _answer_output(model, text, 0)
    return Reply(text=text, reasoning=reasoning, trace=rec.trace)


# ================================================================ ask
async def ask(
        query: str,
        factory: RetrieverFactory | None = None,
        expansion_n: int = 3,
        session_id: str = "test",
        variants: dict | None = None,
        rec: TraceRecorder | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> Reply:
    """translate -> retrieve per variant (parallel) -> pool -> rerank -> answer.

    Query planning (rewrite / expand / HyDE) uses the dedicated translation
    model; the final answer comes from the chat chain. Pass `variants` to reuse
    an already computed plan (see `respond`). The session thread only ever
    stores the raw question and the answer, never the retrieved context.
    """
    rec = rec or TraceRecorder.new(query, session_id)
    rec.trace.route = rec.trace.route or "retrieve"
    if factory is None:
        factory = RetrieverFactory.cached()

    if variants is None:
        history = await _recent_history(session_id)
        variants = await _translate_query(
            get_translation_agent(), query, history, expansion_n, rec,
        )
    base = factory.get_base_ensemble()
    compressor = factory.get_compressor()

    # (variant name, query text); HyDE goes last and uses pure vector search.
    plan = [(n, t) for n, t in variants.items() if n != "hyde" and t]
    if variants.get("hyde"):
        plan.append(("hyde", variants["hyde"]))
    rec.expand("retrieve", [f"retrieve:{n}" for n, _ in plan])

    hybrid = isinstance(base, EnsembleRetriever)

    async def _search(name: str, text: str) -> list[Document]:
        with rec.step(f"retrieve:{name}", text) as st:
            if name == "hyde":
                docs = await factory.vector_store.asimilarity_search(text, k=HYDE_K)
                retriever = "vector (HyDE)"
            else:
                docs = await base.ainvoke(text)
                retriever = "bm25+vector" if hybrid else "vector"
            st.output = {
                "retriever": retriever,
                "hits": len(docs),
                "files": sorted({_source_name(d) for d in docs}),
                "top": [
                    {"file_name": _source_name(d), "snippet": _snippet(d.page_content)}
                    for d in docs[:3]
                ],
            }
            return docs

    async def _safe_search(name: str, text: str) -> list[Document]:
        # One failed variant is recorded (red node) and contributes nothing;
        # the turn carries on with the others.
        try:
            return await _search(name, text)
        except Exception:
            logger.exception("[ask] retrieval for variant %r failed", name)
            return []

    batches = await asyncio.gather(*[_safe_search(n, t) for n, t in plan])

    # Tag every chunk with the variants that found it, BEFORE dedupe.
    found_by: dict[str, list[str]] = {}
    for (name, _), batch in zip(plan, batches):
        for d in batch:
            names = found_by.setdefault(d.page_content.strip(), [])
            if name not in names:
                names.append(name)

    with rec.step("pool", f"{len(plan)} result sets") as st:
        merged = _interleave(list(batches))
        blank = sum(1 for d in merged if not d.page_content.strip())
        deduped = _dedupe_docs(merged)
        pooled = deduped[:MAX_RERANK_INPUT]
        coverage = Counter(n for d in pooled for n in found_by.get(d.page_content.strip(), []))
        st.output = {
            "raw_hits": sum(len(b) for b in batches),
            "blank_hits": blank,
            "sample_hit_lengths": [len(d.page_content.strip()) for d in merged[:8]],
            "after_dedupe": len(deduped),
            "sent_to_rerank": len(pooled),
            "dropped_by_cap": len(deduped) - len(pooled),
            "cap": MAX_RERANK_INPUT,
            "interleaved": True,
            "per_variant_hits": {n: len(b) for (n, _), b in zip(plan, batches)},
            "per_variant_in_pool": dict(coverage),
        }
        if merged and blank == len(merged):
            st.status = "failed"
            st.error = (
                f"All {blank} retrieved chunks are blank, so there is nothing to rerank. "
                "The index holds empty text (e.g. scanned PDFs or images without OCR). "
                "Re-index: the schema was bumped, so Index documents re-processes every file "
                "and reports any file with no extractable text."
            )

    logger.info("Pooled %d unique document(s) across query variants", len(pooled))

    rerank_query = variants.get("rewrite") or query
    with rec.step("rerank", rerank_query) as st:
        fallback = False
        if not pooled:
            reranked: list[Document] = []
        else:
            try:
                reranked = await compressor.acompress_documents(pooled, rerank_query)
            except Exception as exc:
                logger.exception(
                    "[ask] reranker failed; falling back to pool top-%d", RERANK_TOP_N
                )
                reranked = pooled[:RERANK_TOP_N]
                fallback = True
                st.status = "failed"
                st.error = f"reranker failed ({type(exc).__name__}: {exc}); kept pool top-{RERANK_TOP_N}"

        scores = [None] * len(reranked) if fallback else await _score_pairs(reranked, rerank_query)
        rec.trace.documents = [
            {
                "rank": i,
                "file_name": _source_name(d),
                "score": scores[i - 1],
                "snippet": _snippet(d.page_content),
                "found_by": found_by.get(d.page_content.strip(), []),
            }
            for i, d in enumerate(reranked, 1)
        ]
        st.output = {
            "model": RERANKER_MODEL,
            "input_count": len(pooled),
            "kept": len(reranked),
            "top_n": RERANK_TOP_N,
            "fallback": fallback,
            "documents": rec.trace.documents,
        }

    logger.info("Reranked down to %d document(s)", len(reranked))

    context = "\n\n".join(
        f"[Document {i} | Source: {Path(res.metadata.get('source_file', 'Unknown')).name}]\n"
        f"{res.page_content}"
        for i, res in enumerate(reranked, 1)
    )

    logger.info("Answering %r via retrieval", query)
    with rec.step("answer", query) as st:
        reasoning, text, model = await _invoke_full(
            get_agent(),
            query,
            thread_id=_thread_id(session_id),
            context={"retrieved_context": context},
            on_token=on_token,
        )
        rec.trace.model_used = model
        st.output = _answer_output(model, text, len(reranked))
    if reranked:
        await _mark_retrieval(session_id, len(reranked))
    return Reply(text=text, reasoning=reasoning, documents=reranked, trace=rec.trace)


# ================================================================ top level
def _persist(trace: TurnTrace, store: TraceStore | bool | None) -> None:
    if store is False:
        return
    try:
        (default_store() if store in (True, None) else store).save(trace)
    except Exception:
        logger.exception("Could not persist trace %s", trace.turn_id)


async def respond(
        query: str,
        session_id: str = "test",
        factory: RetrieverFactory | None = None,
        expansion_n: int = 3,
        on_update: Callable[[TurnTrace], None] | None = None,
        on_token: Callable[[str], None] | None = None,
        store: TraceStore | bool | None = True,
    ) -> tuple[str, Reply]:
    """Route and answer one message. Returns (kind, reply); `reply.trace` holds
    the full TurnTrace, which is also persisted unless `store=False`.

    Routing and query planning run concurrently, so a retrieval turn costs one
    LLM hop before retrieval instead of two. On a chat turn the speculative
    plan is cancelled and discarded.

    `on_update(trace)` fires after every step change (used for the live graph);
    `on_token(piece)` receives answer tokens as they stream. Both may be called
    from whatever thread the event loop runs on. If the turn raises, the trace
    (with the failing step marked) is still persisted and attached to the
    exception as `exc.trace`.
    """
    rec = TraceRecorder.new(query, session_id, on_update=on_update)
    rec.declare("router", "plan", "retrieve", "pool", "rerank", "answer")
    rec.emit()
    plan = None
    try:
        history = await _recent_history(session_id)
        plan = asyncio.create_task(
            _translate_query(get_translation_agent(), query, history, expansion_n, rec)
        )
        kind = await classify(query, history, session_id, rec)
        rec.trace.route = kind
        rec.emit()
        if kind == "chat":
            plan.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await plan
            ps = rec.get("plan")
            if ps is not None and ps.status in ("ok", "pending"):
                ps.status = "skipped"
                ps.output = {**ps.output, "reason": "chat route: speculative plan discarded"}
            rec.skip_many(["retrieve", "pool", "rerank"], "chat route: no retrieval")
            reply = await chat(query, session_id, rec, on_token)
        else:
            variants = await plan
            reply = await ask(query, factory, expansion_n, session_id, variants, rec, on_token)
        rec.finish(answer=reply.text, reasoning=reply.reasoning)
        reply.trace = rec.trace
        _persist(rec.trace, store)
        return kind, reply
    except BaseException as exc:
        rec.fail_running(exc)
        rec.finish(error=f"{type(exc).__name__}: {exc}")
        _persist(rec.trace, store)
        with contextlib.suppress(Exception):
            exc.trace = rec.trace  # type: ignore[attr-defined]
        raise
    finally:
        if plan is not None and not plan.done():
            plan.cancel()
        if plan is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await plan
