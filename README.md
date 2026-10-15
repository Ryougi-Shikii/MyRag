# Document Q&A: Streamlit UI + pipeline observability

A Streamlit front end for the `rag/` package (BM25 + vector ensemble, cross-encoder rerank,
HyDE / query expansion, Fang Yuan persona) with a per-turn trace layer and an interactive
pipeline graph.

## Run

```bash
python -m venv .venv && source .venv/bin/activate     # Python 3.11+
pip install -r requirements.txt
cp .env.example .env                                   # add GROQ_API_KEY and GOOGLE_API_KEY
streamlit run app.py
```

Put documents in `document/` or upload them in the sidebar, then press **Index documents**.
The CLI (`python -m rag`) still works and shares the same index.

The graph loads `vis-network` from `unpkg.com` in your browser; no Python package is needed.

## Layout

| Path | Role |
|---|---|
| `rag/trace.py` | `StepTrace`, `TurnTrace`, `TraceRecorder` (live updates), `TraceStore` (SQLite `traces.db`) |
| `rag/rag.py` | Pipeline, now instrumented; `Reply.trace` carries the `TurnTrace` |
| `rag/ocr.py` | OCR fallback for scanned PDFs / images |
| `diagnose_index.py` | Per-file chunk / blank-chunk report of the stored index |
| `rag/indexing.py` | `save_uploads`, `sync_documents(progress_cb)`, `list_indexed`, `delete_document` |
| `rag/config.py` | All constants (`TRACES_DB`, `GRAPH_MAX_TURNS`, `INDEX_BATCH_SIZE`, ...) |
| `async_bridge.py` | One background event loop; avoids "event loop already running" on reruns |
| `graph_view.py` | Self-contained HTML/JS graph (vis-network) |
| `app.py` | Streamlit app |

## Scanned PDFs and images

PDF pages with no text layer, and image files (png/jpg/tiff), are read with OCR (RapidOCR via
PyMuPDF; `pip install pymupdf rapidocr-onnxruntime`, already in `requirements.txt`). No Tesseract
or Poppler install is needed. OCR is slow (roughly 1-3 s per page on CPU) and the indexing progress
bar shows `OCR page n/N`. Tune `OCR_DPI`, `OCR_MIN_PAGE_CHARS` and `OCR_MIN_SCORE` in `rag/config.py`.
If a file yields no text at all it is reported as failed instead of being indexed blank. Run
`python diagnose_index.py` to see per-file chunk and blank counts.

## Using the app

* **Chat tab**: streamed answers; under each one, an expander with sources (score, which query
  variants found the chunk) and, with *Show reasoning* on, the model's reasoning.
* **Pipeline Graph tab**: top row is the conversation spine (Current / Last / "n chats ago",
  last 8, older collapsed into one node). Click a chat to expand its pipeline; click a step for
  its exact input and output. Skipped steps are dashed grey, failed steps red. While a turn runs,
  the graph updates live (pending, running, done).
* Traces live in `traces.db`, so the graph and chat history survive restarts.
  **New chat** starts a new session; **Clear history** deletes the current session's traces and
  its in-memory thread.

## Tests

```bash
pytest tests/            # offline: trace recorder + SQLite round trip
python test_pipeline.py  # live: one retrieve turn + one chat turn, asserts the trace shape
                         # (needs API keys and at least one indexed document)
```

## Notes

* Conversation memory (`InMemorySaver`) is per process: after a restart the graph and chat
  transcript are restored from SQLite, but the model no longer remembers earlier turns.
* The graph page is embedded as same-origin HTML; all trace text is HTML-escaped before display.
* HyDE fix: per-variant results are now interleaved (round-robin) before the `MAX_RERANK_INPUT`
  cut, so HyDE hits are no longer always dropped.
