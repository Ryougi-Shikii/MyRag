"""Load files from disk, keep the vector store in sync, maintain the registry."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker

from .ocr import load_image_with_ocr, load_pdf_with_ocr
from .config import (
    BM25_CACHE_FILE, CHUNK_SCHEMA, DOC_FOLDER, INDEX_BATCH_SIZE, REGISTRY_FILE,
    get_vector_store,
)

logger = logging.getLogger(__name__)

# One writer at a time: sync, delete and upload-saving all touch the registry.
_LOCK = threading.RLock()

# progress_cb receives a dict: {"stage", "file", "index", "total", "fraction", "message"}
ProgressCb = Callable[[dict], None]
STAGES = ("hashing", "loading", "chunking", "embedding", "done")

# ---------------------------------------------------------------- loaders
LOADERS = {
    ".pdf":  lambda p: None,   # handled in load_any (text layer first, OCR fallback)
    ".docx": lambda p: Docx2txtLoader(p),
    ".txt":  lambda p: TextLoader(p, encoding="utf-8", autodetect_encoding=True),
    ".md":   lambda p: UnstructuredMarkdownLoader(p),
    ".html": lambda p: UnstructuredHTMLLoader(p),
    ".htm":  lambda p: UnstructuredHTMLLoader(p),
    ".png":  lambda p: None,   # OCR (see load_any)
    ".jpg":  lambda p: None,   # OCR (see load_any)
    ".jpeg": lambda p: None,   # OCR (see load_any)
    ".tiff": lambda p: None,   # OCR (see load_any)
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff"}

# Extensions without the dot, e.g. for st.file_uploader(type=...).
UPLOAD_TYPES = sorted(e.lstrip(".") for e in LOADERS)


def load_any(path_str: str, on_progress=None) -> list[Document]:
    """Load a file into Documents. PDFs use their text layer and fall back to
    OCR for scanned pages; images are OCR'd. `on_progress(done, total, msg)`
    reports OCR progress."""
    ext = Path(path_str).suffix.lower()
    if ext not in LOADERS:
        raise ValueError(f"Unsupported file type: {ext}")
    if ext == ".pdf":
        return load_pdf_with_ocr(path_str, on_progress)
    if ext in IMAGE_EXTS:
        return load_image_with_ocr(path_str, on_progress)
    return LOADERS[ext](path_str).load()


# ---------------------------------------------------------------- registry
def _compute_file_hash(file_path: Path, chunk_size: int = 65536) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_registry(path: str = REGISTRY_FILE) -> dict:
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            logger.warning("Registry at %s is corrupt; starting fresh", path)
            return {}
    return {}


def _save_registry(registry: dict, path: str = REGISTRY_FILE) -> None:
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_text(json.dumps(registry, indent=4))
    tmp.replace(path)


def _doc_folder() -> Path:
    p = Path(DOC_FOLDER)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------- uploads
def _upload_name_and_bytes(item: Any) -> tuple[str, bytes]:
    """Accept a Streamlit UploadedFile, a (name, bytes) pair, or a Path."""
    if isinstance(item, (str, Path)):
        p = Path(item)
        return p.name, p.read_bytes()
    if isinstance(item, tuple) and len(item) == 2:
        return str(item[0]), bytes(item[1])
    name = getattr(item, "name")
    if hasattr(item, "getvalue"):
        return name, item.getvalue()
    return name, item.read()


def save_uploads(files: Iterable[Any]) -> list[Path]:
    """Write uploaded files into DOC_FOLDER and return their paths.

    Unsupported types are skipped (logged). An existing file of the same name
    is replaced; the next sync notices the changed hash and re-indexes it.
    Writes are atomic (temp file + rename), so a crash never leaves a
    half-written document for the indexer to pick up.
    """
    saved: list[Path] = []
    with _LOCK:
        folder = _doc_folder()
        for item in files:
            name, data = _upload_name_and_bytes(item)
            safe = Path(name).name.strip()
            if not safe or safe.startswith("."):
                logger.warning("Skipping upload with unusable name %r", name)
                continue
            if Path(safe).suffix.lower() not in LOADERS:
                logger.warning("Skipping unsupported upload %r", safe)
                continue
            target = folder / safe
            tmp = folder / f".{safe}.part"
            tmp.write_bytes(data)
            tmp.replace(target)
            saved.append(target)
    return saved


# ---------------------------------------------------------------- listing
def list_indexed() -> list[dict]:
    """One dict per document on disk: name, type, size, chunks, status.

    status is "indexed", "changed" (file differs from what was indexed) or
    "pending" (never indexed). `chunks` is None for entries indexed before
    chunk counts were recorded; they fill in after the next re-index.
    """
    with _LOCK:
        registry = _load_registry()
        out: list[dict] = []
        for p in sorted(_doc_folder().iterdir(), key=lambda x: x.name.lower()):
            if not p.is_file() or p.suffix.lower() not in LOADERS:
                continue
            st = p.stat()
            entry = registry.get(str(p.resolve()))
            if entry is None:
                status = "pending"
            elif ("mtime_ns" in entry
                  and (entry.get("size") != st.st_size or entry.get("mtime_ns") != st.st_mtime_ns)):
                status = "changed"
            else:
                status = "indexed"
            out.append({
                "name": p.name,
                "type": p.suffix.lstrip(".").lower(),
                "size": st.st_size,
                "chunks": (entry or {}).get("chunks"),
                "status": status,
            })
        return out


# ---------------------------------------------------------------- sync
def sync_documents(
        vector_store: Chroma | None = None,
        progress_cb: ProgressCb | None = None,
    ) -> bool:
    """Sync DOC_FOLDER with the vector store. Returns True if the DB changed.

    Crash-safe: a file's old vectors are removed only after it has been loaded
    and chunked successfully, and the registry entry is committed per file
    only after its chunks were added; a failed file is retried next run.

    Fast path: the file hashes are compared BEFORE the vector store (and
    therefore the embedding model) is opened. On a no-op run this returns
    without paying the ~20s embedding model load.

    `progress_cb(event)` reports the stages hashing, loading, chunking,
    embedding and done (plus "failed" for a file that errored), each with the
    file name, its position, and an overall `fraction` in [0, 1]. It is called
    from the calling thread, so it is safe to run this in a worker thread.
    """
    if callable(vector_store) and progress_cb is None:  # sync_documents(cb)
        vector_store, progress_cb = None, vector_store
    with _LOCK:
        return _sync(vector_store, progress_cb)


def _sync(vector_store: Chroma | None, progress_cb: ProgressCb | None) -> bool:
    def report(stage: str, file: str | None = None, index: int = 0, total: int = 0,
               fraction: float = 0.0, message: str = "") -> None:
        if progress_cb is None:
            return
        try:
            progress_cb({
                "stage": stage, "file": file, "index": index, "total": total,
                "fraction": max(0.0, min(1.0, fraction)), "message": message,
            })
        except Exception:
            logger.exception("progress callback failed")

    registry = _load_registry()

    current_files = [
        p.resolve()
        for p in _doc_folder().iterdir()
        if p.is_file() and p.suffix.lower() in LOADERS
    ]
    current_paths = {str(p): p for p in sorted(current_files)}
    deleted_files = set(registry.keys()) - set(current_paths)

    # ---- hashing: decide whether any real work is needed (no model load yet) ----
    todo: list[tuple[str, Path, str]] = []
    n_files = len(current_paths)
    for i, (path_str, path) in enumerate(current_paths.items(), 1):
        report("hashing", path.name, i, n_files, 0.0, f"Hashing {path.name}")
        try:
            file_hash = _compute_file_hash(path)
        except OSError:
            logger.exception("[sync] cannot hash %s", path)
            report("failed", path.name, i, n_files, 0.0, f"Cannot read {path.name}")
            continue
        entry = registry.get(path_str)
        if (entry is None
                or entry.get("hash") != file_hash
                or entry.get("schema") != CHUNK_SCHEMA):
            todo.append((path_str, path, file_hash))

    if not deleted_files and not todo:
        logger.info("Nothing to sync; skipping vector store open")
        report("done", None, 0, 0, 1.0, "Index already up to date")
        return False

    # ---- real work ahead: open the store (loads embedding model once) ----
    if vector_store is None:
        vector_store = get_vector_store()
    embeddings = vector_store._embedding_function

    changed = False
    if deleted_files:
        logger.info("Removing %d deleted file(s) from index", len(deleted_files))
        for df in deleted_files:
            vector_store.delete(where={"source_file": df})
            del registry[df]
        _save_registry(registry)
        changed = True

    splitter = None
    total = len(todo)
    for i, (path_str, path, file_hash) in enumerate(todo):
        def frac(within: float) -> float:
            return (i + within) / total

        try:
            logger.info("Ingesting %s", path.name)
            report("loading", path.name, i + 1, total, frac(0.0), f"Loading {path.name}")
            def _ocr_progress(done: int, n: int, msg: str, _i=i, _name=path.name) -> None:
                report("loading", _name, _i + 1, total, (_i + 0.15 * done / max(n, 1)) / total, f"{_name}: {msg}")

            file_docs = load_any(path_str, _ocr_progress)
            stat = os.stat(path_str)
            for doc in file_docs:
                doc.metadata.update({
                    "source_file": path_str,
                    "file_name": path.name,
                    "doc_type": path.suffix.lstrip(".").lower(),
                    "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                })

            report("chunking", path.name, i + 1, total, frac(0.15), f"Chunking {path.name}")
            if splitter is None:
                splitter = SemanticChunker(
                    embeddings=embeddings,
                    breakpoint_threshold_type="percentile",
                    breakpoint_threshold_amount=90,
                )
            chunks = splitter.split_documents(file_docs) if file_docs else []
            chunks = [c for c in chunks if c.page_content and c.page_content.strip()]
            if not chunks:
                # Nothing to search: a scanned/image-only PDF without OCR, an
                # image with no text, an empty file. Fail loudly instead of
                # registering a file whose vectors are all blank.
                raise ValueError(
                    "no extractable text, even after OCR (blank pages, or an image "
                    "the OCR engine could not read; try a higher OCR_DPI in config.py)"
                )

            # Loading and chunking succeeded: only now touch the old vectors.
            # (Always delete by source, so a retry after a half-finished add
            # can never leave duplicates.)
            vector_store.delete(where={"source_file": path_str})

            n = len(chunks)
            for start in range(0, n, INDEX_BATCH_SIZE):
                report("embedding", path.name, i + 1, total,
                       frac(0.4 + 0.6 * start / max(n, 1)),
                       f"Embedding {path.name} ({start}/{n} chunks)")
                vector_store.add_documents(chunks[start:start + INDEX_BATCH_SIZE])
            logger.info("Added %d chunk(s) for %s", n, path.name)

            registry[path_str] = {
                "hash": file_hash,
                "schema": CHUNK_SCHEMA,
                "chunks": n,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
            _save_registry(registry)   # per-file commit
            changed = True
            report("embedding", path.name, i + 1, total, frac(1.0), f"Indexed {path.name} ({n} chunks)")
        except Exception as exc:
            logger.exception("[sync] failed to process %s", path.name)
            report("failed", path.name, i + 1, total, frac(1.0),
                   f"{path.name}: {type(exc).__name__}: {exc}")

    if changed:
        _invalidate_derived_caches()
    report("done", None, total, total, 1.0, "Indexing finished" if changed else "Nothing indexed")
    return changed


# ---------------------------------------------------------------- delete
def delete_document(name: str) -> bool:
    """Remove a document from disk, the vector store and the registry.

    Order matters for crash safety: the file goes first. If we die afterwards,
    the registry still lists a path that no longer exists, which the next
    `sync_documents` treats as a deletion and finishes the cleanup.
    Returns True if anything was removed.
    """
    with _LOCK:
        folder = _doc_folder().resolve()
        target = (folder / Path(name).name).resolve()
        if target.parent != folder:
            raise ValueError(f"Refusing to delete outside the document folder: {name!r}")

        path_str = str(target)
        registry = _load_registry()
        existed = target.exists()
        if existed:
            target.unlink()
        if path_str in registry:
            get_vector_store().delete(where={"source_file": path_str})
            del registry[path_str]
            _save_registry(registry)
            _invalidate_derived_caches()
            return True
        return existed


def _invalidate_derived_caches() -> None:
    bm25_path = Path(BM25_CACHE_FILE)
    if bm25_path.exists():
        bm25_path.unlink()

    # Local import: rag imports indexing at module load, so a top-level
    # import here would create a cycle.
    from .rag import clear_retriever_cache
    from .config import clear_vector_store_cache

    clear_retriever_cache()
    clear_vector_store_cache()
