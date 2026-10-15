"""OCR fallback for scanned PDFs and images (no system binaries needed).

PyPDFLoader only reads a PDF's text layer; a scanned PDF has none, so every
page came back blank. Here, pages with (almost) no text are rendered with
PyMuPDF and read by RapidOCR (ONNX models bundled in the pip wheel, so no
Tesseract / Poppler install is required, which matters on Windows).

    pip install pymupdf rapidocr-onnxruntime
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable

from langchain_core.documents import Document

from .config import OCR_DPI, OCR_MIN_PAGE_CHARS, OCR_MIN_SCORE

logger = logging.getLogger(__name__)

# on_progress(done_pages, total_pages, message)
OcrProgress = Callable[[int, int, str], None]

_engine = None
_engine_lock = threading.Lock()
INSTALL_HINT = "OCR needs two packages: pip install pymupdf rapidocr-onnxruntime"


def _get_engine():
    global _engine
    with _engine_lock:
        if _engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
            except ImportError as exc:
                raise RuntimeError(f"{INSTALL_HINT} ({exc})") from exc
            logger.info("Loading OCR engine (RapidOCR)")
            _engine = RapidOCR()
        return _engine


def _import_pymupdf():
    try:
        import pymupdf  # PyMuPDF >= 1.24.3
        return pymupdf
    except ImportError:
        try:
            import fitz  # older name
            return fitz
        except ImportError as exc:
            raise RuntimeError(f"{INSTALL_HINT} ({exc})") from exc


def ocr_array(img) -> str:
    """OCR one RGB numpy image; returns text with one line per detected line."""
    result, _ = _get_engine()(img)
    if not result:
        return ""
    lines = [(box, text, score) for box, text, score in result
             if text and text.strip() and float(score) >= OCR_MIN_SCORE]
    # Reading order: top-to-bottom, then left-to-right; rows within half a
    # line-height of each other count as the same row.
    def top(b): return min(p[1] for p in b)
    def left(b): return min(p[0] for p in b)
    def height(b): return max(p[1] for p in b) - min(p[1] for p in b)
    lines.sort(key=lambda t: (round(top(t[0]) / max(height(t[0]) * 0.5, 1.0)), left(t[0])))
    return "\n".join(t[1].strip() for t in lines)


def _page_to_array(page, dpi: int):
    import numpy as np
    pix = page.get_pixmap(matrix=_matrix(page, dpi), alpha=False)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].copy()


def _matrix(page, dpi: int):
    mod = _import_pymupdf()
    z = dpi / 72.0
    return mod.Matrix(z, z)


def load_pdf_with_ocr(path: str, on_progress: OcrProgress | None = None) -> list[Document]:
    """One Document per page. Text-layer pages are used as-is; pages with fewer
    than OCR_MIN_PAGE_CHARS characters are OCR'd (metadata `ocr=True`)."""
    from langchain_community.document_loaders import PyPDFLoader

    try:
        pages = PyPDFLoader(path).load()
    except Exception:
        logger.warning("PyPDFLoader failed on %s; falling back to OCR for all pages", path, exc_info=True)
        pages = []

    need: list[int] = []
    if pages:
        need = [i for i, d in enumerate(pages) if len((d.page_content or "").strip()) < OCR_MIN_PAGE_CHARS]
    if not need and pages:
        return pages

    pymupdf = _import_pymupdf()
    docs: list[Document] = list(pages)
    with pymupdf.open(path) as pdf:
        if not docs:  # text layer unreadable: build one doc per page
            docs = [Document(page_content="", metadata={"source": path, "page": i}) for i in range(len(pdf))]
            need = list(range(len(pdf)))
        total = len(need)
        logger.info("%s: %d/%d page(s) have no text layer; running OCR", Path(path).name, total, len(docs))
        for n, i in enumerate(need, 1):
            if on_progress:
                on_progress(n - 1, total, f"OCR page {n}/{total}")
            text = ocr_array(_page_to_array(pdf[i], OCR_DPI))
            docs[i].page_content = text
            docs[i].metadata["ocr"] = True
        if on_progress:
            on_progress(total, total, f"OCR finished ({total} pages)")
    return docs


def load_image_with_ocr(path: str, on_progress: OcrProgress | None = None) -> list[Document]:
    """OCR an image file (all frames for multi-page TIFF)."""
    try:
        import numpy as np
        from PIL import Image, ImageSequence
    except ImportError as exc:
        raise RuntimeError(f"{INSTALL_HINT} (and pillow) ({exc})") from exc

    docs: list[Document] = []
    with Image.open(path) as im:
        frames = list(ImageSequence.Iterator(im))
        for n, frame in enumerate(frames, 1):
            if on_progress:
                on_progress(n - 1, len(frames), f"OCR image {n}/{len(frames)}")
            text = ocr_array(np.array(frame.convert("RGB")))
            docs.append(Document(page_content=text, metadata={"source": path, "page": n - 1, "ocr": True}))
    if on_progress:
        on_progress(len(docs), len(docs), "OCR finished")
    return docs
