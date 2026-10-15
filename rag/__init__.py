"""RAG pipeline for the document Q&A assistant."""
import logging

from .indexing import (
    UPLOAD_TYPES, delete_document, list_indexed, save_uploads, sync_documents,
)
from .rag import (
    ask, chat, classify, forget_session, respond, Reply, RetrieverFactory,
)
from .trace import StepTrace, TraceRecorder, TraceStore, TurnTrace, default_store

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "ask", "chat", "classify", "respond", "Reply", "forget_session",
    "sync_documents", "save_uploads", "list_indexed", "delete_document",
    "UPLOAD_TYPES", "RetrieverFactory",
    "StepTrace", "TurnTrace", "TraceRecorder", "TraceStore", "default_store",
]
