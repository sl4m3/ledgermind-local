"""Projection implementations for the local LedgerMind service."""

from .fts import KnowledgeFTSProjection
from .gguf_vectorizer import GGUFVectorizer
from .git_audit import KnowledgeMarkdownGitAuditProjection, MarkdownGitAuditProjection
from .markdown import KnowledgeMarkdownProjection
from .vector import KnowledgeVectorProjection
from .vector_store import VectorProjectionStore
from .vectorizer import Vectorizer

__all__ = [
    "GGUFVectorizer",
    "KnowledgeFTSProjection",
    "KnowledgeMarkdownGitAuditProjection",
    "KnowledgeMarkdownProjection",
    "KnowledgeVectorProjection",
    "MarkdownGitAuditProjection",
    "VectorProjectionStore",
    "Vectorizer",
]
