"""Projection implementations for the local LedgerMind service."""

from .dispatcher import ProjectionDispatcher
from .vectorizer import Vectorizer
from .gguf_vectorizer import GGUFVectorizer
from .fts import KnowledgeFTSProjection
from .vector_store import VectorProjectionStore
from .vector import KnowledgeVectorProjection
from .markdown import KnowledgeMarkdownProjection
from .git_audit import KnowledgeMarkdownGitAuditProjection, MarkdownGitAuditProjection

__all__ = [
    "ProjectionDispatcher",
    "Vectorizer",
    "GGUFVectorizer",
    "VectorProjectionStore",
    "KnowledgeMarkdownProjection",
    "KnowledgeFTSProjection",
    "KnowledgeVectorProjection",
    "KnowledgeMarkdownGitAuditProjection",
    "MarkdownGitAuditProjection",
]
