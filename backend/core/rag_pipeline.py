from __future__ import annotations
from backend.core.config import COLLECTION_NAME, RETRIEVER_K
from backend.core.generator import query_rag
from backend.core.ingestion import chunk_documents, load_documents
from backend.core.retriever import get_retriever
from backend.core.vectorstore import (
    get_embedding_model,
    load_vectorstore,
    upsert_documents,
)

__all__ = [
    "query_rag",
    "load_documents",
    "chunk_documents",
    "upsert_documents",
    "get_retriever",
    "get_embedding_model",
    "load_vectorstore",
    "COLLECTION_NAME",
    "RETRIEVER_K",
]
