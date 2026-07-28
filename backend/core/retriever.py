from __future__ import annotations
import logging
import pickle
from pathlib import Path
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from backend.core.config import COLLECTION_NAME, QDRANT_PATH, RETRIEVER_K
from backend.core.vectorstore import load_vectorstore

logger = logging.getLogger(__name__)

def get_retriever(
    collection_name: str = COLLECTION_NAME,
    k: int = RETRIEVER_K,
) -> BaseRetriever:
    fetch_k = 10
    vectorstore = load_vectorstore(collection_name=collection_name)
    qdrant_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": fetch_k},
    )

    # Setup Sparse Retriever (BM25)
    chunks_path = QDRANT_PATH.parent / "chunks.pkl"
    if not chunks_path.exists():
        logger.warning(
            "chunks.pkl not found at %s. Did you run ingestion? "
            "Falling back to pure dense retriever.", chunks_path
        )
        # Fallback to simple dense retriever if chunks.pkl is missing
        return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})

    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = fetch_k

    # Combine with EnsembleRetriever (Hybrid Search)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, qdrant_retriever],
        weights=[0.5, 0.5]
    )
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=model, top_n=k)

    # Wrap Ensemble with Compression Retriever
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )

    logger.info(
        "Hybrid Retriever ready — dense+sparse(k=%d), reranked to top_n=%d",
        fetch_k,
        k,
    )
    return compression_retriever
