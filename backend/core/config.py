from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

_THIS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = _THIS_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent
load_dotenv(PROJECT_ROOT / ".env")

# Data paths
RAW_DOCS_DIR: Path = PROJECT_ROOT / "data" / "raw_docs"
QDRANT_PATH: Path = Path(os.getenv("QDRANT_PATH", str(PROJECT_ROOT / "data" / "qdrant_db")))

# Resolve relative QDRANT_PATH against project root
if not QDRANT_PATH.is_absolute():
    QDRANT_PATH = PROJECT_ROOT / QDRANT_PATH

# Qdrant collection
COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION", "vda5050_baseline")

# Embedding model
EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")

# LLM  (Groq)
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "llama-3.3-70b-versatile")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))

# Chunking defaults (Phase 1 — naive baseline)
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))

# Retriever defaults
RETRIEVER_K: int = int(os.getenv("RETRIEVER_K", "3"))

# Backend server
BACKEND_HOST: str = os.getenv("BACKEND_HOST", "127.0.0.1")
BACKEND_PORT: int = int(os.getenv("BACKEND_PORT", "8000"))
