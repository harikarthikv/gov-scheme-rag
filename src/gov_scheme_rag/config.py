"""Environment-backed configuration; importing this module never loads models."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

def _path(name: str, default: str) -> Path:
    path = Path(os.getenv(name, default))
    return path if path.is_absolute() else PROJECT_ROOT / path


DATA_DIR = _path("RAG_DATA_DIR", "data/text_data")
VECTOR_DB_PATH = _path("RAG_DB_PATH", "chroma_db")
COLLECTION_NAME = os.getenv("RAG_COLLECTION", "gov_schemes_local_dataset")

EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DEVICE = os.getenv("RAG_EMBEDDING_DEVICE", "cpu")
QUERY_PREFIX = os.getenv("RAG_QUERY_PREFIX", "")
DOCUMENT_PREFIX = os.getenv("RAG_DOCUMENT_PREFIX", "")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL_NAME = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
JUDGE_MODEL = os.getenv("RAG_JUDGE_MODEL", MODEL_NAME)
DEEPSEEK_THINKING = os.getenv("DEEPSEEK_THINKING", "disabled")
TEMPERATURE = float(os.getenv("RAG_TEMPERATURE", "0.1"))
MAX_TOKENS = int(os.getenv("RAG_MAX_TOKENS", "1024"))

CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
if not 0 <= CHUNK_OVERLAP < CHUNK_SIZE:
    raise ValueError("Require 0 <= RAG_CHUNK_OVERLAP < RAG_CHUNK_SIZE")


@dataclass(frozen=True)
class RetrievalConfig:
    mode: str = os.getenv("RAG_RETRIEVAL_MODE", "hybrid")
    top_k: int = int(os.getenv("RAG_TOP_K", "5"))
    candidate_k: int = int(os.getenv("RAG_CANDIDATE_K", "30"))
    dense_weight: float = float(os.getenv("RAG_DENSE_WEIGHT", "0.5"))
    rrf_constant: int = int(os.getenv("RAG_RRF_CONSTANT", "60"))
    similarity_threshold: float = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.3"))
    reranker_model: str = os.getenv("RAG_RERANKER_MODEL", "")

    def __post_init__(self) -> None:
        if self.mode not in {"baseline", "dense", "bm25", "hybrid"}:
            raise ValueError("Unknown retrieval mode")
        if self.top_k < 1 or self.candidate_k < self.top_k or self.rrf_constant < 1:
            raise ValueError("Require 1 <= top_k <= candidate_k and positive RRF constant")
        if not 0 <= self.dense_weight <= 1:
            raise ValueError("dense_weight must be in [0, 1]")
