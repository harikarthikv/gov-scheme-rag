"""
config.py
─────────
Central configuration for the Indian Government Schemes RAG pipeline.

Every tuneable constant lives here. Import from this module in all
other src/ files so there is a single source of truth.
"""

import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

# ── Load .env  ────────────────────────────────────────────────────────────────
# Resolve the project root relative to this file's location (src/config.py → project_root/)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
load_dotenv(find_dotenv(raise_error_if_not_found=False), override=True)

# Where ChromaDB will persist its index on disk
VECTOR_DB_PATH: Path = PROJECT_ROOT / "chroma_db"

# Directory containing the downloaded government scheme PDFs
DATA_DIR: Path = PROJECT_ROOT / "data" / "gov_myscheme" / "text_data"

# ── HuggingFace Dataset ───────────────────────────────────────────────────────
HF_DATASET_NAME: str = "shrijayan/gov_myscheme"

# ── Embedding model ───────────────────────────────────────────────────────────
# Runs fully locally via sentence-transformers — no API key required
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

# ── LLM (DeepSeek) ────────────────────────────────────────────────────────────
# DeepSeek API is OpenAI-compatible — uses ChatOpenAI with custom base_url.
# Default model; override by setting DEEPSEEK_MODEL in .env
MODEL_NAME: str = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

# Supported alternatives:
#   "deepseek-v4-flash" → DeepSeek V4 Flash (fast, general-purpose)
#   "deepseek-v4-pro"   → DeepSeek V4 Pro (higher quality)

DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"

TEMPERATURE: float = 0.1
MAX_TOKENS: int = 1024

# DeepSeek API key — read from .env; will raise at runtime if missing
DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")

# ── Text splitting ────────────────────────────────────────────────────────────
CHUNK_SIZE: int = 1000      # characters per chunk
CHUNK_OVERLAP: int = 200    # overlap between consecutive chunks

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K: int = 3              # number of chunks to retrieve per query

# Chunks whose relevance score is below this threshold are discarded.
# LangChain's similarity_search_with_relevance_scores returns values in [0, 1]
# where 1.0 = perfect match.
SIMILARITY_THRESHOLD: float = 0.3

# ── ChromaDB ──────────────────────────────────────────────────────────────────
COLLECTION_NAME: str = "gov_schemes"
