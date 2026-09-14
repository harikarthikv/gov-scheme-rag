"""Configuration shared by the command-line and Streamlit applications."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data" / "text_data"
VECTOR_DB_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "gov_schemes_local_dataset"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_NAME = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
TEMPERATURE = 0.1
MAX_TOKENS = 1024

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 3
SIMILARITY_THRESHOLD = 0.3
