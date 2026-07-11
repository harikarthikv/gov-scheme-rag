"""
Shared LLM / inference utilities.
Centralises Ollama availability check and URL/model config so they are not
copy-pasted across every module.
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_TAGS_URL   = f"{OLLAMA_BASE_URL}/api/tags"

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:0.6b")


def is_ollama_available(timeout: int = 3) -> bool:
    """Fast health-check — returns False immediately if Ollama isn't reachable."""
    try:
        r = requests.get(OLLAMA_TAGS_URL, timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False
