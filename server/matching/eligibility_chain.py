# Government Scheme Finder — eligibility matching chain
# Fixes applied:
#   - M1: double-checked locking for ChromaDB singleton
#   - M2: shared utils.llm helpers
#   - A3: raw fallback key_benefit now describes target group correctly
#   - L3: n_results capped to collection size

import json
import os
import re
import threading
import chromadb
import google.genai as genai
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import requests
import time
import logging
from utils.llm import is_ollama_available, OLLAMA_GENERATE_URL, GEMINI_MODEL, OLLAMA_MODEL

logger = logging.getLogger(__name__)
load_dotenv()
_genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "vectorstore", "chroma_db")

_chroma_client = None
_collection = None
_lock = threading.Lock()   # M1: guards singleton initialisation


def get_collection():
    global _chroma_client, _collection
    if _collection is None:
        with _lock:                       # M1: double-checked locking
            if _collection is None:
                _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
                embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="paraphrase-multilingual-MiniLM-L12-v2"
                )
                _collection = _chroma_client.get_collection("schemes", embedding_function=embedding_fn)
    return _collection


def _meta_to_card(meta: dict, reason: str = "Potentially relevant based on your query.") -> dict:
    """Convert raw ChromaDB metadata into a scheme card dict."""
    targets = meta.get("beneficiaries", "")
    key_benefit = (
        f"Targets: {targets} — open the scheme page for full benefit details."
        if targets else "Open the scheme page for benefit and eligibility details."
    )
    return {
        "name": meta.get("scheme_name") or meta.get("name") or "Government Scheme",
        "eligibility_status": "partial",
        "reason": reason,
        "key_benefit": key_benefit,   # A3: was using beneficiaries as if it were a benefit
        "url": meta.get("url", ""),
    }


def raw_chromadb_search(query_text: str, n: int = 5) -> list:
    """Search ChromaDB directly — no LLM evaluation.
    Used as a last-resort fallback when both Gemini and Ollama are unavailable.
    """
    try:
        collection = get_collection()
        safe_n = min(n, collection.count())   # L3: guard against empty collection
        if safe_n == 0:
            return []
        results = collection.query(query_texts=[query_text], n_results=safe_n)
        return [
            _meta_to_card(meta, "Retrieved as potentially relevant — eligibility not yet evaluated.")
            for meta in results["metadatas"][0]
        ]
    except Exception as e:
        logger.error("raw_chromadb_search failed: %s", e)
        return []


def build_query(profile: dict) -> str:
    parts = []
    if profile.get("age"):        parts.append(f"I am {profile['age']} years old")
    if profile.get("gender"):     parts.append(profile["gender"])
    if profile.get("state"):      parts.append(f"from {profile['state']}")
    if profile.get("occupation"): parts.append(f"working as a {profile['occupation']}")
    if profile.get("annual_income"):
        parts.append(f"with annual income of Rs {profile['annual_income']}")
    if profile.get("caste_category"):
        parts.append(f"belonging to {profile['caste_category']} category")
    if profile.get("disability"):  parts.append("with a disability")
    if profile.get("education"):   parts.append(f"with {profile['education']} qualification")
    if profile.get("marital_status"):
        parts.append(f"marital status: {profile['marital_status']}")
    if profile.get("family_size"): parts.append(f"family size of {profile['family_size']}")
    return ". ".join(parts) + ". Which government schemes am I eligible for?"


ELIGIBILITY_PROMPT = """You are a government scheme eligibility assistant for India.

User Profile:
{profile}

The following schemes were retrieved as potentially relevant:
{schemes}

For each scheme, evaluate whether the user is eligible based on their profile.

Return ONLY a JSON array. Each item must have:
  "name": scheme name (string)
  "eligibility_status": "eligible" or "partial"
  "reason": one sentence explaining why, referencing the user's profile
  "key_benefit": the single most valuable benefit for this specific user
  "url": the scheme URL

Include only schemes where the user is eligible or partially eligible.
Partial means they likely qualify but should verify one specific condition.
Return only the JSON array, no other text."""


def match_schemes(profile: dict) -> list:
    collection = get_collection()
    query = build_query(profile)

    safe_n = min(10, collection.count())   # L3
    if safe_n == 0:
        return []
    results = collection.query(query_texts=[query], n_results=safe_n)

    # Full prompt for Gemini (large context fine)
    schemes_text_full = ""
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        schemes_text_full += f"\n--- Scheme {i+1} ---\n{doc}\nURL: {meta['url']}\n"

    # Trimmed prompt for Ollama (0.6B model, limited context)
    schemes_text_trimmed = ""
    for i, (doc, meta) in enumerate(zip(results["documents"][0][:3], results["metadatas"][0][:3])):
        snippet = doc[:400].rstrip() + "..."
        schemes_text_trimmed += f"\n--- Scheme {i+1} ---\n{snippet}\nURL: {meta['url']}\n"

    prompt_full    = ELIGIBILITY_PROMPT.format(profile=json.dumps(profile, indent=2), schemes=schemes_text_full)
    prompt_trimmed = ELIGIBILITY_PROMPT.format(profile=json.dumps(profile, indent=2), schemes=schemes_text_trimmed)

    raw = None
    try:
        response = _genai_client.models.generate_content(model=GEMINI_MODEL, contents=prompt_full)
        raw = response.text.strip()
    except Exception as e:
        e_str = str(e)
        if "429" in e_str or "RESOURCE_EXHAUSTED" in e_str:
            delay_match = re.search(r"retryDelay.*?(\d+)s", e_str)
            wait = int(delay_match.group(1)) + 2 if delay_match else 25
            logger.info("Gemini 429 in match_schemes, waiting %ds then retrying...", wait)
            time.sleep(wait)
            try:
                response = _genai_client.models.generate_content(model=GEMINI_MODEL, contents=prompt_full)
                raw = response.text.strip()
            except Exception:
                raw = None

        if raw is None:
            if not is_ollama_available():
                logger.warning("Ollama not reachable — returning raw ChromaDB results.")
                return [_meta_to_card(m) for m in results["metadatas"][0][:5]]
            logger.info("Gemini failed in match_schemes, falling back to Ollama: %s", e_str[:80])
            try:
                res = requests.post(OLLAMA_GENERATE_URL, json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt_trimmed,
                    "stream": False,
                    "options": {"num_predict": 1024, "temperature": 0.1},
                }, timeout=120)
                raw = res.json().get("response", "").strip()
            except Exception as oe:
                logger.error("Ollama fallback failed: %s — returning raw ChromaDB results.", oe)
                return [_meta_to_card(m) for m in results["metadatas"][0][:5]]

    # Strip Qwen3 thinking tags and markdown fences
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        parsed = json.loads(raw)
        if not parsed:
            return [_meta_to_card(m) for m in results["metadatas"][0][:5]]
        return parsed
    except json.JSONDecodeError:
        return [_meta_to_card(m) for m in results["metadatas"][0][:5]]
