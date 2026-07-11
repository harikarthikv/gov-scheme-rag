# Government Scheme Finder — profile extractor
# Fixes applied:
#   - C4: was using undefined _re alias — now uses re throughout
#   - M2: is_ollama_available() and URL constants come from utils.llm

import json
import os
import re
import google.genai as genai
from dotenv import load_dotenv
import requests
import time
import logging
from utils.llm import is_ollama_available, OLLAMA_GENERATE_URL, GEMINI_MODEL, OLLAMA_MODEL

logger = logging.getLogger(__name__)
load_dotenv()
_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

EXTRACTION_PROMPT = """You are a profile extraction assistant for an Indian government scheme finder.

Read the conversation below and extract the user's profile. Users speak naturally — infer fields from context.

Examples of inference:
- "poor family" → annual_income is low (infer < 100000)
- "just finished 12th" → age ~18, education = "12th pass"
- "small farm" → occupation = "farmer"
- "my husband passed away" → marital_status = "widowed"
- "two kids" → family_size >= 3

Return ONLY a JSON object with exactly two keys:
- "profile": an object with any of these fields you can confidently extract or infer:
    age (integer),
    gender (string),
    state (string: full state name in English),
    occupation (string: farmer/student/self-employed/salaried/unemployed/etc.),
    annual_income (integer in rupees),
    caste_category (string: one of General / OBC / SC / ST),
    disability (boolean),
    education (string),
    marital_status (string: single/married/widowed/divorced),
    family_size (integer)
- "missing": a list of field names that would most meaningfully improve scheme matching.
  Prioritise: state, occupation, age, annual_income, caste_category.
  Only include fields that are genuinely unknown AND would change which schemes are shown.
  Return an empty list if the profile is rich enough for a good match.

Do not include fields you are not confident about.
Do not add any explanation. Return only the JSON object.

Conversation:
{conversation}"""


def extract_profile(messages: list[dict]) -> dict:
    """
    messages: [{"role": "user"|"assistant", "content": "..."}]
    returns:  {"profile": {...}, "missing": [...]}
    """
    conversation_text = ""
    for m in messages:
        role = "User" if m["role"] == "user" else "Assistant"
        conversation_text += f"{role}: {m['content']}\n"

    prompt = EXTRACTION_PROMPT.format(conversation=conversation_text.strip())

    raw = None
    try:
        response = _client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        raw = response.text.strip()
    except Exception as e:
        e_str = str(e)
        if "429" in e_str or "RESOURCE_EXHAUSTED" in e_str:
            delay_match = re.search(r"retryDelay.*?(\d+)s", e_str)   # C4 fix: was _re
            wait = int(delay_match.group(1)) + 2 if delay_match else 25
            logger.info("Gemini 429 in extract_profile, waiting %ds then retrying...", wait)
            time.sleep(wait)
            try:
                response = _client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
                raw = response.text.strip()
            except Exception:
                raw = None
        # raw stays None — fall through to Ollama

        if raw is None:
            if not is_ollama_available():
                logger.warning("Ollama not reachable — returning empty profile.")
                return {"profile": {}, "missing": ["age", "state", "occupation"]}
            logger.info("Gemini failed in extract_profile, falling back to Ollama: %s", e_str[:80])
            try:
                res = requests.post(OLLAMA_GENERATE_URL, json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 512, "temperature": 0.1},
                }, timeout=60)
                raw = res.json().get("response", "").strip()
            except Exception as oe:
                logger.error("Ollama fallback failed: %s", oe)
                return {"profile": {}, "missing": ["age", "state", "occupation"]}

    # Strip Qwen3 <think>...</think> blocks and markdown fences
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        result = json.loads(raw)
        return {
            "profile": result.get("profile", {}),
            "missing": result.get("missing", []),
        }
    except json.JSONDecodeError:
        return {"profile": {}, "missing": ["age", "state", "occupation"]}
