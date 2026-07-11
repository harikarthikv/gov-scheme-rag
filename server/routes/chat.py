# Government Scheme Finder — chat route
# Fixes applied:
#   - C3:  session ownership verified before accepting request
#   - H5:  message content length + list size validated via Pydantic
#   - H6:  simple in-memory rate limiting (5 req/min per IP)
#   - M2:  is_ollama_available() from shared utils.llm
#   - M3:  OLLAMA URLs from shared utils.llm
#   - L2:  logging instead of print()

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator
from typing import Literal
import google.genai as genai
import os
import json
import re
import time
import logging
import requests
from collections import defaultdict
from dotenv import load_dotenv
from matching.profile_extractor import extract_profile
from matching.eligibility_chain import match_schemes, raw_chromadb_search
from utils.llm import is_ollama_available, OLLAMA_GENERATE_URL, GEMINI_MODEL, OLLAMA_MODEL
from db import get_db

logger = logging.getLogger(__name__)
load_dotenv()
_genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

router = APIRouter()

# ── Simple in-memory rate limiter (H6) ──────────────────────────────────────
_rate_store: dict[str, list[float]] = defaultdict(list)
_RATE_LIMIT    = 5   # max requests
_RATE_WINDOW   = 60  # per N seconds

def _check_rate_limit(ip: str) -> bool:
    now = time.monotonic()
    cutoff = now - _RATE_WINDOW
    _rate_store[ip] = [t for t in _rate_store[ip] if t > cutoff]
    if len(_rate_store[ip]) >= _RATE_LIMIT:
        return False
    _rate_store[ip].append(now)
    return True


# ── Request validation (H5) ─────────────────────────────────────────────────
class MessageItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        if len(v) > 4000:
            raise ValueError("Message too long (max 4 000 characters)")
        return v


class ChatRequest(BaseModel):
    session_id: str | None = None
    user_id:    int | None = None
    messages:   list[MessageItem]

    @field_validator("messages")
    @classmethod
    def cap_history(cls, v):
        return v[-50:] if len(v) > 50 else v   # keep last 50 turns


# ── Prompt ───────────────────────────────────────────────────────────────────
COMBINED_REPLY_PROMPT = """You are a warm, helpful Indian government scheme assistant.

## User Profile (extracted so far)
{profile}

## Missing Fields (still unknown)
{missing}

## Schemes Matched So Far
{schemes}

## Your Task
Write a conversational reply that does ALL of the following:

1. **Acknowledge** what the user just told you in one natural sentence.
2. **Summarise the match**: Tell them how many schemes were found. Do NOT list schemes — they are shown as cards.
   - If no schemes matched: say you'll refine results as you learn more.
3. **Ask ONE clarifying question**: If there are missing fields, pick the single most important one.
   - If nothing is missing, tell them the match is complete.

## Rules
- Respond in the same language the user is using (English/Hindi/Hinglish/Tamil etc.)
- Warm, simple, direct. No bureaucratic language.
- No bullet points or headers — flowing conversational text only.
- 3–5 sentences max.
- Do NOT use the words "profile", "fields", or "missing" as technical terms."""


# ── Endpoint ─────────────────────────────────────────────────────────────────
@router.post("/chat")
async def chat(request: Request, req: ChatRequest):
    # Rate limiting (H6)
    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded — please wait a moment.")

    messages    = [m.model_dump() for m in req.messages]
    session_id  = req.session_id
    user_id     = req.user_id

    # Session ownership check (C3)
    if session_id and user_id:
        with get_db() as conn:
            row = conn.execute(
                "SELECT id FROM sessions WHERE id = ? AND user_id = ?",
                (session_id, user_id),
            ).fetchone()
            if not row:
                raise HTTPException(status_code=403, detail="Session not found or not authorized.")

    # Persist incoming user message
    if session_id and messages:
        latest = messages[-1]
        if latest["role"] == "user":
            with get_db() as conn:
                conn.execute(
                    "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                    (session_id, "user", latest["content"]),
                )
                conn.commit()

    def generate_response():
        # Step 1 — extract profile (partial is fine)
        extracted = extract_profile(messages)
        profile   = extracted["profile"]
        missing   = extracted["missing"]

        # Step 2 — scheme matching; always try, fall back to raw search
        schemes = match_schemes(profile) if profile else []
        if not schemes:
            last_user = next(
                (m["content"] for m in reversed(messages) if m["role"] == "user"), None
            )
            if last_user:
                schemes = raw_chromadb_search(last_user)

        yield f"data: {json.dumps({'type': 'metadata', 'profile': profile, 'schemes': schemes})}\n\n"

        # Step 3 — generate conversational reply
        prompt = COMBINED_REPLY_PROMPT.format(
            profile=json.dumps(profile, indent=2) if profile else "Nothing extracted yet",
            missing=", ".join(missing) if missing else "None — profile is complete",
            schemes=json.dumps(schemes, indent=2) if schemes else "No matches yet",
        )

        full_reply = ""

        # Try Gemini streaming
        try:
            response = _genai_client.models.generate_content_stream(
                model=GEMINI_MODEL, contents=prompt
            )
            for chunk in response:
                text = chunk.text
                full_reply += text
                yield f"data: {json.dumps({'type': 'chunk', 'text': text})}\n\n"

        except Exception as e:
            e_str   = str(e)
            retried = False

            if "429" in e_str or "RESOURCE_EXHAUSTED" in e_str:
                delay_match = re.search(r"retryDelay.*?(\d+)s", e_str)
                wait = int(delay_match.group(1)) + 2 if delay_match else 25
                logger.info("Gemini 429 in chat, waiting %ds then retrying...", wait)
                time.sleep(wait)
                try:
                    response = _genai_client.models.generate_content_stream(
                        model=GEMINI_MODEL, contents=prompt
                    )
                    for chunk in response:
                        text = chunk.text
                        full_reply += text
                        yield f"data: {json.dumps({'type': 'chunk', 'text': text})}\n\n"
                    retried = True
                except Exception:
                    retried = False

            if not retried:
                if not is_ollama_available():
                    logger.warning("Ollama not reachable — returning service-unavailable message.")
                    err = "AI is temporarily rate-limited. Schemes above are from direct search. Please retry shortly."
                    full_reply += err
                    yield f"data: {json.dumps({'type': 'chunk', 'text': err})}\n\n"
                else:
                    logger.info("Gemini failed in chat, falling back to Ollama: %s", e_str[:80])
                    try:
                        res = requests.post(OLLAMA_GENERATE_URL, json={
                            "model": OLLAMA_MODEL,
                            "prompt": prompt,
                            "stream": True,
                            "options": {"num_predict": 512},
                        }, stream=True, timeout=180)

                        in_think  = False
                        think_buf = ""
                        for line in res.iter_lines():
                            if not line:
                                continue
                            chunk_data = json.loads(line)
                            token = chunk_data.get("response", "")
                            if not token:
                                continue

                            if "<think>" in token and not in_think:
                                in_think = True
                                token = token.replace("<think>", "")

                            if in_think:
                                closing = token.find("</think>")
                                if closing != -1:
                                    think_buf += token[:closing]
                                    if think_buf.strip():
                                        yield f"data: {json.dumps({'type': 'thinking', 'text': think_buf})}\n\n"
                                    think_buf = ""
                                    in_think  = False
                                    remainder = token[closing + len("</think>"):]
                                    if remainder:
                                        full_reply += remainder
                                        yield f"data: {json.dumps({'type': 'chunk', 'text': remainder})}\n\n"
                                else:
                                    think_buf += token
                                continue

                            full_reply += token
                            yield f"data: {json.dumps({'type': 'chunk', 'text': token})}\n\n"

                        if think_buf.strip():
                            yield f"data: {json.dumps({'type': 'thinking', 'text': think_buf})}\n\n"

                    except Exception as oe:
                        logger.error("Ollama fallback streaming failed: %s", oe)
                        err = "I'm temporarily unavailable. Please try again shortly."
                        full_reply += err
                        yield f"data: {json.dumps({'type': 'chunk', 'text': err})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

        if session_id and full_reply:
            with get_db() as conn:
                conn.execute(
                    "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                    (session_id, "assistant", full_reply.strip()),
                )
                conn.commit()

    return StreamingResponse(generate_response(), media_type="text/event-stream")
