# LLM Fallback Validation Report

## Overview
The Government Scheme Finder RAG application relies primarily on the Google Gemini API. To ensure high availability and robust system resilience, a multi-tier fallback architecture is implemented. This document serves as the formal validation report of the fallback mechanisms.

## Fallback Architecture
1. **Primary LLM:** Gemini (via Google GenAI SDK). Used for profile extraction, complex scheme matching, and natural language response generation.
2. **Secondary LLM (Tier 1 Fallback):** Local Ollama (`qwen3:0.6b`). Used if Gemini rate limits (429) or throws authentication/authorization (400, 403) errors.
3. **Tertiary Search (Tier 2 Fallback):** Raw ChromaDB similarity search. Used if both Gemini and the local Ollama daemon are unreachable.

## Validation Scenarios and Test Results

### Scenario 1: Gemini Offline, Ollama Online
- **Objective:** Verify the system gracefully degrades to the local `qwen3:0.6b` model when Gemini fails.
- **Methodology:** We manually invalidated the `GEMINI_API_KEY` in the `.env` file to force a 400 API_KEY_INVALID error from the Gemini SDK. The Ollama daemon was left running on `localhost:11434`. We ran the `test_fallback.py` script.
- **Results:**
  - **Detection:** The system immediately caught the Gemini authentication error.
  - **Transition:** `is_ollama_available()` returned `True`. The system seamlessly routed the profile extraction and matching prompts to the local Ollama instance.
  - **Output:** The system returned 3 matched schemes via ChromaDB and successfully streamed a coherent textual response from Ollama.
  - **Status:** **PASS** ✅

### Scenario 2: Gemini Offline, Ollama Offline
- **Objective:** Verify the system falls back to a raw database similarity search when no LLM processing is available.
- **Methodology:** With Gemini still disabled, we altered the `OLLAMA_BASE_URL` to point to a non-existent port (`11435`) and restarted the server. We then ran the `test_fallback.py` script.
- **Results:**
  - **Detection:** `is_ollama_available()` immediately failed and returned `False`.
  - **Transition:** The application aborted profile extraction entirely (returning an empty profile). It then performed a `raw_chromadb_search` using the raw user query string against the vector database.
  - **Output:** The system successfully retrieved 5 raw scheme matches from ChromaDB based purely on vector similarity. It streamed a hardcoded fallback string (`AI is temporarily rate-limited. Schemes above are from direct search. Please retry shortly.`).
  - **Status:** **PASS** ✅

## Conclusion
The application demonstrates robust fault tolerance. It handles complete cloud provider outages and local daemon crashes without dropping the user session or throwing 5xx Server Errors, guaranteeing that users will always receive scheme recommendations, regardless of upstream LLM availability.
