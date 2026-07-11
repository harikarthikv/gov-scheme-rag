# Eligibility Matching Workflow

Once candidate schemes are retrieved from ChromaDB, they must be rigorously evaluated. Vector similarity alone only proves *semantic relevance*, not *legal eligibility*. A 30-year-old male might retrieve a maternal health scheme because both involve "healthcare in rural areas".

The Eligibility Matching chain (`server/matching/eligibility_chain.py`) solves this.

## 1. Top-K Evaluation
ChromaDB returns the top 10 most semantically relevant schemes based on the structured profile query. 
These 10 schemes (with their full text chunks) and the user's extracted JSON profile are injected into the Eligibility Prompt.

## 2. LLM Evaluation
The primary LLM (Gemini) is instructed to act as a strict eligibility evaluator. 
It compares the user's exact profile attributes against the strict criteria documented in the scheme chunks.

It outputs a structured JSON array filtering down the top 10 list into only the schemes where the user is:
- `"eligible"`: All known criteria are met.
- `"partial"`: Most criteria are met, but some unmentioned criteria (e.g., "Must hold a BPL card") need verification.

## 3. Formatting the Output
For every eligible/partial scheme, the LLM must extract:
- `name`: Scheme Name.
- `reason`: A concise, 1-sentence personalized explanation of *why* the user qualifies (e.g., "As a 30-year-old farmer in Tamil Nadu earning under 1 Lakh, you meet the demographic criteria.").
- `key_benefit`: The single most valuable benefit.
- `url`: The application link (pulled directly from the ChromaDB metadata).

## 4. Fallback Tiers

### Tier 1: Local Ollama Reasoning
If Gemini fails, the top 3 (trimmed) schemes are sent to the local Ollama `qwen3:0.6b` model. Since the local model has a smaller context window, we aggressively truncate the scheme documents to 400 characters to prevent OOM errors and timeouts.

### Tier 2: Raw ChromaDB (Direct Search)
If both Gemini and Ollama are unavailable, or if the LLM produces invalid JSON, the system completely bypasses the eligibility evaluation phase. 
It falls back to `raw_chromadb_search()`, instantly returning the raw metadata of the top 5 schemes retrieved from the vector store with a default `partial` status and a warning that eligibility is unverified.
