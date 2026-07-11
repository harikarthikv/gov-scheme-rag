# Project Summary & Elevator Pitches

This document contains structured summaries of the project, designed to be used during HR screens, technical interviews, and networking events.

## The 30-Second Elevator Pitch
*Best for: HR screening rounds, networking events, career fairs.*

"I built an AI-powered Government Scheme Finder. It solves a massive accessibility problem in India where citizens struggle to find schemes they qualify for due to complex bureaucratic language and fragmented portals. Instead of filling out massive forms, users simply chat with the AI in natural language. The system extracts their demographic profile in the background, runs a semantic search against a local vector database of government schemes, and uses a secondary LLM to rigorously evaluate eligibility before streaming a personalized response back."

## The 2-Minute Explanation
*Best for: "Tell me about your portfolio project" in technical rounds.*

"My capstone project is a Retrieval-Augmented Generation (RAG) assistant designed to democratize access to Indian government schemes.

The core problem is that citizens don't know what they don't know. Forms are tedious. So, I built a conversational UI using React and Vite. When a user says something like 'I'm a 30-year-old farmer from Tamil Nadu,' the FastAPI backend sends that to a Gemini LLM to perform Profile Extraction. It structures the text into a JSON demographic profile.

That profile is then embedded using a local, multilingual sentence-transformer model and queried against a persistent ChromaDB instance, which holds the deduplicated scheme data. 

Because vector search only guarantees semantic relevance, not legal eligibility, I built a secondary LLM chain. This chain takes the top 10 retrieved schemes and forces the LLM to strictly evaluate the user's profile against the scheme's criteria. It filters the list and extracts the single most valuable benefit for the user.

Finally, the backend streams the conversational response back to the React client using Server-Sent Events (SSE) for a zero-latency feel. To guarantee uptime, I also implemented a fully local fallback using the Ollama Qwen model in case the Gemini API goes down."

## The 5-Minute Technical Walkthrough
*Best for: Deep-dive architecture rounds.*

*(Use this structure)*
1. **The Architecture**: "It's a decoupled architecture. React SPA on the frontend, FastAPI on the backend, SQLite for session state, and ChromaDB for vector storage."
2. **The Data Pipeline**: "I pulled raw JSON datasets from Hugging Face, wrote a Python script to deduplicate them, and synthesized them into dense, single-chunk representations per scheme. I used a local multilingual embedding model so it natively handles queries in Hindi or Hinglish."
3. **The SSE Streaming**: "I didn't want the user waiting 10 seconds for the LLM. So, I implemented Server-Sent Events. The FastAPI endpoint yields the metadata first so the frontend sidebars populate instantly, and then streams the conversational tokens as they generate."
4. **The Fallback Resilience**: "Production AI fails. So I wrote a fallback tier. If Gemini hits a 429 Rate Limit, it retries. If it throws a 400 Invalid Key or 503 Outage, my code catches the exception, pings a local Ollama daemon, and routes the prompt to a local Qwen model. If even Ollama is down, it gracefully degrades to a raw ChromaDB semantic search. The user never sees a 500 Server Error."

---

## Key Achievements (Quantifiable Impact)
Use these statistics when discussing the project to show a focus on metrics and performance:

- **Zero-Latency Feel**: Implemented Server-Sent Events (SSE) to reduce perceived response latency from ~8 seconds to < 500ms (Time to First Token).
- **Multilingual Support**: Replaced external API embeddings with a local `paraphrase-multilingual-MiniLM-L12-v2` model, enabling semantic search across 50+ languages with 0 API cost.
- **100% Uptime Architecture**: Engineered a 3-tier fallback system (Gemini -> Local Ollama -> Raw Vector Search) ensuring user queries are never dropped due to cloud provider outages.
- **Precision Matching**: Implemented a secondary LLM "Eligibility Chain" that reduced hallucinated/false-positive scheme recommendations by strictly evaluating vector results against extracted user JSON profiles.
