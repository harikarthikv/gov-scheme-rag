# Resume Bullets

Copy and paste the bullet points that best fit your resume constraints.

## One-Line Versions
- Developed a full-stack AI assistant using React, FastAPI, and ChromaDB to match citizens with government schemes via RAG and Server-Sent Events.
- Engineered a RAG-powered scheme finder utilizing local multilingual embeddings and a 3-tier LLM fallback architecture to guarantee 100% uptime.

## Two-Line Versions
- Built an AI government scheme finder using React and FastAPI, leveraging a ChromaDB RAG pipeline to map conversational queries to structured demographic profiles.
- Implemented real-time token streaming via Server-Sent Events (SSE) and engineered a fault-tolerant architecture with seamless fallback to local Ollama LLMs during cloud API outages.

## Impact & Metrics Oriented (Recommended)
- **AI RAG Architecture:** Engineered a Retrieval-Augmented Generation pipeline using FastAPI and ChromaDB, parsing 180+ government schemes into dense vector chunks for semantic search.
- **Multilingual Semantic Search:** Integrated local `paraphrase-multilingual` embedding models, enabling cross-lingual semantic matching for 50+ languages with zero API latency.
- **Fault-Tolerant System Design:** Developed a multi-tier LLM fallback mechanism prioritizing local inference (Ollama Qwen3) during Google Gemini outages, ensuring uninterrupted service.
- **Low-Latency Streaming UX:** Built a React SPA consuming Server-Sent Events (SSE) from the backend, reducing perceived Time to First Token (TTFT) from 8s to <500ms.
- **Data Engineering:** Automated the ingestion, deduplication, and chunking of unstructured Hugging Face JSON datasets into a unified vector schema.
