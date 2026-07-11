# Project Defense: Decisions & Trade-offs

During interviews, Senior Engineers care less about *what* you built, and more about *why* you built it that way. This document provides strong, defensible arguments for the architectural decisions made in this project.

## 1. Why RAG instead of Fine-Tuning?
**Reason**: Government schemes change constantly. Budgets are updated, deadlines are extended, and eligibility criteria shift. 
**Defense**: If I fine-tuned a model on scheme data, the knowledge would be frozen in its weights. Updating it would require an expensive retraining job. With Retrieval-Augmented Generation (RAG), the LLM acts purely as a reasoning engine. The knowledge is stored in the database (ChromaDB). Updating a scheme is as simple as updating a row in the database, allowing the system to always query real-time, ground-truth data without retraining.

## 2. Why ChromaDB over Pinecone or pgvector?
**Reason**: Simplicity, locality, and cost.
**Defense**: Pinecone is a great managed service, but it introduces network latency and vendor lock-in. pgvector is extremely powerful but requires managing a PostgreSQL server instance. Since this is a portfolio application handling a static dataset of a few hundred schemes, ChromaDB was the perfect fit. It runs entirely locally via persistent storage, requires zero configuration, has zero monthly cost, and is easily distributable.

## 3. Why SSE (Server-Sent Events) over WebSockets?
**Reason**: Unidirectional data flow.
**Defense**: WebSockets are excellent for bidirectional, stateful, real-time communication (like a multiplayer game). However, LLM generation is inherently unidirectional: the client sends a single POST request, and the server streams a long response back. WebSockets would require complex connection management and heartbeat ping/pongs. SSE is built on top of standard HTTP, integrates easily with FastAPI's `StreamingResponse`, and natively handles unidirectional streams with much less overhead.

## 4. Why use a Local Multilingual Embedding Model?
**Reason**: Cost, speed, and inclusivity.
**Defense**: Originally, the project might use an API-based embedding model (like OpenAI or Gemini). However, API calls take time and cost money for every query. By switching to `paraphrase-multilingual-MiniLM-L12-v2` via `sentence-transformers`, the embeddings run directly on the server CPU in milliseconds. Furthermore, because it's a multilingual model, an Indian citizen can query the system in Hindi, and the model maps that query to the exact same vector space as the English scheme documentation, enabling cross-lingual semantic search natively.

## 5. Why Chat instead of a traditional Form?
**Reason**: Accessibility and UX.
**Defense**: Traditional portals require users to fill out 50-field forms because they rely on SQL `WHERE` clauses. If a user leaves a field blank, the query fails. An LLM-powered chat interface allows users to provide unstructured information naturally. The "Profile Extractor" chain infers data (e.g., inferring "low income" from the phrase "I'm a poor farmer") and explicitly asks follow-up questions only for the missing fields that actually matter.

## 6. Why implement an Ollama Fallback?
**Reason**: Fault tolerance and system resilience.
**Defense**: Cloud APIs go down, or they hit rate limits. If the Gemini API returns a 503 Service Unavailable, a standard application crashes and returns a 500 error to the user. I engineered a fallback system where a caught exception automatically routes the prompt to a local Ollama daemon running `qwen3:0.6b`. This ensures the application degrades gracefully, prioritizing availability over the slightly higher reasoning quality of the cloud model.

## 7. Why SQLite instead of PostgreSQL/MySQL?
**Reason**: MVP speed and zero-configuration portability.
**Defense**: The relational data requirements (Users, Sessions, Messages) are extremely simple. SQLite is a C-language library that implements a self-contained SQL database engine. By enabling WAL (Write-Ahead Logging) mode, it handles concurrent reads flawlessly. For an MVP, introducing a standalone Postgres container would overcomplicate the deployment without offering tangible benefits at this scale.
