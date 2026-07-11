# Interview Preparation Q&A

This document anticipates technical and HR questions an interviewer might ask about this project.

## AI & Machine Learning

**Q: Explain how RAG differs from Fine-Tuning.**
A: Fine-tuning updates the internal weights of a model, teaching it new patterns or styles. It is computationally expensive and the knowledge becomes stale the moment it's finished. RAG (Retrieval-Augmented Generation) leaves the model weights alone. Instead, it queries an external database for real-time information, injects that context into the prompt, and asks the model to reason over it. RAG is much better for factual, frequently changing data like government schemes.

**Q: How does semantic search actually work in your project?**
A: Semantic search compares the *meaning* of text rather than exact keywords. I use a sentence-transformer model to convert both the scheme documents and the user's profile into dense 384-dimensional mathematical vectors. ChromaDB then calculates the cosine similarity between the user's vector and the scheme vectors. Vectors that point in the same direction in the latent space are semantically similar.

**Q: How do you reduce LLM hallucinations?**
A: Two main ways. First, through RAG itself—by grounding the LLM's response strictly in retrieved context. Second, I built a two-step LLM chain. Instead of blindly trusting the vector database's top 10 results, I force the LLM to explicitly evaluate the user's demographic JSON against the scheme criteria JSON, filtering out false positives before generating the final text response.

## Software Engineering

**Q: Why did you choose FastAPI over Flask or Django?**
A: FastAPI is built from the ground up for asynchronous programming, which is critical for I/O bound tasks like calling LLM APIs or streaming responses. It also auto-generates Swagger documentation and uses Pydantic for extremely rigorous request validation, catching bad payloads before they even hit my route logic.

**Q: Explain how Server-Sent Events (SSE) work in your app.**
A: SSE allows the FastAPI backend to push data to the React frontend over a single, long-lived HTTP connection. I return a `StreamingResponse` from FastAPI that yields JSON chunks formatted with `data: ... \n\n`. On the frontend, `fetch` handles the stream using a `TextDecoder`, allowing me to update the React state token-by-token for a typing effect.

**Q: How do you handle failures if the Gemini API goes down?**
A: I built a multi-tier fallback architecture. A `try/except` block wraps the Gemini call. If it catches an authentication error or a 503 outage, it falls back to a local Ollama daemon running the `qwen3:0.6b` model. If Ollama is also unreachable, it bypasses the LLM entirely and performs a raw ChromaDB semantic search, returning the top 5 vector results so the user is never left with a blank screen.

## System Design (Scaling)

**Q: How would you scale this to 100,000 concurrent users?**
A: I would decouple the architecture. First, I'd move the state out of SQLite into a distributed Postgres database for users/sessions, and Redis for rate-limiting. Second, I'd move the LLM generation off the web thread into a distributed task queue (like Celery or RabbitMQ) pushing to WebSockets, or use a high-throughput API gateway. Finally, I'd migrate ChromaDB to a managed vector store like Pinecone or Weaviate to handle the search throughput.

**Q: How would you improve retrieval quality?**
A: I would add a Reranker. Currently, it's a bi-encoder similarity search. I would introduce a Cross-Encoder (like Cohere Rerank or BGE-Reranker) as a second stage. ChromaDB would fetch the top 50 cheap semantic results, and the Cross-Encoder would precisely re-score and rank the top 5 before sending them to the LLM.
