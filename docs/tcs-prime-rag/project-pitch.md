# Project Pitch — Government Schemes RAG

> **30-second elevator pitch for TCS Prime interview**

---

## What Problem Does This Solve?

Indian citizens struggle to navigate **2,800+ government welfare schemes** spread across PDFs from 50+ ministries. Information is fragmented, PDFs are dense, and eligibility criteria are buried in legal language. Citizens either miss out on benefits they qualify for, or waste time on schemes they don't.

## What Did I Build?

A **Retrieval-Augmented Generation (RAG)** system that lets anyone ask natural-language questions about government schemes and get **accurate, sourced, hallucination-free answers** — as if they were talking to a government expert.

**Input:** "What schemes offer free healthcare for senior citizens in Maharashtra?"

**Output:** A grounded answer citing the exact scheme name, eligibility criteria, benefits, application process, and the source PDF page number — all verified against the official documents.

## Technical One-Liner

> A modular RAG pipeline ingesting 1,744 government scheme PDFs via PyMuPDF, indexed in ChromaDB with all-MiniLM-L6-v2 embeddings (384-dim), served by DeepSeek V4 Flash with temperature 0.1 and a strict no-hallucination prompt template.

## Key Numbers (for the whiteboard)

| Metric | Value |
|---|---|
| Source documents | 1,744 Indian Government Scheme PDFs (~2,876 total) |
| Embedding model | all-MiniLM-L6-v2 (384 dimensions, 90MB, runs on CPU) |
| Chunk strategy | RecursiveCharacterTextSplitter, 1000 chars, 200 overlap |
| Vector DB | ChromaDB, persistent on-disk, collection: `gov_schemes` |
| LLM | DeepSeek V4 Flash, temperature 0.1, max 1024 tokens |
| Retrieval | Top-3 cosine similarity, 0.3 minimum threshold |
| Confidence | Average similarity score of retrieved chunks |
| Latency | ~5-8s end-to-end (embedding + retrieval + LLM inference) |
| Storage | ~50-200MB on disk for ChromaDB index |

## Why This Matters for TCS Prime

- Demonstrates end-to-end ML system design (ingestion → indexing → retrieval → generation)
- Shows production-ready modular architecture (not a Jupyter notebook)
- Solves a real Indian citizen problem (government scheme awareness)
- Covers every RAG concept interviewers ask about (chunking, embeddings, vector DBs, similarity search, prompt engineering, hallucination reduction)

---

[Back to index →](./revision-sheet.md)
