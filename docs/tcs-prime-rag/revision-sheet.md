# One-Page Revision Sheet — TCS Prime RAG Interview

> **Print this. Review in 10 minutes before the interview.**

---

## Pipeline (Whiteboard)

```
PDFs (1,744) → PyMuPDF → Documents → Chunks (1000/200) → Embed (384-dim)
    → ChromaDB (persistent, UUIDs)
    → Query embed → Top-3 cosine → Threshold 0.3 → Prompt → DeepSeek V4 Flash (T=0.1)
    → {answer, sources, confidence_score}
```

---

## Key Numbers

| What | Value | Why |
|---|---|---|
| Chunk size | 1000 chars | Captures full scheme sections (~250 words) |
| Overlap | 200 (20%) | Prevents mid-sentence splits |
| Embedding dim | 384 | all-MiniLM-L6-v2, free, runs on CPU, 90MB |
| Top-K | 3 | ~3,000 chars context; more = noise ("lost in middle") |
| Threshold | 0.3 | Conservative; below = noise, above = related |
| Temperature | 0.1 | Near-deterministic → anti-hallucination |
| Max tokens | 1024 | Enough for scheme answers (~750 words) |
| LLM | DeepSeek V4 Flash | OpenAI-compatible, $0.0002/query |
| Collection | `gov_schemes` | ChromaDB persistent at `chroma_db/` |

---

## Why These Choices Matter (Top 8)

| Decision | Why | Alternative |
|---|---|---|
| **RAG vs Fine-tuning** | Fresh data (add PDFs → re-index), source citations, no GPU training | Fine-tuning for format control |
| **all-MiniLM-L6-v2** | Free, offline, CPU, 384-dim. 8 MTEB points less than OpenAI but $0 cost | OpenAI embeddings (better but $$, needs internet) |
| **ChromaDB** | Built-in persistence, metadata filtering, LangChain-native | FAISS (faster at scale, manual save/load) |
| **Cosine similarity** | Length-invariant (short query vs long doc) | L2 (magnitude-sensitive) |
| **Top-K=3** | Enough context, minimizes noise | 5 (more context but risk dilution) |
| **Threshold=0.3** | Filters noise, passes related schemes | 0.5 (misses related), 0.1 (passes noise) |
| **DeepSeek** | OpenAI-compatible API, 128K context, $0.0002/query | GPT-4o (better, 18× cost) |
| **Prompt rules** | 5 STRICT RULES: context-only, admit uncertainty, no fabrication, cite, citizen-friendly | Simpler prompt (higher hallucination) |

---

## Anti-Hallucination Defense (3 Layers)

```
Layer 1: Similarity threshold (0.3) → filter irrelevant chunks before LLM
Layer 2: Temperature (0.1) → near-deterministic output
Layer 3: Prompt rules → explicit "NEVER fabricate", exact fallback phrase
```

**If no chunks pass threshold:** Return fallback WITHOUT calling LLM. Zero hallucination risk.

---

## Code Map (8 Files)

| File | Responsibility | Key Class/Function |
|---|---|---|
| `src/config.py` | All constants, .env loading | `CHUNK_SIZE`, `TOP_K`, `MODEL_NAME` |
| `src/data_loader.py` | PDF → LangChain Documents | `SchemeDataLoader.load()` |
| `src/embedding.py` | Chunking + 384-dim vectors | `EmbeddingPipeline` |
| `src/vector_store.py` | ChromaDB create/load/persist | `SchemeVectorStore` |
| `src/retriever.py` | Top-K similarity + threshold | `Retriever.retrieve()` |
| `src/prompt.py` | ChatPromptTemplate | `get_prompt_template()` |
| `src/rag.py` | Full pipeline orchestrator | `RAGPipeline.query()` |
| `app.py` | Smart init + CLI loop | `main()` |

---

## Failure Modes (Top 4)

| Failure | What Happens | Fix |
|---|---|---|
| **Empty retrieval** | Fallback, no LLM call | Lower threshold, query expansion |
| **Irrelevant chunks** | LLM gets bad context | Hybrid BM25+semantic, re-ranking |
| **LLM hallucination** | Generated info not in context | Post-gen verification (LLM-as-judge) |
| **PDF parse failure** | Scheme silently skipped | OCR fallback (Tesseract) |

---

## Improvements (If Asked "What Next?")

1. **Hybrid search** (BM25 + semantic ensemble) → better recall
2. **Cross-encoder re-ranking** → Top-10 → re-rank to Top-3
3. **Streaming responses** → first token in <500ms
4. **Evaluation framework** → Recall@K, MRR, faithfulness metrics
5. **Multi-turn conversations** → chat history in prompt

---

## 30-Second Project Pitch

> "I built a RAG system that ingests 1,744 Indian government scheme PDFs, indexes them in ChromaDB using all-MiniLM-L6-v2 embeddings, and answers citizen questions through DeepSeek with strict hallucination controls. The system can tell you any scheme's eligibility, benefits, and application process — with exact source citations from the original PDFs. If it doesn't know, it admits it instead of guessing."

---

## Quick Answers to Common Follow-Ups

**Q: Why not use a larger embedding model?**
A: 384-dim captures enough semantics for government text. Larger models (768, 1024) add cost without proportional quality gain. I can swap to BGE-small (384-dim, better MTEB) as a drop-in replacement.

**Q: How do you know retrieval is working?**
A: I test with known schemes — "PM Kisan" should return pm_kisan.pdf chunks. I also test with out-of-domain queries — should trigger fallback. For production, I'd add Recall@K evaluation.

**Q: What if the PDFs have tables?**
A: PyMuPDF extracts table text but loses structure. For table-heavy PDFs, I'd use Unstructured.io or Camelot for table extraction.

**Q: How would you deploy this?**
A: FastAPI + Docker + Kubernetes. Embedding model as singleton. ChromaDB as persistent volume. DeepSeek API for LLM. Redis for query cache. Prometheus for monitoring.

**Q: What's your cost per month at 1000 queries/day?**
A: ~$6/month (DeepSeek API). Embedding and retrieval cost $0. The embedding model is a one-time 90MB download.

---

**Good luck! 🍀**
