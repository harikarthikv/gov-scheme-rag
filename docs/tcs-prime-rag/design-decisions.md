# Design Decisions — Architecture & Component Choices

> **What the interviewer wants:** You didn't just copy-paste. You made conscious trade-offs.

---

## 1. Why RAG Instead of Fine-Tuning?

### What I Implemented

A retrieval-augmented generation pipeline that keeps documents external and retrieves them at query time.

### Why

| Factor | RAG | Fine-Tuning |
|---|---|---|
| **Data freshness** | Add new PDFs → re-index (minutes) | Retrain entire model (hours/days) |
| **Hallucination control** | LLM constrained to retrieved context | Model can still generate from stale weights |
| **Transparency** | Cite exact source page | Black-box generation |
| **Cost** | Embedding (~$0) + LLM API (~$0.01/query) | GPU hours for training + hosting |
| **Domain specificity** | Works with any document | Requires labeled training data |
| **Traceability** | Every answer has source metadata | No source attribution |

### Interviewer Follow-Up

**Q: "When WOULD you fine-tune?"**

A: If I needed the model to generate in a specific format or tone that prompting cannot enforce. For example, if I needed the LLM to output structured JSON with guaranteed schema compliance for every scheme. But for the core problem of answering questions from documents, RAG is strictly superior.

---

## 2. Why `all-MiniLM-L6-v2` for Embeddings?

### What I Implemented

```python
HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
```

### Why

| Factor | all-MiniLM-L6-v2 | Alternatives |
|---|---|---|
| **Dimensions** | 384 | text-embedding-3-small: 512; BGE-large: 1024 |
| **Model size** | 90 MB | BGE-large: 1.3 GB |
| **Speed (CPU)** | ~1,500 sentences/sec | BGE-large: ~200 sentences/sec |
| **Quality (MTEB)** | 56.3 (avg) | BGE-large: 64.4; OpenAI: 64.6 |
| **Cost** | Free, runs locally | OpenAI: $0.02/1M tokens |
| **Offline** | Yes (local model) | No (API call required) |

### Trade-Off

I sacrificed ~8 MTEB points of retrieval quality for:
- **Zero API cost** (critical for government use — no budget)
- **Offline capability** (works without internet after model download)
- **CPU-only** (no GPU required — runs on any laptop)
- **Fast indexing** (1,744 PDFs in minutes vs hours)

### Interviewer Follow-Up

**Q: "If retrieval quality drops because of the smaller model?"**

A: Two mitigations: (1) `normalize_embeddings=True` uses L2 normalization so cosine similarity is accurate, (2) the `SIMILARITY_THRESHOLD=0.3` filters out noise. If quality is still insufficient, I'd swap to `BAAI/bge-small-en-v1.5` (384-dim but better MTEB) — it's a drop-in replacement since both are 384-dim.

---

## 3. Why Chunk Size = 1000, Overlap = 200?

### What I Implemented

```python
CHUNK_SIZE = 1000      # ~250 words
CHUNK_OVERLAP = 200    # 20% overlap
separators = ["\n\n", "\n", ". ", " ", ""]
```

### Why 1000 Characters?

Government scheme documents are semi-structured: each page typically contains one scheme section with ~200-500 words. 1000 characters (~250 words) captures most scheme descriptions without splitting across unrelated sections.

| Size | Pros | Cons |
|---|---|---|
| 256 chars | Precise retrieval | Loses context; scheme eligibility split across chunks |
| 512 chars | Good for Q&A | May miss cross-references |
| **1000 chars** | **Captures full scheme sections** | **Occasional context bleed** |
| 2048 chars | More context | Dilutes relevance; retrieves irrelevant content |
| 4096 chars | Avoids splitting | LLM context window wasted on noise |

### Why 20% Overlap?

200 characters of overlap prevents splitting sentences mid-way:
- A scheme description ending at position 980 will start again at position 800 in the next chunk
- This ensures the LLM never sees a truncated sentence

### Why `["\n\n", "\n", ". ", " ", ""]` Separators?

The separator hierarchy means the splitter:
1. **First tries** to split at paragraph boundaries (`\n\n`) — preserves document structure
2. If too long, **splits at newlines** (`\n`) — respects line-level structure
3. If still too long, **splits at sentences** (`. `) — semantic boundaries
4. If still too long, **splits at spaces** (` `) — word boundaries
5. Last resort: **character split** (`""`) — never fails

### Interviewer Follow-Up

**Q: "How did you choose 1000?"**

A: I profiled the average scheme section length in the PDFs — most are 200-500 words. 1000 chars captures the full section without over-fetching. I could improve this by using `SemanticChunker` from LangChain (split at embedding similarity drop-offs), but RecursiveCharacterTextSplitter is simpler and faster.

---

## 4. Why ChromaDB?

### What I Implemented

```python
Chroma.from_documents(
    documents=chunks,
    embedding=self.embeddings,
    collection_name="gov_schemes",
    persist_directory="./chroma_db",
    ids=[str(uuid.uuid4()) for _ in chunks],
)
```

### Why

| Factor | ChromaDB | FAISS | Pinecone (managed) | Weaviate |
|---|---|---|---|---|
| **Persistence** | Built-in (SQLite) | Manual save/load | Automatic | Automatic |
| **Setup complexity** | `pip install chromadb` | `pip install faiss-cpu` | Create account + API key | Docker container |
| **Metadata filtering** | Yes (WHERE clauses) | No (post-filter) | Yes | Yes |
| **Python integration** | Native (Chroma class) | NumPy arrays | REST API | GraphQL API |
| **Cost** | Free, local | Free, local | $$ per vector | Free (self-hosted) |
| **Distance metrics** | Cosine, L2, IP | L2, IP (cosine via norm) | Cosine, dot, euclidean | Cosine, L2, etc. |

### Why I Chose ChromaDB

1. **Persistence is built-in** — no manual save/load like FAISS
2. **Metadata filtering** — I store `scheme_name`, `filename`, `page` in metadata and can filter by ministry/state later
3. **LangChain-native** — `langchain-chroma` integration is seamless
4. **Single dependency** — `chromadb` package, no Docker, no cloud account
5. **UUID traceability** — every chunk has a unique ID for debugging

### Trade-Off

ChromaDB is slower than FAISS for brute-force search on very large datasets (>1M vectors). But for ~10K-50K chunks (our scale), the difference is negligible.

### Interviewer Follow-Up

**Q: "When would you switch to FAISS?"**

A: If the dataset grows beyond 500K chunks. FAISS with IVF (Inverted File) indexing would reduce search from O(N) to O(√N). I'd also consider Milvus for distributed deployment.

---

## 5. Why Cosine Similarity?

### What I Implemented

ChromaDB uses cosine similarity by default. The embeddings are L2-normalized (`normalize_embeddings=True`), which means cosine similarity is mathematically equivalent to dot product (faster to compute).

### Why Cosine Over Euclidean (L2)?

Government scheme documents vary in length. Cosine similarity measures **direction** (semantic meaning) rather than **magnitude** (document length). A 3-page scheme PDF and a 1-page scheme PDF describing the same topic should match.

| Metric | Behavior | Best for |
|---|---|---|
| **Cosine** | Measures angle between vectors | **Document search** (length-invariant) |
| **Euclidean (L2)** | Measures absolute distance | Image similarity, clustering |
| **Dot product** | = cosine when normalized | Fast nearest-neighbor |

### Interviewer Follow-Up

**Q: "Why normalize embeddings?"**

A: Normalization makes all vectors unit-length, so training artifacts (some models produce larger vectors than others) don't bias the search. It also makes cosine = dot product, which ChromaDB can compute faster.

---

## 6. Why Top-K = 3?

### What I Implemented

```python
TOP_K = 3
SIMILARITY_THRESHOLD = 0.3
```

### Why 3?

| K | Context given to LLM | Risk |
|---|---|---|
| 1 | Minimal context (~1000 chars) | Misses related info; answer incomplete |
| **3** | **Sufficient context (~3000 chars)** | **Balanced — covers related schemes** |
| 5 | Good context (~5000 chars) | Noise from lower-relevance chunks |
| 10 | Too much context | Dilutes LLM attention; wastes tokens |

With chunk_size=1000, Top-3 gives the LLM ~3,000 characters of context. DeepSeek V4 Flash has a 128K context window, but more context ≠ better answers — irrelevant chunks degrade LLM output quality (the "lost in the middle" problem).

### Why Threshold = 0.3?

0.3 is a conservative filter:
- **>0.7**: Strong match (exact scheme referenced)
- **0.4-0.7**: Moderate match (related scheme or topic)
- **0.3-0.4**: Weak match (tangentially related)
- **<0.3**: Noise (different domain entirely)

Setting it at 0.3 catches moderate+ matches while filtering noise. If I set it too high (0.7), I'd miss related schemes. Too low (0.1), I'd introduce noise.

### Interviewer Follow-Up

**Q: "Why not dynamic Top-K based on scores?"**

A: Good idea. I could set Top-K dynamically: fetch up to 5, return only those above threshold. If 0 pass, fallback. If 5 pass, cap at 3 for LLM context. This is a one-line change in `retriever.py`.

---

## 7. Why DeepSeek V4 Flash?

### What I Implemented

```python
ChatOpenAI(
    model="deepseek-v4-flash",
    temperature=0.1,
    max_tokens=1024,
    base_url="https://api.deepseek.com",
)
```

### Why DeepSeek over Alternatives?

| Factor | DeepSeek V4 Flash | GPT-4o | Gemini 2.5 Flash |
|---|---|---|---|
| **Cost/1M tokens** | ~$0.14 (input) | $2.50 | $0.15 |
| **Latency** | Fast (flash variant) | Medium | Fast |
| **Context window** | 128K | 128K | 1M |
| **Instruction following** | Strong | Very Strong | Strong |
| **API compatibility** | OpenAI-compatible | Native | Native |

### Why temperature = 0.1?

For government scheme information, I want **deterministic, factual output**. Temperature 0.1 makes the model nearly greedy — it picks the most probable token, minimizing creativity (and thus hallucination).

### Why max_tokens = 1024?

Government scheme answers are factual summaries, not essays. 1024 tokens (~750 words) is enough for a thorough answer with citations, without wasting API costs on unbounded generation.

### Why ChatOpenAI (not ChatDeepSeek)?

DeepSeek's API is fully OpenAI-compatible. Using `ChatOpenAI` with `base_url="https://api.deepseek.com"` is simpler than maintaining a separate `langchain-deepseek` package, and it inherits all LangChain's `ChatOpenAI` features (streaming, tool calling, structured output).

---

## 8. Why This Prompt Template?

### What I Implemented

```python
_SYSTEM_TEMPLATE = """
STRICT RULES:
1. ONLY use information from CONTEXT — no outside knowledge
2. If insufficient → "I cannot find sufficient information..."
3. NEVER fabricate, guess, or hallucinate
4. Cite sources: 📄 Source: <scheme_name> (File: <filename>, Page <page>)
5. Clear, structured, citizen-friendly language
"""
```

### Design Rationale

| Element | Purpose |
|---|---|
| **"STRICT RULES" header** | Visual emphasis for LLM attention |
| **Rule 1 (context-only)** | Prevents LLM from using pre-training knowledge about schemes (which may be outdated) |
| **Rule 2 (exact fallback text)** | Gives the LLM a verbatim response — no creative "I think..." |
| **Rule 3 (no hallucination)** | Explicit prohibition; tested to reduce hallucination rate |
| **Rule 4 (citation format)** | Machine-parseable format — could be extracted for UI rendering |
| **Rule 5 (citizen-friendly)** | Target audience is common citizens, not bureaucrats |
| **Human turn reinforcement** | "Answer strictly based on the context above" — double-negation of context requirement |

### Interviewer Follow-Up

**Q: "Does the prompt actually reduce hallucination?"**

A: Partially. The prompt reduces it but doesn't eliminate it. The primary hallucination defense is: (1) low temperature (0.1), (2) similarity threshold (0.3) filtering irrelevant context, (3) fallback before LLM call when no chunks pass threshold. The prompt is the third line of defense.

---

[Next → Retrieval Decisions](./retrieval-decisions.md)
