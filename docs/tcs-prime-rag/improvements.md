# Improvements — Production-Ready Enhancements

> **What the interviewer wants:** You don't think your current implementation is perfect. You have a roadmap.

---

## Retrieval Quality Improvements

### 1. Hybrid Search (BM25 + Semantic)

**Current:** Pure semantic (dense) retrieval via cosine similarity.

**Problem:** Semantic search misses exact keyword matches. Query: "PM-KISAN scheme number 2024" → Semantic might return similar schemes; BM25 would find the exact scheme ID.

**Implementation:**
```python
from langchain.retrievers import EnsembleRetriever
bm25_retriever = BM25Retriever.from_documents(chunks)
semantic_retriever = db.as_retriever(search_kwargs={"k": 5})
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, semantic_retriever],
    weights=[0.3, 0.7]  # 30% BM25, 70% semantic
)
```

**Trade-off:** Adds BM25 index build time (~seconds), slightly higher latency, but significantly better recall.

### 2. Query Expansion

**Current:** User's raw query is embedded directly.

**Problem:** "Help for old people" → might not match "Senior Citizen Welfare Scheme."

**Implementation:**
```python
# Use LLM to expand query before embedding
expanded_query = llm.invoke(
    "Generate 3 alternative phrasings for this search query: " + user_query
)
# Embed all 4 queries, average the embeddings
```

**Trade-off:** 1 extra LLM call per query (~$0.001). Worth it for recall-critical applications.

### 3. Re-Ranking

**Current:** Top-3 chunks go directly to LLM in ChromaDB score order.

**Problem:** ChromaDB's cosine similarity is approximate. A re-ranker (cross-encoder) would be more precise.

**Implementation:**
```python
from sentence_transformers import CrossEncoder
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
# Retrieve top-10 with ChromaDB, re-rank to top-3 with cross-encoder
```

**Trade-off:** Adds ~100ms per query. Cross-encoder is O(N×M) vs bi-encoder O(N+M) (N=chunks, M=query tokens), but for top-10 only, the overhead is manageable.

### 4. Multi-Vector / Parent-Child Retrieval

**Current:** Small chunks (1000 chars) both indexed and returned to LLM.

**Problem:** A scheme's full context may span 3-4 pages. Small chunks might miss surrounding context.

**Implementation:** Index small child chunks (500 chars) for precise retrieval, but return the full parent page (entire PDF page text) to the LLM.

**Trade-off:** Larger context to LLM = higher token cost. But richer answers.

---

## Latency Optimizations

### 1. Query Caching

**Current:** Every query triggers full pipeline (embed + search + LLM).

**Implementation:**
```python
import hashlib, functools

@functools.lru_cache(maxsize=1000)
def cached_query(query: str) -> dict:
    return rag.query(query)
```

**Impact:** Repeated queries ("What is PM Kisan?") return instantly. Cache hit rate depends on user behavior.

### 2. Async LLM Calls

**Current:** Synchronous `llm.invoke()` blocks the CLI loop.

**Implementation:**
```python
# Use async invoke
response = await llm.ainvoke(prompt_messages)
```

**Impact:** CLI feels faster (but actual latency unchanged). Essential for web server deployment.

### 3. Embedding Model Quantization

**Current:** 32-bit float embeddings (384 × 4 bytes = 1,536 bytes per vector).

**Implementation:** Use `sentence-transformers` with `int8` quantization or ONNX runtime.

**Impact:** 4× smaller vectors → faster search, less memory. Negligible accuracy loss (<1%).

### 4. Approximate Nearest Neighbor (ANN) Index

**Current:** ChromaDB brute-force (O(N) per query).

**Implementation:** Switch ChromaDB backend to HNSW (configurable via `hnsw:space` and `hnsw:construction_ef`).

**Impact:** O(log N) search. Essential above 100K chunks.

---

## Robustness Improvements

### 1. Retry Logic with Exponential Backoff

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def query_with_retry(question):
    return rag.query(question)
```

### 2. PDF OCR Fallback

```python
try:
    text = page.get_text()   # PyMuPDF (fast)
    if not text.strip():     # Empty → likely scanned PDF
        import pytesseract
        pix = page.get_pixmap()
        text = pytesseract.image_to_string(pix)
except Exception:
    pass  # Skip unparseable PDFs
```

### 3. Streaming Responses

```python
# Instead of llm.invoke() → wait for full response
for chunk in llm.stream(prompt_messages):
    yield chunk.content   # Show tokens as they arrive
```

**Impact:** Perceived latency drops from 5s to <1s for first token.

---

## Scale Improvements

### Current Architecture Limits

| Component | Current | Limit | Breaks At |
|---|---|---|---|
| PDFs | 1,744 | ~10K | PyMuPDF I/O bottleneck |
| Chunks | ~50K (est.) | ~500K | ChromaDB brute-force |
| Queries/sec | 1 (CLI) | ~10 | Synchronous LLM |

### Path to Millions of Documents

1. **Distributed Ingestion:** Spark/PySpark for parallel PDF processing
2. **Milvus / Weaviate:** Distributed vector DB with sharding
3. **Kubernetes:** Auto-scale retrieval + LLM services
4. **Kafka:** Async ingestion pipeline with dead-letter queue for failed PDFs
5. **Redis caching:** Frequently accessed embeddings in memory

---

## Monitoring & Observability (Not Implemented)

| What | Why |
|---|---|
| Retrieval latency histogram | Detect performance regression |
| Similarity score distribution | Tune threshold dynamically |
| LLM token usage per query | Cost monitoring |
| Fallback rate (% queries with 0 results) | Detect knowledge gaps |
| User feedback loop (thumbs up/down) | Ground-truth for eval |
| PDF parse failure rate | Detect dataset quality issues |

---

## Evaluation Framework (Not Implemented)

### What I Would Build

```python
eval_dataset = [
    {
        "query": "What is the eligibility for PM-KISAN?",
        "expected_sources": ["pm_kisan.pdf"],
        "expected_answer_contains": ["landholding", "farmer", "income support"],
    },
    # ... 100+ curated query-answer pairs
]
```

### Metrics

- **Recall@K:** Fraction of queries where relevant document is in top-K
- **MRR (Mean Reciprocal Rank):** Average of 1/rank of first relevant document
- **Faithfulness:** % of answer claims supported by context (LLM-as-judge)
- **Answer Relevance:** Cosine similarity between answer and query embeddings

---

[Next → Debugging](./debugging.md)
