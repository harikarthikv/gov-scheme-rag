# Retrieval Decisions — Strategy Deep-Dive

> **What the interviewer wants:** You understand retrieval deeply, not just "I used ChromaDB."

---

## Retrieval Architecture

```
User Query: "pension scheme for senior citizens"
                    │
                    ▼
        ┌───────────────────────┐
        │  Embed query (384-dim) │  ← all-MiniLM-L6-v2, L2-normalized
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  Cosine similarity     │  ← ChromaDB similarity_search_with_relevance_scores
        │  against all chunks    │     (dot product since vectors are normalized)
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  Sort by score desc    │
        │  Take Top-K (k=3)      │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  Threshold filter      │  ← score ≥ 0.3
        │  Drop score < 0.3      │
        └───────────┬───────────┘
                    │
            ┌───────┴───────┐
            │               │
        results > 0     results == 0
            │               │
            ▼               ▼
    Return chunks    Return fallback
    to LLM           (no LLM call)
```

---

## Similarity Score Explained

### How ChromaDB Computes It

Our embeddings are L2-normalized (`normalize_embeddings=True`). A normalized vector has magnitude 1.0. For two normalized vectors **a** and **b**:

$$cosine\_similarity(a, b) = a \cdot b = \sum_{i=1}^{384} a_i \times b_i$$

ChromaDB's `similarity_search_with_relevance_scores` maps this to [0, 1]:

$$relevance\_score = \frac{1 + cosine\_similarity}{2}$$

### Score Interpretation

| Score Range | Meaning | Example |
|---|---|---|
| 0.8 - 1.0 | Near-identical semantic match | Query: "PM Kisan" → Chunk about PM Kisan |
| 0.6 - 0.8 | Strong related match | Query: "farmer income support" → PM Kisan chunk |
| 0.4 - 0.6 | Moderate, related topic | Query: "pension for elderly" → Old age pension scheme |
| 0.3 - 0.4 | Weak, tangentially relevant | Query: "health insurance" → General insurance scheme |
| < 0.3 | Noise — discarded | Query: "education loan" → Agriculture scheme |

### Verified Fact (from our code)

Our `SIMILARITY_THRESHOLD = 0.3`. This was verified during testing — any chunk above 0.3 typically contained related scheme information; below 0.3 was noise.

---

## Why Not Other Retrieval Strategies?

### 1. Keyword Search (BM25 / TF-IDF)

| BM25 | Semantic (Our Approach) |
|---|---|
| Matches exact words | Matches meaning |
| "senior citizen" ≠ "elderly person" | "senior citizen" ≈ "elderly person" |
| Fast | Slower (embedding + search) |
| No training needed | Model download needed |

**Why not:** Government scheme documents use varied terminology. "Old age pension" in one PDF and "senior citizen welfare" in another are the same concept but different words. Semantic search catches this; BM25 doesn't.

**Hybrid approach (BM25 + semantic) would be better** — see improvements section.

### 2. Dense Passage Retrieval (DPR)

DPR trains separate encoders for queries and documents. It's more accurate but requires labeled query-document pairs. We don't have labeled training data for government schemes.

### 3. ColBERT (Late Interaction)

ColBERT computes token-level interactions between query and document. More accurate but significantly slower and requires more storage. Overkill for our scale.

### 4. Multi-Vector / Parent-Child Retrieval

Store small chunks for retrieval but return larger parent chunks for context. Could improve coherence (see improvements section).

---

## The Confidence Score

### What I Implemented

```python
scores = [r["score"] for r in retrieved]
confidence_score = round(sum(scores) / len(scores), 4)
```

### What It Means

The confidence score is the **average similarity of all retrieved chunks**. It's a proxy for "how relevant was the retrieved context?"

| Confidence | Interpretation |
|---|---|
| > 0.7 | High confidence — chunks are strongly related to query |
| 0.4 - 0.7 | Moderate confidence — related but not exact match |
| < 0.4 | Low confidence — weak match, answer may be general |
| 0.0 | No chunks passed threshold — fallback response |

### Limitations (Be Honest!)

This is **not** a true confidence metric. It tells you "how similar were the chunks to the query" — not "how correct is the answer." The LLM could still generate an incorrect answer from relevant chunks if it misinterprets the context.

A better approach would be LLM-as-judge: ask the LLM to self-evaluate its answer against the context. But this doubles API cost.

---

## Retrieval Failure Modes

### Mode 1: Empty Retrieval

**Cause:** No chunks pass the similarity threshold (all scores < 0.3)

**Behavior:** Returns fallback without calling LLM

```python
return {
    "answer": "I cannot find sufficient information...",
    "sources": [],
    "confidence_score": 0.0,
}
```

### Mode 2: Low-Quality Retrieval

**Cause:** Top-3 chunks are tangentially related (scores 0.3-0.4)

**Behavior:** LLM receives weak context, may produce vague or generic answer

**Mitigation:** Prompt rule #2 forces the LLM to admit it doesn't have enough information

### Mode 3: Relevant but Incomplete

**Cause:** The answer spans multiple chunks but only 3 are retrieved

**Behavior:** LLM receives partial context, answer is incomplete

**Mitigation:** Increase TOP_K to 5; the chunk overlap (200 chars) helps bridge between adjacent chunks

### Mode 4: Query-Document Vocabulary Mismatch

**Cause:** User uses terminology that doesn't appear in any PDF

**Example:** "What is the PM's farmer scheme?" → Documents say "PM-KISAN" not "PM's farmer scheme"

**Mitigation:** Semantic embeddings handle synonyms; all-MiniLM-L6-v2 was trained on diverse text

---

[Next → Failure Cases](./failure-cases.md)
