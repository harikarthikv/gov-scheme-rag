# Failure Cases — What Can Go Wrong

> **What the interviewer wants:** You don't just demo the happy path. You anticipate failures.

---

## 1. No Relevant Documents Found (Empty Retrieval)

### Trigger

User asks about a topic not covered in any government scheme PDF.

**Example:** "What is the US Social Security system?"

### What Happens

```
[Retriever] → 0 chunks pass threshold (all scores < 0.3)
[RAG Pipeline] → returns fallback BEFORE calling LLM
```

```json
{
  "answer": "I cannot find sufficient information about this in the available
             scheme details. Please try rephrasing your question or ask about
             a specific scheme name.",
  "sources": [],
  "confidence_score": 0.0
}
```

### Why This Is Important

No LLM call = zero hallucination risk, zero API cost, instant response. This is the strongest anti-hallucination defense in the system.

### Interviewer Follow-Up

**Q: "What if the information IS in the documents but retrieval fails?"**

A: This is a **false negative retrieval failure.** Causes:
1. Query phrasing doesn't match document vocabulary
2. Embedding model didn't capture the semantic relationship
3. Threshold is too aggressive

**Fixes:** Lower threshold to 0.2, use query expansion (add synonyms), implement hybrid BM25+semantic search.

---

## 2. Irrelevant Chunks Retrieved (False Positive)

### Trigger

Retrieval returns chunks that are semantically "close" but not actually about the user's question.

**Example:** User asks "pension schemes in Kerala" → Retrieves "Kerala flood relief scheme" (both mention "Kerala," embeddings confused by geographic proximity)

### What Happens

LLM receives:
```
Chunk 1: "Kerala State Pension Scheme provides..."
Chunk 2: "Kerala Flood Relief provides financial assistance..."  ← irrelevant!
Chunk 3: "National Old Age Pension Scheme..."
```

The LLM might incorporate flood relief information into a pension answer if the prompt isn't strong enough.

### Mitigation

1. **Prompt rule #1** ("ONLY use context") prevents the LLM from inventing connections
2. **Temperature 0.1** keeps output deterministic
3. **Confidence score** would be lower (mix of high and low scores → ~0.5)

### Interviewer Follow-Up

**Q: "How would you detect this in production?"**

A: I'd compare the confidence score against a threshold (e.g., 0.6). Below that, flag for human review or ask the user "Did this answer your question?" to collect feedback.

---

## 3. LLM Hallucination Despite Context

### Trigger

The LLM receives relevant context but generates information not present in the chunks.

**Example:** Context says "PM-KISAN provides ₹6,000 per year." LLM generates "₹6,000 per year, paid in 4 installments of ₹1,500 each." The "4 installments" part might be correct pre-training knowledge but isn't in the retrieved chunk.

### What Happens

The answer contains correct information from context PLUS hallucinated details from pre-training.

### Mitigation

1. **Temperature = 0.1** reduces creative generation
2. **Prompt rule #1** prohibits outside knowledge (but LLMs aren't perfect at following this)
3. **No post-generation verification** — this is a system limitation

### Interviewer Follow-Up

**Q: "How would you fix this?"**

A: Post-generation verification:
1. Use a second LLM call to check: "Is every claim in this answer supported by the context?"
2. Or use an NLI (Natural Language Inference) model to check entailment
3. Or implement **Self-RAG** — the LLM evaluates its own generations and retries if unsupported

---

## 4. Chunk Boundary Problems

### Trigger

A scheme's eligibility criteria span a chunk boundary. The LLM sees only half the criteria.

**Example:**
```
Chunk ending at position 1000: "...annual income less than"
Chunk starting at position 800:   "₹1,00,000 per annum, landholding less than 2 hectares..."
```

### What Happens

LLM sees "annual income less than" without the actual threshold. If the chunk overlap is too small (or 0), the context is truncated.

### Mitigation

**200-character overlap** (20%) ensures most sentences that span chunk boundaries appear in both chunks. The `["\n\n", "\n", ". ", " ", ""]` separator hierarchy tries to split at sentence boundaries first.

### Remaining Risk

Long bullet points or tables in PDFs may still get split. PyMuPDF may not preserve PDF structure perfectly.

---

## 5. Embedding Model Drift

### Trigger

The user's query uses modern terminology not present in the embedding model's training data (2019 cutoff for MiniLM). Or the document uses domain-specific abbreviations the model hasn't seen.

**Example:** Query: "AgriStack scheme" (a recent initiative). The embedding model may not capture its relationship to "digital agriculture platform."

### What Happens

Poor semantic matching → low similarity scores → retrieved chunks may not include the right documents.

### Mitigation

- `SIMILARITY_THRESHOLD = 0.3` is lenient enough to catch partial matches
- Could retrain/fine-tune embeddings on government-domain data (TSDAE or SimCSE)

---

## 6. Large-Scale Latency Degradation

### Trigger

With 1,744 PDFs, retrieval is fast (~0.1s). At 100K+ PDFs with millions of chunks, ChromaDB brute-force search becomes O(N).

### What Happens

Latency grows linearly with number of chunks:
- 10K chunks: ~0.1s
- 100K chunks: ~1s
- 1M chunks: ~10s
- 10M chunks: ~100s (unusable)

### Mitigation

Switch to FAISS with IVF (Inverted File) index → O(√N) search. Or use Annoy/HNSW for approximate nearest neighbor.

---

## 7. PDF Parsing Failures

### Trigger

PyMuPDF cannot extract text from:
- Scanned PDFs (image-only, no text layer)
- Password-protected PDFs
- PDFs with complex tables or multi-column layouts

### What Happens

```python
except Exception as exc:
    logger.warning(f"Skipping '{pdf_path.name}': {exc}")
    failed += 1
```

The PDF is skipped. The system **silently loses** that scheme's information.

### Mitigation

1. Add OCR fallback (Tesseract/pytesseract) for scanned PDFs
2. Log skipped PDFs and alert operators
3. Use pdfplumber as fallback (better at tables; PyMuPDF is better at text extraction speed)

---

## 8. DeepSeek API Failures

### Trigger

- API key invalid/expired
- Rate limiting (429)
- DeepSeek servers down
- Network timeout

### What Happens

The exception propagates to `main()`:
```python
except Exception as exc:
    logger.error(f"Error processing query: {exc}")
    print(f"\n⚠️  An error occurred: {exc}\n")
```

The user sees an error. The CLI loop continues (doesn't crash).

### Missing (Not Yet Implemented)

- Retry logic (exponential backoff)
- Circuit breaker (stop calling after N failures)
- Graceful degradation (if DeepSeek is down, respond "Service temporarily unavailable")

---

## Failure Mode Summary Table

| # | Failure | Detected? | Mitigated? | Impact |
|---|---|---|---|---|
| 1 | Empty retrieval | ✅ (threshold check) | ✅ (fallback, no LLM call) | Low |
| 2 | Irrelevant chunks | ⚠️ (confidence score) | ⚠️ (prompt, low temp) | Medium |
| 3 | LLM hallucination | ❌ (no verification) | ⚠️ (prompt + temp) | High |
| 4 | Chunk boundary split | ⚠️ (200 overlap) | ⚠️ (separators) | Medium |
| 5 | Embedding drift | ❌ (no monitoring) | ⚠️ (lenient threshold) | Medium |
| 6 | Scale latency | ❌ (brute-force) | ❌ (not implemented) | Low (current scale) |
| 7 | PDF parse failure | ✅ (logged, skipped) | ⚠️ (PyMuPDF retry) | Medium |
| 8 | API failure | ✅ (exception caught) | ❌ (no retry logic) | High |

**Legend:** ✅ = handled, ⚠️ = partially handled, ❌ = not handled

---

[Next → Improvements](./improvements.md)
