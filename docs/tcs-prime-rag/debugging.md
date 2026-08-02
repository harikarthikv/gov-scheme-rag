# Debugging — Troubleshooting the RAG Pipeline

> **What the interviewer wants:** You can debug production issues systematically.

---

## Debugging Toolkit

### 1. Is the Data Loading Correctly?

```bash
# Quick check: how many PDFs and pages are we ingesting?
uv run python -c "
from src.data_loader import load_all_documents
docs = load_all_documents()
print(f'Documents (pages): {len(docs)}')
print(f'Sample metadata: {docs[0].metadata}')
print(f'Sample content[:200]: {docs[0].page_content[:200]}')
"
```

**What to check:**
- `len(docs)` should be >0 and roughly proportional to PDF count
- Metadata should have `filename`, `page`, `scheme_name`
- `page_content` should not be empty
- Any "Skipping" warnings in logs indicate PDF parse failures

### 2. Is Chunking Working?

```bash
uv run python -c "
from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline
docs = load_all_documents()
ep = EmbeddingPipeline()
chunks = ep.split_documents(docs)
print(f'Chunks: {len(chunks)}')
print(f'Sample chunk: {chunks[0].page_content[:200]}')
print(f'Metadata inherited: {chunks[0].metadata}')
"
```

**What to check:**
- `len(chunks) > len(docs)` (each page produces 1+ chunks)
- Chunk metadata should inherit from parent Document
- No chunks should be empty

### 3. Are Embeddings the Right Dimension?

```bash
uv run python -c "
from src.embedding import EmbeddingPipeline
ep = EmbeddingPipeline()
print(f'Dimension: {ep.dimension}')  # Should be 384
# Test single embedding
import numpy as np
v = ep.embed_query('test query')
print(f'Shape: {v.shape}')  # Should be (384,)
print(f'Normalized: {np.linalg.norm(v):.6f}')  # Should be ~1.0
"
```

**What to check:**
- `dimension == 384` (all-MiniLM-L6-v2)
- `norm(v) ≈ 1.0` (L2 normalized)
- If dimension is wrong → wrong model loaded; check `EMBEDDING_MODEL` in config

### 4. Is ChromaDB Persisting Correctly?

```bash
# Check if ChromaDB exists on disk
ls chroma_db/
# Expected: chroma.sqlite3 + index files

uv run python -c "
from src.config import VECTOR_DB_PATH
from src.embedding import EmbeddingPipeline
from src.vector_store import SchemeVectorStore
ep = EmbeddingPipeline()
store = SchemeVectorStore(embedding_pipeline=ep)
print(f'DB exists: {store.db_exists()}')
if store.db_exists():
    db = store.load()
    print(f'Collection count: {db._collection.count()}')
"
```

**What to check:**
- `db_exists()` returns True after first build
- `_collection.count()` matches expected chunk count
- If count is 0 → chunks were not added; check `build_from_documents()` logs

### 5. Is Retrieval Returning Good Results?

```bash
uv run python -c "
from src.embedding import EmbeddingPipeline
from src.vector_store import SchemeVectorStore
from src.retriever import Retriever
ep = EmbeddingPipeline()
store = SchemeVectorStore(embedding_pipeline=ep)
db = store.load()
retriever = Retriever(db=db)
results = retriever.retrieve('What is PM Kisan scheme?')
for i, r in enumerate(results):
    print(f'[{i+1}] Score: {r[\"score\"]:.4f}')
    print(f'    Source: {r[\"metadata\"].get(\"scheme_name\", \"?\")}')
    print(f'    Chunk[:100]: {r[\"chunk\"][:100]}')
    print()
"
```

**What to check:**
- Results have scores > `SIMILARITY_THRESHOLD`
- Top result should be about PM Kisan (or related agriculture scheme)
- If 0 results: threshold too high, or query-concept mismatch
- If irrelevant results: embedding model not capturing semantics

### 6. Is the Prompt Correctly Formatted?

```bash
uv run python -c "
from src.prompt import get_prompt_template
prompt = get_prompt_template()
msgs = prompt.format_messages(
    context='TEST CONTEXT: PM-KISAN provides income support.',
    sources='[1] pm_kisan.pdf (score: 0.95)',
    query='What is PM-KISAN?'
)
print(msgs[0].content[:500])   # System message
print('---')
print(msgs[1].content)         # Human message (query)
"
```

**What to check:**
- Context is injected into the system message
- Sources are formatted correctly
- Query appears in the human message
- No `{placeholder}` left un-filled (would cause KeyError at runtime)

### 7. Is the LLM Responding (and Not Hallucinating)?

```bash
uv run python -c "
from src.embedding import EmbeddingPipeline
from src.vector_store import SchemeVectorStore
from src.retriever import Retriever
from src.rag import RAGPipeline
ep = EmbeddingPipeline()
store = SchemeVectorStore(embedding_pipeline=ep)
db = store.load()
retriever = Retriever(db=db)
rag = RAGPipeline(retriever=retriever)
# Test with a known scheme
result = rag.query('What is the eligibility for PM Kisan Samman Nidhi?')
print(f'Answer: {result[\"answer\"]}')
print(f'Sources: {len(result[\"sources\"])}')
print(f'Confidence: {result[\"confidence_score\"]}')
"
```

**What to check:**
- Answer mentions PM-KISAN details (not generic "income support")
- Sources are non-empty
- Confidence > 0.3
- Answer doesn't contain "I cannot find" (should find PM-KISAN)

---

## Common Bugs & Fixes

### Bug 1: `ModuleNotFoundError: No module named 'chromadb'`

**Cause:** venv not activated or requirements not installed.

**Fix:**
```bash
uv pip install -r requirements.txt
```

### Bug 2: `openai.OpenAIError: Missing credentials`

**Cause:** `.env` file missing or `DEEPSEEK_API_KEY` not set.

**Debug:**
```bash
uv run python -c "from src.config import DEEPSEEK_API_KEY; print(len(DEEPSEEK_API_KEY))"
```

If `0` → create `.env` file with `DEEPSEEK_API_KEY=sk-...`

### Bug 3: ChromaDB loads 0 vectors

**Cause:** `load()` was called before `build_from_documents()`, or the persist directory is empty.

**Debug:** Check `chroma_db/chroma.sqlite3` file size (should be > 2MB with data).

**Fix:** Delete `chroma_db/` and re-run `app.py` to rebuild.

### Bug 4: All retrieval scores are ~0.5 (no discrimination)

**Cause:** Embedding model not producing meaningful vectors (possible wrong model, or all documents are identical).

**Debug:**
```python
v1 = ep.embed_query("farmer scheme")
v2 = ep.embed_query("health insurance")
from numpy import dot
from numpy.linalg import norm
similarity = dot(v1, v2) / (norm(v1) * norm(v2))
print(similarity)  # Should be < 0.5 for different topics
```

If all similarities are > 0.9 → model issue (check model download).

### Bug 5: `KeyError: 'scheme_name'` during source display

**Cause:** Older ChromaDB metadata doesn't have `scheme_name` field.

**Fix:** The code already falls back:
```python
r['metadata'].get('scheme_name', r['metadata'].get('filename', 'Unknown'))
```

---

## Debugging Philosophy for TCS Prime

**Interviewers want to hear:**
1. "First, I check if data ingestion is correct..."
2. "Then I verify the embedding dimension..."
3. "I isolate retrieval from generation..."
4. "I use the built-in logging (INFO level timestamps)..."

**The systematic approach:**
```
Data Layer → Embedding Layer → Storage Layer → Retrieval Layer → Generation Layer
     ↑              ↑               ↑                ↑                ↑
  Isolate        Isolate         Isolate          Isolate          Isolate
  & test         & test          & test           & test           & test
```

---

[Next → Interview Questions](./interview-questions.md)
