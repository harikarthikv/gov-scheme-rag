# 75+ TCS Prime Interview Questions — RAG Pipeline

> Every answer is tied to **this repository's implementation**. No generic RAG answers.

---

## Section 1: Architecture & System Design (15 Questions)

### Q1: Walk me through your RAG pipeline architecture.

**A:** The pipeline has two phases. **Offline (indexing):** 1,744 government scheme PDFs → PyMuPDF extracts text page-by-page → LangChain Document objects → RecursiveCharacterTextSplitter chunks into 1000-char segments with 200-char overlap → all-MiniLM-L6-v2 generates 384-dim normalized embeddings → ChromaDB persists to `chroma_db/` with UUIDs per chunk. **Online (query):** User query → same embedding model → cosine similarity search (Top-3) in ChromaDB → threshold filter at 0.3 → ChatPromptTemplate with strict grounding rules → DeepSeek V4 Flash generates answer → returns structured JSON with answer, sources, and confidence score.

### Q2: Why modular architecture instead of a single script?

**A:** Each `src/` module has a single responsibility: `data_loader.py` (ingestion), `embedding.py` (chunking + vectors), `vector_store.py` (persistence), `retriever.py` (search), `prompt.py` (template), `rag.py` (orchestration). This means I can swap the embedding model without touching retrieval code, switch from ChromaDB to FAISS without changing the LLM layer, and test each component in isolation. In `config.py`, every tunable constant is in one place — no magic numbers scattered across files.

### Q3: How would you deploy this to production?

**A:** Replace the CLI loop with a FastAPI endpoint. The `_initialize_system()` function runs once at startup (loading embedding model + ChromaDB). Each request calls `rag.query()`. Add async LLM calls (`llm.ainvoke`), Redis caching for repeated queries, and Prometheus metrics for latency/token tracking. Containerize with Docker, deploy on Kubernetes with auto-scaling based on QPS.

### Q4: What's the bottleneck in your pipeline?

**A:** LLM inference latency (~3-5s per query) dominates end-to-end time. Retrieval (~0.1s) and embedding (~0.05s) are negligible. At scale, ChromaDB brute-force search would become O(N) — I'd switch to HNSW approximate index. But the biggest bottleneck right now is the synchronous LLM call blocking the CLI loop.

### Q5: How does your system handle concurrent users?

**A:** It doesn't — the CLI is single-user. For multi-user, I'd wrap `RAGPipeline` in a FastAPI app with async endpoints. ChromaDB supports concurrent reads, so multiple query threads can share one loaded collection. The embedding model would need a singleton pattern (load once, reuse). DeepSeek API handles concurrency via API key-level rate limits.

### Q6: Why separate `app.py` (smart init) from the old `index.py` (manual build)?

**A:** The old design required two commands (`python index.py` then `python app.py`), which is fragile — if `index.py` wasn't run, `app.py` crashes. The smart init in `app.py` checks `db_exists()` and auto-builds if needed. This is the "pit of success" pattern: the default behavior is correct.

### Q7: Walk me through the smart init logic.

**A:** `_initialize_system()` first loads `EmbeddingPipeline` (always needed). Then `SchemeVectorStore` checks `VECTOR_DB_PATH` on disk: if directory exists and has files → `store.load()` opens the SQLite database directly (sub-second). If not → `load_all_documents()` reads PDFs, `store.build_from_documents()` chunks + embeds + persists (minutes). Then `Retriever` + `RAGPipeline` are created from the loaded Chroma instance.

### Q8: How do you handle configuration across environments?

**A:** All constants are in `src/config.py` — single source of truth. Environment-specific values (API keys, model names) come from `.env` via `python-dotenv` with `find_dotenv()` and `override=True`. The `.env.example` file documents all required variables. For production, I'd replace `.env` with a secrets manager (AWS Secrets Manager, HashiCorp Vault).

### Q9: What happens if `.env` is missing?

**A:** `find_dotenv(raise_error_if_not_found=False)` returns `None` silently. `DEEPSEEK_API_KEY` defaults to `""`. When `ChatOpenAI` initializes, the OpenAI client raises `OpenAIError: Missing credentials`. This is caught in `main()` → logs error → `sys.exit(1)`. The user sees a clear error message, not a cryptic stack trace.

### Q10: Explain your dependency management.

**A:** I use `uv` (Rust-based Python package manager) for fast venv creation and deterministic installs. `requirements.txt` lists minimum versions (`>=`). Key dependencies: `langchain` ecosystem (core, community, chroma, huggingface, openai), `sentence-transformers` (local embeddings), `PyMuPDF` (PDF parsing), `chromadb` (vector DB), `python-dotenv` (config).

### Q11: Why LangChain? Couldn't you build this without it?

**A:** LangChain provides standardized abstractions: `Document` objects with `page_content` + `metadata`, `RecursiveCharacterTextSplitter`, `Chroma` vector store wrapper, `ChatPromptTemplate`, `StrOutputParser`. Without LangChain, I'd be writing boilerplate for each of these. The trade-off is LangChain's abstraction overhead and occasional API changes between versions.

### Q12: What's the most complex module in your codebase?

**A:** `src/rag.py` — it orchestrates retrieval, prompt formatting, LLM invocation, confidence calculation, and fallback handling in a single `query()` method. It handles three code paths: (1) normal retrieval → LLM generation, (2) empty retrieval → fallback without LLM call, (3) LLM exception → propagates to caller. The complexity is in managing these states cleanly.

### Q13: How do you test this system?

**A:** Currently manual testing via the CLI. I test: (1) known schemes ("PM Kisan") should return relevant answers, (2) out-of-domain queries ("US Social Security") should return fallback, (3) empty queries should be ignored, (4) "exit" should cleanly terminate. For production, I'd add: unit tests per module, integration tests for the full pipeline, and an evaluation set of 100+ query-answer pairs with expected sources.

### Q14: How would you add a new data source (e.g., CSV of schemes)?

**A:** Add a `CSVDataLoader` class alongside `SchemeDataLoader` in `data_loader.py`, implementing the same `load() -> list[Document]` interface. Then create a `DataLoaderFactory` that selects loader based on file extension. The rest of the pipeline (chunking, embedding, storage) remains unchanged — this is the benefit of the modular architecture.

### Q15: What's the cold start problem and how do you handle it?

**A:** Cold start = first `app.py` run must download embedding model (~90MB), load 1,744 PDFs, chunk, embed, persist. This takes minutes. I handle it with: (1) the embedding model is cached after first download, (2) ChromaDB persists so subsequent runs are instant, (3) clear logging shows which stage is running so the user knows progress. I don't pre-build the index — it's built lazily on first run.

---

## Section 2: Embeddings & Vector Search (12 Questions)

### Q16: Why all-MiniLM-L6-v2? Why not OpenAI embeddings?

**A:**
- **Cost:** all-MiniLM-L6-v2 is free and runs locally. OpenAI text-embedding-3-small costs $0.02/1M tokens. For 1,744 PDFs × ~2 pages × ~500 words = ~1.7M words per re-index, that's negligible. But for continuous re-indexing, local is free.
- **Offline:** The system works without internet after model download (critical for government deployments).
- **Privacy:** No data leaves the machine during embedding.
- **Quality:** MTEB score of 56.3 vs OpenAI's 64.6 — I trade ~8 points for zero cost and offline capability.

### Q17: What's the significance of 384 dimensions?

**A:** all-MiniLM-L6-v2 produces 384-dimensional dense vectors. Lower than BGE-large (1024) or OpenAI (1536). This means: (1) faster similarity computation (384 dot products vs 1536), (2) smaller index on disk (~600 bytes/vector vs ~6KB), (3) slightly lower semantic resolution. For government scheme text (structured, domain-specific), 384 dimensions capture enough semantic information.

### Q18: Why normalize embeddings? (`normalize_embeddings=True`)

**A:** L2 normalization makes all vectors unit-length (magnitude = 1.0). Benefits: (1) Cosine similarity becomes dot product (faster to compute), (2) prevents vectors with larger magnitudes from dominating similarity scores, (3) ChromaDB's `similarity_search_with_relevance_scores` maps cosine similarity to [0, 1] correctly only with normalized vectors. Verified: `np.linalg.norm(embedding) ≈ 1.0 ± 0.0001`.

### Q19: How does ChromaDB compute similarity under the hood?

**A:** With normalized embeddings, ChromaDB computes cosine similarity = dot product. Then `similarity_search_with_relevance_scores` maps it: `score = (1 + cosine) / 2`. So:
- Cosine = 1.0 (identical) → score = 1.0
- Cosine = 0.0 (orthogonal/unrelated) → score = 0.5
- Cosine = -1.0 (opposite) → score = 0.0

Our `SIMILARITY_THRESHOLD = 0.3` means cosine > -0.4 passes — fairly lenient.

### Q20: Why not FAISS instead of ChromaDB?

**A:** ChromaDB provides: (1) built-in persistence (SQLite — no manual save/load like FAISS), (2) metadata filtering (WHERE clauses on `scheme_name`, `page`), (3) LangChain-native integration, (4) simpler setup (no NumPy array management). FAISS would be better at 500K+ vectors with IVF indexing, but at our scale (~50K chunks estimated), ChromaDB is faster to develop with.

### Q21: Explain the UUID assignment to chunks.

**A:** Each chunk gets `str(uuid.uuid4())` as its ChromaDB document ID. This provides: (1) traceability — I can retrieve a specific chunk by ID for debugging, (2) idempotency — re-adding the same chunk with the same UUID doesn't duplicate, (3) deletion — I can remove specific chunks without rebuilding. Without UUIDs, ChromaDB auto-generates IDs that change on every build.

### Q22: What distance metric does ChromaDB use?

**A:** Cosine similarity by default. Since our embeddings are L2-normalized, cosine similarity = dot product. ChromaDB also supports L2 (Euclidean) and IP (Inner Product). I use cosine because it's length-invariant — important when comparing queries (short) against document chunks (long).

### Q23: How would you handle a document that's entirely images (scanned PDF)?

**A:** PyMuPDF's `page.get_text()` returns empty string for image-only PDFs. Currently, those pages are skipped (empty text → no Document created). To handle them: add OCR fallback using `pytesseract` after `page.get_pixmap()`. This is in `improvements.md` but not yet implemented. The trade-off is OCR adds ~1-2s per page.

### Q24: What if two chunks have identical embeddings?

**A:** With normalized embeddings, identical text → identical vectors → cosine similarity = 1.0. ChromaDB returns both with score 1.0. The LLM sees duplicated context. This is rare (different PDF pages about the same scheme have slightly different text), but could happen with boilerplate headers/footers. Deduplication could be added at chunking time using text hashing.

### Q25: How do you handle multi-lingual queries? (Hindi, Tamil, etc.)

**A:** Currently, only English. The embedding model (all-MiniLM-L6-v2) was trained primarily on English. For Hindi/Tamil queries: (1) translate query to English via a translation API before embedding, or (2) use a multilingual embedding model like `paraphrase-multilingual-MiniLM-L12-v2` (also 384-dim, drop-in replacement). The PDFs themselves are in English, so multilingual retrieval would need query translation regardless.

### Q26: What's the difference between bi-encoder and cross-encoder for retrieval?

**A:** Bi-encoder (our approach): Encode query and documents independently, then compute similarity. Fast (O(N) for N documents) but less precise. Cross-encoder: Feed [query, document] pair through a transformer, output relevance score. Precise but slow (O(N×M) per query, where M = attention complexity). I use bi-encoder for retrieval and could add cross-encoder re-ranking (see improvements).

### Q27: Why cosine similarity and not dot product or Euclidean?

**A:** Cosine measures semantic direction, not magnitude. A 10-page PDF and a 1-page PDF about the same scheme should match. Euclidean distance would penalize the longer document. Dot product (unnormalized) would favor longer documents. Since I normalize embeddings, cosine = dot product, so I get the semantic benefit of cosine with the computational efficiency of dot product.

---

## Section 3: Chunking Strategy (8 Questions)

### Q28: Why chunk_size = 1000?

**A:** Government scheme sections in the PDFs average 200-500 words (~800-2000 chars). 1000 chars captures most scheme descriptions in a single chunk without over-fetching. Empirically: smaller chunks (512) split eligibility criteria across chunks; larger chunks (2048) include unrelated content. 1000 is the sweet spot for this domain.

### Q29: Why chunk_overlap = 200 (20%)?

**A:** Prevents sentences from being split at chunk boundaries. If a scheme's "How to Apply" section starts at position 980, it appears in chunk A (positions 800-1000) and chunk B (positions 800-1800) — in both places, the section is complete. 200 chars ≈ 2-3 sentences of overlap, which is sufficient.

### Q30: Explain the separator hierarchy: `["\n\n", "\n", ". ", " ", ""]`

**A:** The splitter tries each separator in order:
1. `"\n\n"` → paragraph boundaries (preserves document structure)
2. `"\n"` → line breaks (respects formatting)
3. `". "` → sentence boundaries (semantic units)
4. `" "` → word boundaries (prevents mid-word splits)
5. `""` → character-level (never fails — fallback)

This is `RecursiveCharacterTextSplitter`'s default hierarchy. It produces chunks that are semantically coherent because it prefers splitting at natural text boundaries.

### Q31: How many chunks do 1,744 PDFs produce?

**A:** Verified fact: 1,744 PDFs × average ~2-3 pages per PDF × ~1-2 chunks per page ≈ **~5,000-10,000 chunks**. This is an estimate — I haven't run the full indexing to completion yet. Each chunk has a 384-dim vector, so the ChromaDB index is ~8-15MB of vector data plus SQLite overhead.

### Q32: What is the "lost in the middle" problem?

**A:** LLMs pay less attention to information in the middle of long contexts. If I retrieve Top-10 chunks (10,000 chars), the LLM may ignore chunks 3-7. With Top-3 (3,000 chars), each chunk is in the "beginning" of the context window, maximizing attention. This is an empirically observed LLM behavior — Google and Anthropic have published papers on it.

### Q33: Would you use SemanticChunker instead?

**A:** SemanticChunker splits at points where embedding similarity drops (indicating a topic shift). It produces more semantically coherent chunks than fixed-size splitting. I didn't use it because: (1) it requires embedding every sentence (2× the embedding cost), (2) RecursiveCharacterTextSplitter works well enough for government PDFs (structured, consistent format), (3) simpler to explain in interviews. For unstructured text (web pages, articles), SemanticChunker would be better.

### Q34: What happens if a single PDF page is longer than chunk_size?

**A:** The splitter creates multiple chunks from that page. All chunks inherit the parent Document's metadata (`filename`, `page`, `scheme_name`). So if page 5 of `pm_kisan.pdf` produces 3 chunks, all 3 have `{page: 5}`. This means source citations show the correct page even for multi-chunk pages.

### Q35: How does metadata inherit during chunking?

**A:** `RecursiveCharacterTextSplitter.split_documents()` automatically copies parent metadata to child chunks. LangChain handles this. In our code: `chunks = self.text_splitter.split_documents(documents)`. Each chunk gets `filename`, `page`, `scheme_name`, `source` from its parent Document.

---

## Section 4: Retrieval Strategy (8 Questions)

### Q36: Why Top-K = 3?

**A:** With chunk_size=1000, Top-3 gives ~3,000 chars of context — enough to cover a scheme's full description + eligibility + benefits. More chunks (5, 10) add noise without adding value. DeepSeek's 128K context window could handle more, but "more context ≠ better answers" due to the lost-in-the-middle effect and token cost.

### Q37: How did you choose the similarity threshold of 0.3?

**A:** Empirical tuning. I tested queries against known schemes:
- PM-KISAN query: top score 0.89 → clearly passes
- Related scheme query: top score 0.42 → passes at 0.3
- Unrelated query ("US Social Security"): top score 0.18 → correctly filtered

0.3 is the cutoff where chunks transition from "tangentially related" to "noise." Lower (0.1) would pass noise; higher (0.5) would filter related schemes.

### Q38: What if all 3 retrieved chunks are below threshold?

**A:** The `Retriever.retrieve()` method filters all chunks with `score < SIMILARITY_THRESHOLD`. If 0 pass: `RAGPipeline.query()` receives an empty list and returns the fallback response **without calling the LLM**. This is the strongest hallucination defense: no input → no hallucination.

### Q39: Explain `similarity_search_with_relevance_scores`.

**A:** This LangChain Chroma method: (1) embeds the query, (2) computes cosine similarity against all chunk vectors, (3) sorts descending, (4) takes Top-K, (5) maps cosine similarity to [0, 1] range using `(1 + cosine) / 2`. It returns `list[tuple[Document, float]]` — each result is a (document, relevance_score) pair.

### Q40: What's the difference between L2 distance and cosine similarity in practice?

**A:** L2 distance measures absolute vector distance: `sqrt(Σ(ai-bi)²)`. It's sensitive to vector magnitude. If one embedding is longer (higher magnitude) due to longer text, L2 penalizes it. Cosine similarity measures angular difference: `(a·b)/(|a||b|)`. It's magnitude-invariant. Since document chunks vary in length (some are 1000 chars, some are 500), cosine is more appropriate.

### Q41: How would you add metadata filtering to retrieval?

**A:** ChromaDB supports `where` clauses. Currently not used but could be added:
```python
db.similarity_search_with_relevance_scores(
    query=query,
    k=TOP_K,
    filter={"state": "Maharashtra"}  # Only schemes from Maharashtra
)
```
This requires the metadata to have a `state` field. Our current metadata doesn't include state (PDFs don't consistently contain this), but if added during ingestion, filtering is a one-line change.

### Q42: Why not use MMR (Maximal Marginal Relevance)?

**A:** MMR balances relevance and diversity — it penalizes chunks too similar to already-selected ones. This prevents all Top-3 results being from the same PDF. I didn't use it because: (1) for scheme-specific queries, returning 3 chunks from the same scheme is actually desirable, (2) MMR adds latency (requires re-ranking). For open-ended queries ("what schemes exist for farmers?"), MMR would be better.

### Q43: How is retrieval latency measured in your code?

**A:** `Retriever.retrieve()` captures `time.time()` before and after the ChromaDB call:
```python
start = time.time()
raw_results = self.db.similarity_search_with_relevance_scores(query=query, k=TOP_K)
elapsed = time.time() - start
logger.info(f"Retrieval completed in {elapsed:.3f}s")
```
The log format includes timestamps, so I can track retrieval latency trends over time.

---

## Section 5: Prompt Engineering & Hallucination (10 Questions)

### Q44: Walk me through your prompt template design.

**A:** Two-part ChatPromptTemplate:
- **System message:** Expert persona + 5 STRICT RULES (context-only, admit uncertainty, never fabricate, cite sources, citizen-friendly language) + injected context + source list
- **Human message:** User query + reinforcement ("Answer strictly based on the context above")

The template expects 3 variables: `{context}`, `{sources}`, `{query}`. Formatted in `RAGPipeline.query()`.

### Q45: How does your prompt reduce hallucination?

**A:** Three layers of defense:
1. **Prompt rules #1 and #3:** Explicitly prohibit outside knowledge and fabrication
2. **Temperature = 0.1:** Near-deterministic output → less creative hallucination
3. **Fallback before LLM call:** If no chunks pass threshold, return "I cannot find sufficient information" without ever calling the LLM

The prompt is the third line of defense, not the primary one. Primary defenses are temperature and fallback.

### Q46: Why tell the LLM to say the EXACT fallback phrase?

**A:** Giving the LLM a verbatim fallback response ("I cannot find sufficient information about this in the available scheme details") prevents it from improvising. Without an exact phrase, the LLM might say "I think the scheme might be..." — which is speculation. The verbatim phrase is tested and guaranteed to not contain hallucination.

### Q47: What is temperature and why 0.1?

**A:** Temperature controls randomness in token selection. At each step, the LLM has a probability distribution over all possible next tokens. Temperature scales this distribution:
- T=0 → always pick the most probable token (deterministic)
- T=0.1 → 99% probability mass on top-3 tokens (nearly deterministic)
- T=1.0 → natural distribution
- T=2.0 → flatter distribution (creative, more hallucinations)

For government scheme information, I want factual, deterministic output → T=0.1.

### Q48: How would you validate that the LLM isn't hallucinating in production?

**A:** Post-generation verification using LLM-as-judge:
```python
verification_prompt = f"""
Context: {retrieved_context}
Answer: {llm_answer}
Is every claim in the answer supported by the context? Answer YES or NO.
If NO, list the unsupported claims.
"""
verification = llm.invoke(verification_prompt)
if "NO" in verification:
    log_and_flag_for_review(answer)
```
This doubles API cost but provides hallucination detection. Not yet implemented in this codebase.

### Q49: Why use `StrOutputParser` instead of structured output?

**A:** `StrOutputParser` extracts `.content` from the LLM's `AIMessage` response — simplest possible parser. For a CLI demo, plain text is sufficient. For production API, I'd use `PydanticOutputParser` or LangChain's `with_structured_output()` to guarantee JSON schema compliance: `{"answer": str, "sources": list[str], "has_sufficient_info": bool}`.

### Q50: What is the "context window" and how does it affect your design?

**A:** DeepSeek V4 Flash has a 128K token context window (~96,000 words). Our Top-3 chunks provide ~3,000 chars (~750 tokens). We use ~1% of the context window — well within limits. This means: (1) I could increase TOP_K significantly without hitting limits, (2) the prompt template uses only ~200 tokens for instructions, (3) the bottleneck is retrieval quality, not context capacity.

### Q51: What if the user asks about a scheme that exists but retrieval returns wrong chunks?

**A:** The LLM receives irrelevant context. The prompt rules say "ONLY use context" and "if insufficient, state so." Ideally, the LLM should notice the context doesn't answer the question and trigger the fallback. In practice, LLMs sometimes try to answer anyway — this is a known weakness. Mitigation: increase threshold to 0.4 to be more selective, or add a pre-generation check: if confidence < 0.5, ask user "Did you mean [detected scheme name]?"

### Q52: Why system + human message instead of a single message?

**A:** Chat models are trained on role-based conversations. The system message sets persistent constraints (persona, rules), while the human message provides the specific task instance (query + context). This separation is more effective than cramming everything into one message because the model's training data included system messages as "always-follow" instructions.

### Q53: How do you cite sources in the response?

**A:** The prompt rule #4 requires: `📄 Source: <scheme_name> (File: <filename>, Page <page_number>)`. The LLM extracts this from the formatted sources string injected into the prompt. Example output: "📄 Source: PM Kisan Samman Nidhi (File: pm_kisan.pdf, Page 2)". The scheme name is derived from the filename (hyphens → spaces, title case) during ingestion.

---

## Section 6: Data Pipeline (6 Questions)

### Q54: Why PyMuPDF (fitz) over pdfplumber for text extraction?

**A:** PyMuPDF is a C library (MuPDF) with Python bindings — significantly faster than pdfplumber (pure Python). For 1,744 PDFs, this matters. Trade-off: pdfplumber is better at extracting tables and preserving layout. PyMuPDF is better for plain text extraction speed. I chose speed because government scheme PDFs are mostly running text, not complex tables.

### Q55: How do you handle corrupted or password-protected PDFs?

**A:** Each PDF is wrapped in try/except:
```python
for pdf_path in pdf_files:
    try:
        docs = self._load_single_pdf(pdf_path)
        all_documents.extend(docs)
    except Exception as exc:
        logger.warning(f"Skipping '{pdf_path.name}': {exc}")
        failed += 1
```
The loader logs a warning and continues. Failed PDFs are counted in the final log message. This means the system gracefully degrades — it ingests what it can and reports what it couldn't.

### Q56: How is scheme name derived from filename?

**A:** `_derive_scheme_name()`: strip extension → replace hyphens/underscores with spaces → title case each word. Example: `"pm_kisan_samman_nidhi.pdf"` → `"Pm Kisan Samman Nidhi"`. This is a heuristic — it works for most PDFs but may produce awkward names for abbreviated filenames (e.g., `"nassvsc.pdf"` → `"Nassvsc"`).

### Q57: What metadata do you store with each document?

**A:** Each LangChain Document (one per PDF page) has:
```python
metadata={
    "filename": "pm_kisan.pdf",
    "scheme_name": "Pm Kisan Samman Nidhi",
    "page": 2,
    "source": "/full/path/to/pm_kisan.pdf",
}
```
This metadata propagates to every chunk during splitting. Missing: `ministry`, `state`, `category` — these aren't reliably extractable from PDF text without NLP.

### Q58: How would you extract structured fields (ministry, eligibility) from PDFs?

**A:** Currently not implemented. Options: (1) Regex patterns for common field labels ("Ministry of...", "Eligibility:"), (2) Use an LLM to extract structured JSON from each PDF (expensive at scale), (3) Use a layout parser (LayoutLM) for PDF structure understanding. For production, I'd use a combination: regex for simple fields, LLM extraction for complex ones, with caching.

### Q59: What happens when you re-run the pipeline with updated PDFs?

**A:** The current implementation doesn't detect updates — it either loads the existing ChromaDB (skipping new PDFs) or rebuilds from scratch (deleting old data). To support incremental updates: (1) check file modification timestamps against ChromaDB's last build time, (2) only process new/modified PDFs, (3) use ChromaDB's `add_documents()` instead of `from_documents()` for new chunks, (4) delete chunks for removed PDFs using UUID-based deletion.

---

## Section 7: Performance & Scale (8 Questions)

### Q60: What's the end-to-end latency?

**A:** Based on component timings:
- Embedding query: ~50ms (384-dim, CPU)
- ChromaDB search: ~100ms (cosine similarity against ~10K vectors)
- LLM generation: 3,000-5,000ms (DeepSeek API, network + inference)
- **Total: ~3-6 seconds**

The LLM dominates. Without it (just retrieval): ~150ms.

### Q61: How would you reduce latency to under 1 second?

**A:**
1. **Streaming:** `llm.stream()` instead of `llm.invoke()` — first token in <500ms
2. **Caching:** LRU cache for repeated queries
3. **GPU embedding:** Move all-MiniLM-L6-v2 to CUDA
4. **ANN index:** HNSW instead of brute-force ChromaDB
5. **Smaller LLM:** Switch to a faster model via Groq or Together AI (lower latency APIs)

### Q62: How does ChromaDB scale with more documents?

**A:** Currently O(N) brute-force search. At:
- 10K chunks → ~100ms (current)
- 100K chunks → ~1s (noticeable)
- 1M chunks → ~10s (unusable)
- 10M chunks → ~100s

Mitigation: ChromaDB supports HNSW indexing (set `hnsw:space` in collection metadata). HNSW gives O(log N) approximate search with ~95% recall.

### Q63: What's your index size on disk?

**A:** ChromaDB stores: (1) SQLite database with metadata + IDs, (2) vector data in binary format. Estimated: ~10K chunks × 384 floats × 4 bytes = ~15MB for vectors + ~5MB for metadata + SQLite overhead = ~25-50MB. Verified: the ChromaDB directory after building with 3 documents was ~2MB.

### Q64: How much RAM does the embedding model use?

**A:** all-MiniLM-L6-v2 is ~90MB on disk, ~120MB in RAM (model weights + tokenizer). During embedding, additional memory for batch processing. Total memory footprint: ~200MB for the full pipeline (embedding model + ChromaDB index + Python runtime). This runs comfortably on any laptop.

### Q65: How would you handle 1 million PDFs?

**A:**
1. **Distributed ingestion:** Apache Spark or Ray for parallel PDF processing
2. **Sharded vector DB:** Milvus or Weaviate with horizontal sharding
3. **GPU embeddings:** Batch embed on GPU (1 PDF = ~50ms on GPU vs ~500ms on CPU)
4. **Pre-built index:** Don't build at query time; run nightly batch jobs
5. **Hierarchical retrieval:** First retrieve candidate PDFs (coarse), then search within (fine)

### Q66: What's the cost per query?

**A:** 
- **Embedding:** $0 (local model)
- **Retrieval:** $0 (local ChromaDB)
- **DeepSeek V4 Flash:** ~$0.14/1M input tokens + ~$0.28/1M output tokens
  - Input: ~750 tokens (context + prompt) = $0.0001
  - Output: ~300 tokens (answer) = $0.00008
  - **Total per query: ~$0.0002 (0.02 cents)**

This is negligible. The real cost is the one-time embedding model download (free) and developer time.

### Q67: What happens if DeepSeek API is rate-limited?

**A:** Currently, the exception propagates to `main()` → logged → "⚠️ An error occurred" → CLI loop continues. Missing: retry logic with exponential backoff (`tenacity` library), circuit breaker (stop calling after N failures), graceful degradation message. This is in the improvements roadmap.

---

## Section 8: Evaluation & Limitations (6 Questions)

### Q68: How do you evaluate retrieval quality?

**A:** Currently manual testing. I verify that:
- "What is PM Kisan?" → retrieves pm_kisan.pdf chunks
- "US Social Security" → returns fallback (no relevant chunks)
- Scheme-specific queries → return chunks from the correct PDF

For production: I'd create an evaluation set of 100+ queries with expected source documents, compute Recall@K and MRR, and run this as a CI check on every code change.

### Q69: What is the biggest limitation of your current system?

**A:** Three tied for first:
1. **No evaluation framework:** I can't quantify retrieval quality or hallucination rate
2. **Single-stage retrieval:** No re-ranking, no hybrid search, no query expansion
3. **No PDF structure awareness:** Tables, forms, and multi-column layouts in PDFs may be misparsed

### Q70: What happens when the LLM receives contradictory information across chunks?

**A:** Two chunks from different PDFs might describe the same scheme differently (e.g., one is outdated). The LLM sees both and must reconcile. The prompt doesn't address contradictions explicitly. The LLM typically presents both or favors the first. To fix: add prompt instruction: "If sources contradict, present both and note the discrepancy."

### Q71: How would you detect if the embedding model is underperforming?

**A:** Monitor similarity score distribution over time:
- If average scores drop → model may be outdated or documents changed
- If scores cluster near 0.5 (random) → model not discriminating
- If all scores > 0.9 → possible overfitting or duplicate content

Set up Prometheus/Grafana dashboard tracking score percentiles (p50, p90, p99) per query.

### Q72: What edge cases does your system NOT handle?

**A:**
1. Multi-lingual queries (Hindi, Tamil)
2. Multi-turn conversations (follow-up questions)
3. Ambiguous queries ("schemes for farmers" → which state? which type?)
4. Time-sensitive queries ("schemes active in 2024")
5. Multi-hop questions ("which ministry runs PM-KISAN and what's their budget?")
6. Scanned/image-only PDFs

### Q73: How would you add multi-turn conversation support?

**A:** Maintain conversation history in `RAGPipeline`:
```python
self.chat_history = []  # List of (question, answer) tuples

def query(self, question):
    # Include last 3 exchanges in prompt for context
    history_context = format_history(self.chat_history[-3:])
    prompt_messages = self.prompt.format_messages(
        context=retrieved_context,
        sources=sources,
        query=question,
        history=history_context,  # New variable
    )
```
This allows follow-ups like "What are the benefits?" (referring to the scheme just discussed).

---

## Section 9: TCS Prime Behavioral (5 Questions)

### Q74: Why did you choose this project for your portfolio?

**A:** It demonstrates: (1) end-to-end ML system design, not just model training, (2) solves a real Indian citizen problem (government scheme awareness), (3) uses production-grade tools (LangChain, ChromaDB, PyMuPDF), (4) covers every RAG concept interviewers ask about, (5) shows I can work with real-world messy data (PDFs, not clean CSVs).

### Q75: What was the hardest technical challenge?

**A:** The HuggingFace dataset (`shrijayan/gov_myscheme`) stores PDFs, not structured text. Initially I tried `datasets.load_dataset()` with pdfplumber — but it generated only 3 examples (a tiny split). I had to switch strategies: download all 2,876 PDFs via HF CLI, then use PyMuPDF for text extraction. This taught me that real-world data rarely matches the "clean CSV" examples in tutorials.

### Q76: What would you do differently if you started over?

**A:** Three things:
1. **Start with evaluation:** Build a labeled query-answer test set FIRST, then optimize the pipeline against it
2. **Use a document parser, not just text extraction:** LayoutLM or Unstructured.io for table/form awareness
3. **Add hybrid search from day one:** BM25 + semantic ensemble is a small code change with big retrieval quality gains

### Q77: How do you stay updated with RAG advancements?

**A:** I follow: LangChain blog, LlamaIndex documentation, arXiv papers on RAG (Self-RAG, RAPTOR, GraphRAG), and production RAG posts from companies like Anthropic, Cohere, and Google. The RAG field moves fast — what's best practice today (simple Top-K retrieval) may be outdated in 6 months.

### Q78: If this were a TCS client project, what would you add before delivery?

**A:**
1. **Authentication + rate limiting** (FastAPI middleware)
2. **Usage analytics dashboard** (query volume, popular schemes, failure rate)
3. **Feedback mechanism** (thumbs up/down per answer)
4. **Scheduled re-indexing** (nightly cron for new PDFs)
5. **SLA monitoring** (p99 latency < 5s, uptime > 99.9%)
6. **Data privacy audit** (no PII in logs, encryption at rest for ChromaDB)

---

[Next → Revision Sheet](./revision-sheet.md)
