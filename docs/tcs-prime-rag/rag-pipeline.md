# RAG Pipeline — End-to-End Architecture

> **What the interviewer wants:** You can draw the pipeline on a whiteboard and explain every stage.

---

## Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        OFFLINE (INDEXING)                           │
│                                                                     │
│  1,744 PDFs           PyMuPDF          LangChain       Recursive    │
│  (data/gov_myscheme) ──────►  Page     ──────►        Character    │
│  text_data/*.pdf       (fitz)  Texts    Documents     TextSplitter  │
│                                                                     │
│                               ┌──────────────────────────┐         │
│                               │  1000 chars / 200 overlap │         │
│                               │  ["\n\n","\n",". "," ",""]│         │
│                               └──────────┬───────────────┘         │
│                                          │                          │
│                               ┌──────────▼───────────────┐         │
│                               │  all-MiniLM-L6-v2        │         │
│                               │  384-dim embeddings (CPU) │         │
│                               └──────────┬───────────────┘         │
│                                          │                          │
│                               ┌──────────▼───────────────┐         │
│                               │  ChromaDB                │         │
│                               │  persistent (chroma_db/) │         │
│                               │  collection: gov_schemes │         │
│                               │  UUIDs per chunk          │         │
│                               └──────────────────────────┘         │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        ONLINE (QUERY)                                │
│                                                                     │
│  User Query          all-MiniLM-L6-v2       Cosine Similarity       │
│  "What schemes    ──────►  384-dim    ──────►  Top-3 chunks         │
│   for farmers?"          embedding           (ChromaDB)             │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │  Threshold filter: score ≥ 0.3                           │      │
│  │  If 0 results → graceful "not found" fallback            │      │
│  └──────────────────────────┬───────────────────────────────┘      │
│                             │                                       │
│  ┌──────────────────────────▼───────────────────────────────┐      │
│  │  ChatPromptTemplate                                      │      │
│  │  System: "You are an expert Government Scheme Assistant" │      │
│  │  Rules: NO outside knowledge, cite sources, no guessing  │      │
│  │  Context: {retrieved chunks}                             │      │
│  │  Sources: {filename + page numbers}                      │      │
│  │  Query: {user question}                                  │      │
│  └──────────────────────────┬───────────────────────────────┘      │
│                             │                                       │
│  ┌──────────────────────────▼───────────────────────────────┐      │
│  │  DeepSeek V4 Flash (ChatOpenAI)                          │      │
│  │  base_url = https://api.deepseek.com                     │      │
│  │  temperature = 0.1, max_tokens = 1024                    │      │
│  └──────────────────────────┬───────────────────────────────┘      │
│                             │                                       │
│  ┌──────────────────────────▼───────────────────────────────┐      │
│  │  Structured Response                                     │      │
│  │  {                                                       │      │
│  │    "answer": "PM-KISAN provides...",                     │      │
│  │    "sources": [{scheme_name, filename, page}],           │      │
│  │    "similarity_scores": [0.89, 0.78, 0.65],             │      │
│  │    "confidence_score": 0.7733                            │      │
│  │  }                                                       │      │
│  └──────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
```

## Pipeline Stages (Code Trace)

### Stage 1: Data Ingestion (`src/data_loader.py`)

```python
# config.py
DATA_DIR = PROJECT_ROOT / "data" / "gov_myscheme" / "text_data"

# data_loader.py
class SchemeDataLoader:
    def load(self) -> list[Document]:
        pdf_files = sorted(self.data_dir.rglob("*.pdf"))   # 1,744 PDFs
        for pdf_path in pdf_files:
            with fitz.open(pdf_path) as pdf:               # PyMuPDF
                for page in pdf:
                    text = page.get_text()                  # extract text
                    Document(page_content=text, metadata={...})
```

**Key decisions:**
- PyMuPDF over pdfplumber: faster (C library), handles malformed PDFs better
- One Document per PDF **page** (not per PDF file): preserves page-level granularity for source citation
- Metadata includes `filename`, `page`, `scheme_name` (derived from filename)

### Stage 2: Chunking (`src/embedding.py` → `split_documents()`)

```python
RecursiveCharacterTextSplitter(
    chunk_size=1000,        # ~250 words per chunk
    chunk_overlap=200,      # 20% overlap
    separators=["\n\n", "\n", ". ", " ", ""]
)
```

**How it works:** Tries to split at paragraph boundaries first (`\n\n`), then newlines, then sentences (`. `), then words (` `), and finally character-by-character (`""`). This preserves semantic coherence.

### Stage 3: Embedding (`src/embedding.py`)

```python
HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}   # L2 normalization
)
```

**Properties:** 384-dim dense vectors, normalized to unit length (cosine similarity becomes dot product).

### Stage 4: Vector Storage (`src/vector_store.py`)

```python
Chroma.from_documents(
    documents=chunks,
    embedding=self.embeddings,
    collection_name="gov_schemes",
    persist_directory="./chroma_db",
    ids=[str(uuid.uuid4()) for _ in chunks],   # traceable UUIDs
)
```

**Persistence:** ChromaDB writes to `chroma_db/chroma.sqlite3` on disk. On restart, `db_exists()` checks the directory and `load()` reopens the collection.

### Stage 5: Retrieval (`src/retriever.py`)

```python
raw_results = db.similarity_search_with_relevance_scores(
    query=query,
    k=TOP_K,       # k=3
)
# Filter: score >= 0.3
```

**Distance metric:** Cosine similarity (implied by normalized embeddings). ChromaDB returns relevance scores in [0, 1] where 1.0 = perfect match.

### Stage 6: Generation (`src/rag.py`)

```python
ChatOpenAI(
    model="deepseek-v4-flash",
    temperature=0.1,         # near-deterministic
    max_tokens=1024,
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
)
```

**Prompt strategy:** System message enforces strict rules (no outside knowledge, cite sources, admit uncertainty). The human message provides context + question.

### Stage 7: Fallback Handling

```python
if not retrieved:
    return {"answer": "I cannot find sufficient information...",
            "sources": [], "confidence_score": 0.0}
```

No LLM call when retrieval returns empty — avoids hallucination entirely.

---

## File-to-Stage Mapping

| File | Stage | Responsibility |
|---|---|---|
| `src/data_loader.py` | Ingestion | PDF → LangChain Documents |
| `src/embedding.py` | Chunking + Embedding | Split + 384-dim vectors |
| `src/vector_store.py` | Storage | ChromaDB create/load/persist |
| `src/retriever.py` | Retrieval | Top-K similarity + threshold filter |
| `src/prompt.py` | Prompt | ChatPromptTemplate with grounding rules |
| `src/rag.py` | Generation | LLM call + structured response |
| `src/config.py` | Configuration | Single source of truth for all constants |
| `app.py` | Orchestration | Smart init + CLI loop |

---

[Next → Design Decisions](./design-decisions.md)
