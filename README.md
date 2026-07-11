# Indian Government Schemes — RAG Application

A clean, modular **Retrieval-Augmented Generation (RAG)** pipeline built for
understanding RAG architecture using real Indian Government Scheme documents.

---

## Architecture

```
Government PDFs (2,876 docs)
        ↓
   PyMuPDF (fitz)
        ↓
  LangChain Documents
        ↓
  RecursiveCharacterTextSplitter
        ↓
  all-MiniLM-L6-v2 Embeddings
        ↓
  Persistent ChromaDB
        ↓
     Retriever
        ↓
  ChatPromptTemplate
        ↓
  Google Gemini 2.5 Flash
        ↓
   Grounded Answer
```

---

## Tech Stack

| Component       | Library / Tool                          |
|-----------------|-----------------------------------------|
| Language        | Python 3.13                             |
| Package manager | uv                                      |
| LLM             | Google Gemini 2.5 Flash (via Gemini API)|
| LangChain       | langchain, langchain-core, langchain-community |
| Embeddings      | sentence-transformers/all-MiniLM-L6-v2 (local) |
| PDF parsing     | PyMuPDF (fitz)                          |
| Vector store    | ChromaDB (persistent, on-disk)          |
| Env management  | python-dotenv                           |

---

## Project Structure

```
indian-gov-rag/
├── src/
│   ├── __init__.py       # makes src a Python package
│   ├── config.py         # all tuneable constants
│   ├── data_loader.py    # PDF → LangChain Documents (PyMuPDF)
│   ├── embedding.py      # text splitting + HuggingFace embeddings
│   ├── vector_store.py   # ChromaDB load-or-create
│   ├── retriever.py      # similarity search + threshold filtering
│   ├── prompt.py         # RAG ChatPromptTemplate
│   └── rag.py            # end-to-end pipeline orchestrator
├── data/
│   └── gov_myscheme/     # ~2,876 government scheme PDFs
├── chroma_db/            # auto-created after running index.py
├── index.py              # one-time indexing script
├── app.py                # interactive CLI chatbot
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone / open the project

```bash
cd indian-gov-rag
```

### 2. Install dependencies

```bash
uv pip install -r requirements.txt
```

### 3. Create `.env` file

```bash
# .env
GOOGLE_API_KEY=your_google_ai_api_key_here

# Optional — override the default model
# GEMINI_MODEL=gemini-2.5-flash
```

Get a free API key at: https://aistudio.google.com/app/apikey

---

## Usage

### Step 1 — Build the Vector Index (run once)

```bash
uv run python index.py
```

This will:
- Load all 2,876 PDFs from `data/gov_myscheme/`
- Chunk them into overlapping segments
- Generate embeddings locally using `all-MiniLM-L6-v2`
- Persist the index to `chroma_db/`

> ⚠️ This takes a while (~15–30 min depending on your machine). Run it only once.
> The index is persisted — subsequent runs will load from disk instantly.

### Step 2 — Start the CLI Chatbot

```bash
uv run python app.py
```

Example session:

```
═══════════════════════════════════════════════════════════════════════
  🇮🇳  Indian Government Schemes — RAG Assistant
  Ask about any scheme. Type 'exit' to quit.
═══════════════════════════════════════════════════════════════════════

❓  Your Question: What is PM Kisan Samman Nidhi and who is eligible?

──────────────────────────────────────────────────────────────────────

📋  QUESTION
    What is PM Kisan Samman Nidhi and who is eligible?

📂  RETRIEVED SOURCES
    [1]  pmkisan.pdf  |  Page 2  |  Score: 0.8412
    [2]  pmkisan.pdf  |  Page 3  |  Score: 0.7903
    ...

💬  ANSWER

    PM Kisan Samman Nidhi (PM-KISAN) is a Central Sector scheme ...
    📄 Source: pmkisan.pdf, Page 2

──────────────────────────────────────────────────────────────────────
```

---

## Configuration

All settings are in [`src/config.py`](src/config.py):

| Constant               | Default                          | Description                          |
|------------------------|----------------------------------|--------------------------------------|
| `DATA_DIR`             | `data/gov_myscheme`              | Root PDF directory                   |
| `VECTOR_DB_PATH`       | `chroma_db/`                     | ChromaDB persistence directory       |
| `EMBEDDING_MODEL`      | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model        |
| `MODEL_NAME`           | `gemini-2.5-flash`               | Gemini model (overridable via env)   |
| `TEMPERATURE`          | `0.1`                            | LLM temperature (low = factual)      |
| `CHUNK_SIZE`           | `1000`                           | Characters per chunk                 |
| `CHUNK_OVERLAP`        | `200`                            | Overlap between chunks               |
| `TOP_K`                | `5`                              | Chunks retrieved per query           |
| `SIMILARITY_THRESHOLD` | `0.3`                            | Minimum score to keep a result       |

To use a different Gemini model, set in `.env`:

```bash
GEMINI_MODEL=gemini-2.5-pro
```

---

## Module Responsibilities

| File              | Single Responsibility                                  |
|-------------------|--------------------------------------------------------|
| `config.py`       | All constants and environment variable loading         |
| `data_loader.py`  | PDF discovery, text extraction, Document creation      |
| `embedding.py`    | Text splitting + embedding model initialisation        |
| `vector_store.py` | ChromaDB create-or-load logic                          |
| `retriever.py`    | Similarity search + threshold filtering                |
| `prompt.py`       | RAG prompt template definition                         |
| `rag.py`          | Orchestration: retriever → prompt → LLM → response     |
| `index.py`        | Entry point: build the vector index (run once)         |
| `app.py`          | Entry point: interactive CLI chatbot                   |

---

## Rebuilding the Index

If you want to re-index (e.g. after adding new PDFs):

```bash
rm -rf chroma_db/
uv run python index.py
```
