# Government Schemes — RAG Application

A clean, modular **Retrieval-Augmented Generation (RAG)** pipeline built for
understanding RAG architecture using the **shrijayan/gov_myscheme** dataset
from Hugging Face — comprehensive Indian Government Scheme details.

---

## Architecture

```
shrijayan/gov_myscheme (HuggingFace Dataset)
        ↓
   datasets library
        ↓
  LangChain Documents  (scheme_name, eligibility, benefits, etc.)
        ↓
  RecursiveCharacterTextSplitter  (chunk_size=1000, overlap=200)
        ↓
  all-MiniLM-L6-v2 Embeddings  (384-dim, local)
        ↓
  Persistent ChromaDB  (chroma_db/)
        ↓
     Retriever  (top_k=3, similarity threshold)
        ↓
  ChatPromptTemplate  (grounded, no-hallucination)
        ↓
  DeepSeek LLM  (deepseek-v4-flash / deepseek-v4-pro, temp=0.1)
        ↓
   Grounded Answer  + Sources + Confidence Score
```

---

## Tech Stack

| Component       | Library / Tool                               |
|-----------------|----------------------------------------------|
| Language        | Python 3.10+                                 |
| Package manager | uv                                           |
| LLM             | DeepSeek API (deepseek-v4-flash / deepseek-v4-pro)   |
| LangChain       | langchain, langchain-core, langchain-community |
| Embeddings      | sentence-transformers/all-MiniLM-L6-v2 (local)|
| Dataset         | shrijayan/gov_myscheme (HuggingFace datasets) |
| Vector store    | ChromaDB (persistent, on-disk)               |
| Env management  | python-dotenv                                |

---

## Project Structure

```
gov-scheme-rag/
├── src/
│   ├── __init__.py       # makes src a Python package
│   ├── config.py         # all tuneable constants
│   ├── data_loader.py    # HF dataset → LangChain Documents
│   ├── embedding.py      # text splitting + embedding generation
│   ├── vector_store.py   # ChromaDB load / build / persist
│   ├── retriever.py      # similarity search + threshold filtering
│   ├── prompt.py         # RAG ChatPromptTemplate
│   └── rag.py            # end-to-end pipeline orchestrator
├── chroma_db/            # auto-created on first run
├── app.py                # main entrypoint (smart init + CLI)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

### 1. Clone / open the project

```bash
cd gov-scheme-rag
```

### 2. Create virtual environment & install dependencies

```bash
uv venv
uv pip install -r requirements.txt
```

### 3. Create `.env` file

```bash
# .env
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# Optional — override the default model
# DEEPSEEK_MODEL=deepseek-v4-flash
# DEEPSEEK_MODEL=deepseek-v4-pro
```

Get a free API key at: https://platform.deepseek.com/api_keys

---

## Usage

### Just run app.py — it handles everything

```bash
uv run python app.py
```

**First run**: Downloads the `shrijayan/gov_myscheme` dataset from Hugging Face,
chunks it, generates 384-dimensional embeddings, and persists to `chroma_db/`.

**Subsequent runs**: Loads the existing ChromaDB instantly from disk — no
re-downloading, no re-embedding.

### Example session

```
═══════════════════════════════════════════════════════════════════════
  🇮🇳  Indian Government Schemes — RAG Assistant
  Powered by DeepSeek + ChromaDB + sentence-transformers
  Dataset: shrijayan/gov_myscheme
  Ask about any scheme. Type 'exit' to quit.
═══════════════════════════════════════════════════════════════════════

Ask about a government scheme (or type 'exit'): What is PM Kisan and who is eligible?

──────────────────────────────────────────────────────────────────────

📋  QUESTION
    What is PM Kisan and who is eligible?

📂  RETRIEVED SOURCES
    [1]  PM Kisan Samman Nidhi  |  Ministry of Agriculture  |  Central  |  Score: 0.8912
    [2]  PM Kisan Yojana        |  Ministry of Agriculture  |  Central  |  Score: 0.7845

    📊  Confidence Score: 0.8379

💬  ANSWER

    PM Kisan Samman Nidhi is a Central Sector scheme that provides
    income support to all landholding farmer families ...
    📄 Source: PM Kisan Samman Nidhi (Ministry: Ministry of Agriculture, Category: Central)

──────────────────────────────────────────────────────────────────────
```

---

## Configuration

All settings are in [`src/config.py`](src/config.py):

| Constant               | Default                                | Description                          |
|------------------------|----------------------------------------|--------------------------------------|
| `HF_DATASET_NAME`      | `shrijayan/gov_myscheme`              | HuggingFace dataset identifier       |
| `VECTOR_DB_PATH`       | `chroma_db/`                          | ChromaDB persistence directory       |
| `EMBEDDING_MODEL`      | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model (384-dim)  |
| `MODEL_NAME`           | `deepseek-chat`                        | DeepSeek model (overridable via env)  |
| `TEMPERATURE`          | `0.1`                                 | LLM temperature (low = factual)      |
| `MAX_TOKENS`           | `1024`                                | Max response tokens                  |
| `CHUNK_SIZE`           | `1000`                                | Characters per chunk                 |
| `CHUNK_OVERLAP`        | `200`                                 | Overlap between chunks               |
| `TOP_K`                | `3`                                   | Chunks retrieved per query           |
| `SIMILARITY_THRESHOLD` | `0.3`                                 | Minimum score to keep a result       |

To use a different DeepSeek model, set in `.env`:

```bash
DEEPSEEK_MODEL=deepseek-reasoner
```

---

## Module Responsibilities

| File              | Single Responsibility                                        |
|-------------------|--------------------------------------------------------------|
| `config.py`       | All constants and environment variable loading               |
| `data_loader.py`  | HF dataset loading → LangChain Document conversion           |
| `embedding.py`    | Text splitting + embedding model + numpy embedding arrays    |
| `vector_store.py` | ChromaDB create / load / persist with UUIDs                  |
| `retriever.py`    | Similarity search + threshold filtering                      |
| `prompt.py`       | RAG prompt template (grounded, no-hallucination)             |
| `rag.py`          | Orchestration: retriever → prompt → DeepSeek → structured result |
| `app.py`          | Entry point: smart init + interactive CLI                    |

---

## RAG Pipeline Evaluation

### Methodology

The RAG pipeline was evaluated on a **gold-standard dataset** of 10 hand-crafted
question-answer pairs sourced from specific scheme PDFs (e.g., *25-ciss.pdf*,
*aabcs.pdf*). We evaluated the **retriever** and **generator** independently to
isolate search quality from LLM generation quality.

### Metrics

| Metric                     | Score      | Description |
|----------------------------|------------|-------------|
| **Retrieval Recall @ 3**   | **70.0%**  | Percentage of test cases where at least one of the top-3 retrieved chunks came from the correct source PDF and page. |
| **Avg. LLM Judge Score**   | **3.80 / 5** | LLM-as-a-judge correctness score averaged across all 10 test cases (scale: 1 = incorrect, 5 = excellent). |

**Grade Distribution** (1 = incorrect, 5 = excellent):

| Grade | Count |
|-------|-------|
| 5 (Excellent) | 5 |
| 4 (Good)      | 1 |
| 3 (Acceptable)| 2 |
| 2 (Poor)      | 1 |
| 1 (Incorrect) | 1 |

### Resume Highlights

- **Designed and executed an automated RAG evaluation framework** using a
  10-question gold-standard dataset, achieving **70% retrieval recall @ 3** and
  an **average LLM-as-a-judge correctness score of 3.80/5** on a production
  government-schemes Q&A pipeline.
- **Implemented dual-metric evaluation** combining deterministic
  source-document/page recall with LLM-based semantic correctness grading,
  isolating retriever quality from generator performance across 723 scheme PDFs.
- **Built a reproducible evaluation harness** (`run_evaluation.py`) with
  per-case diagnostics, grade distribution analysis, and auto-generated
  documentation updates, enabling continuous benchmarking of RAG pipeline
  improvements.

## Verification Checklist

1. **Data Ingestion**: HF records are fetched and wrapped in LangChain Documents
   with scheme metadata (scheme_name, ministry, category, state, target_audience).
2. **Vector Dimension**: Embeddings are 384-dimensional, matching
   `all-MiniLM-L6-v2` and ChromaDB's index.
3. **No Regeneration**: Running `app.py` a second time loads from `chroma_db/`
   instantly without re-downloading.
4. **Grounded Generation**: Queries about topics not in the dataset receive a
   graceful "I cannot find sufficient information" response — no hallucination.

```
