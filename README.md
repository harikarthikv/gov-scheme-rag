# Government scheme RAG assistant

A Streamlit application that answers questions from Indian government scheme PDFs
using MiniLM embeddings, ChromaDB, hybrid BM25/vector retrieval and DeepSeek.
Answers include numbered citations and the retrieved PDF excerpts.

## Setup

Use Python 3.14 and run commands from the repository root:

```powershell
uv sync --frozen
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Set `DEEPSEEK_API_KEY` in `.env`. Keep an existing environment file; it is ignored
by Git. Without uv, create a virtual environment and run `python -m pip install -e .`.
Dependencies are declared only in [pyproject.toml](pyproject.toml); [uv.lock](uv.lock)
records the reproducible environment.

Place the source PDFs under `data/text_data/`. The original collection contains
2,876 PDFs from Hugging Face `shrijayan/gov_myscheme`; PDFs and generated indexes
are not included in Git. Embedding weights download on first use. After caching
them, `RAG_LOCAL_MODELS_ONLY=1` enables offline embedding initialization.

## Run

```powershell
uv run streamlit run app.py
```

Select **Load assistant**, then enter a self-contained question. Streamlit is the
only user interface. Source numbers in the answer match the filename, 1-based page
and excerpt shown below it. Chat history is displayed but is not sent to the model.

The app reuses the configured index or builds it if missing. To ingest explicitly:

```powershell
uv run python -m scripts.ingest
```

An existing collection is never overwritten. After changing PDFs, embeddings or
chunking, set a new `RAG_COLLECTION` in `.env`, ingest and restart the app. Streamlit
file watching is disabled because scanning Transformers' lazy modules imports
unrelated optional backends; restart the server after changing code.

## Pipeline and configuration

```text
PDF pages -> overlapping chunks + stable IDs -> normalized MiniLM -> Chroma
Question -> dense + BM25 candidates -> deduplication + RRF -> optional re-ranking
         -> labeled excerpts -> DeepSeek -> citation validation -> answer
```

The default embedding model is `sentence-transformers/all-MiniLM-L6-v2` on CPU.
Chunks contain 1,000 characters with 200-character overlap. Hybrid retrieval fuses
30 candidates per branch with equal weights and an RRF constant of 60, returning
5 contexts. A cross-encoder is available but disabled: it did not improve ranking
in the development benchmark.

Set overrides in [.env.example](.env.example)'s corresponding `.env` entries:

| Setting | Default / purpose |
|---|---|
| `RAG_RETRIEVAL_MODE` | `hybrid`; also `baseline`, `dense`, `bm25` |
| `RAG_TOP_K`, `RAG_CANDIDATE_K` | `5`, `30` |
| `RAG_DENSE_WEIGHT`, `RAG_RRF_CONSTANT` | `0.5`, `60` |
| `RAG_SIMILARITY_THRESHOLD` | `0.3`, used only by baseline/dense retrieval |
| `RAG_RERANKER_MODEL` | Empty; optional Sentence Transformers cross-encoder |
| `RAG_EMBEDDING_MODEL`, `RAG_EMBEDDING_DEVICE` | MiniLM, `cpu` |
| `RAG_QUERY_PREFIX`, `RAG_DOCUMENT_PREFIX` | Empty; prefixes for models such as E5 |
| `RAG_CHUNK_SIZE`, `RAG_CHUNK_OVERLAP` | `1000`, `200` |
| `RAG_DATA_DIR`, `RAG_DB_PATH` | `data/text_data`, `chroma_db` |
| `RAG_COLLECTION` | `gov_schemes_local_dataset` |
| `DEEPSEEK_MODEL`, `RAG_JUDGE_MODEL` | Answer model and optional evaluation judge |

Relative corpus/index paths resolve from the project root. Process environment
values override `.env`. Model/chunk settings must match the saved collection.
New indexes use cosine distance and record compatibility/completeness metadata;
the original index retains L2. Dense, BM25, RRF and cross-encoder scores are ranking
values, not confidence, and do not share a meaningful acceptance threshold.

## Evaluate and test

The committed [benchmark](evaluation/schemes_v1.jsonl) has 110 evidence-backed
questions from 59 PDFs, with 25 development and 85 held-out test cases.

```powershell
# Retrieval only; no API key required
uv run python -m scripts.evaluate --output evaluation/results/retrieval

# Generation plus structured LLM judging; incurs DeepSeek charges
uv run python -m scripts.evaluate --generate --modes hybrid --ks 1 3 5 --workers 4 --output evaluation/results/generation

# Offline regression and headless Streamlit tests
uv run python -m unittest discover -s tests -v

# Real-PDF ingestion/reload/retrieval smoke test; add --generate for DeepSeek
uv run python -m scripts.smoke_pipeline

# Check every indexed chunk against the local source PDFs
uv run python -m scripts.audit_corpus
```

Use `--split dev` for tuning, then `--split test`. Runs save configuration, hashes,
case-level results, errors and metric denominators under the ignored output
folder. Repeating the same command resumes a compatible run; changed settings need
a new output directory. Failed calls never become invented scores.

[Evaluation methodology](evaluation/README.md) describes the reusable dataset
schema, metric definitions, dataset regeneration and report commands.
[Recorded results](evaluation/RESULTS.md) distinguish measured improvements from
limitations. The recorded page Recall@3 increased from 0.8727 to 0.9091; this
cleanup does not change the retrieval or generation algorithms.

## Project layout

```text
app.py                    Streamlit interface
src/gov_scheme_rag/        Ingestion, embeddings, storage, retrieval, generation, metrics
scripts/                  Ingestion, evaluation and reproducibility commands
tests/                    Offline regression and Streamlit tests
evaluation/               Reviewed benchmark, seed data, methodology and result snapshot
.streamlit/config.toml    Streamlit server configuration
pyproject.toml / uv.lock   Dependencies and package configuration
data/ / chroma_db/         Local-only corpus and index (ignored)
```

## Limits

PDF extraction has no OCR or table reconstruction. Archived PDFs can contain
outdated, contradictory or malformed information. Character splitting may omit
necessary context or exceed the embedding model's token limit. Verify current
policy against official sources.

Citation validation checks source indices, not factual entailment. The prompt
requires grounding and abstention, but the model and evaluation judge can still
be wrong. The benchmark is English, small and often scheme-specific; it does not
establish multilingual or general citizen-query accuracy. BM25 holds the corpus
in memory, and source edits require explicit auditing/re-ingestion. This is a local
MVP without authentication, rate limiting or deployment hardening.

Secrets, PDFs, database files, temporary indexes and raw evaluation output are
ignored. Add any custom output/index directory to `.gitignore` before publishing.
