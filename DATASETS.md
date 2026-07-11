# Dataset Documentation

This document describes the datasets used in the Government Scheme Finder RAG application, their purpose, schema, and preprocessing steps.

## Data Sources

The application consolidates Indian government schemes from two primary Hugging Face datasets:

1. **`satyajitdas/bharatschemes-v1`**
   - **Origin & Purpose:** Contains structured data about various Indian government schemes, primarily focused on eligibility, benefits, and application processes.
   - **Format:** JSON.
   - **Language:** English.

2. **`shrijayan/gov_myscheme`**
   - **Origin & Purpose:** Scraped/extracted from the official MyScheme portal. Contains detailed descriptions and metadata for numerous state and central government schemes.
   - **Format:** Parquet.
   - **Language:** Multilingual (primarily English, with regional applicability metadata).

## Processing Pipeline

The script `server/scraper/prepare_datasets.py` orchestrates the data pipeline, standardising these disparate sources into a unified schema for our vector store.

### Preprocessing and Cleaning Steps
- **Format Normalisation:** Converts Parquet to JSON and unifies field names (e.g., standardising `scheme_name` vs `title`).
- **Data Deduplication:** Merges schemes based on exact name matches, preferring descriptions from `bharatschemes-v1` while enriching with application links from `gov_myscheme`.
- **Text Chunking:** Constructs a unified `chunk_text` field for each scheme. This text explicitly concatenates the Scheme Name, Ministry, Description, Eligibility, Benefits, and Application details to maximise context retrieval during the embedding process.
- **Output:** The final, cleaned, and deduplicated dataset is written to `processed/schemes_rag_chunks.json`.

### Schema

The resulting `processed/schemes_rag_chunks.json` contains a list of objects conforming to the following structure:

| Field | Type | Description |
|---|---|---|
| `scheme_id` | String | A unique identifier for the scheme (usually a hashed or sanitized version of the scheme name). |
| `chunk_text` | String | The comprehensive, concatenated text representation of the scheme used to generate the embedding vector. |
| `metadata` | Object | Structured data used for filtering and display in the frontend UI. |
| `metadata.name` | String | The official name of the scheme. |
| `metadata.ministry` | String | The government ministry or department responsible for the scheme. |
| `metadata.beneficiaries` | List[String] | Target demographics (e.g., "Farmers", "Women", "Students"). |
| `metadata.state` | String | The specific state the scheme applies to, or "Central" for nationwide schemes. |
| `metadata.url` | String | (Formerly `application_link`) The URL to apply for the scheme or view official details. |

## Multilingual Considerations

The RAG application is designed to be accessible to a diverse user base.
- **Dataset Bias:** The raw data from Hugging Face is overwhelmingly in English.
- **Embedding Model:** We use `paraphrase-multilingual-MiniLM-L12-v2`. This model supports 50+ languages, enabling the system to match queries written in Hindi, Tamil, or Hinglish directly against the English scheme descriptions in the vector store without requiring explicit translation steps.

## Assumptions & Limitations
- **Staleness:** Government schemes frequently change. The datasets represent a point-in-time snapshot. The pipeline relies on rebuilding from Hugging Face if the upstream dataset is updated.
- **Deduplication Bias:** Simple name-matching is used for deduplication. Highly similar schemes with different naming conventions may result in duplicates in the vector store.
- **Field Completeness:** Not all datasets have comprehensive metadata (e.g., missing URLs or ministries). The fallback behavior handles missing fields gracefully by omitting them or providing generic placeholders.
