# Architecture

The code has two paths: indexing and answering.

## Indexing

`build_vectordb.py` calls `load_all_documents()` to read non-empty pages from PDFs below `data/text_data`. Each page becomes a LangChain `Document` with its filename, derived scheme name, page number, and local path.

`EmbeddingPipeline` splits pages with `RecursiveCharacterTextSplitter` and uses `sentence-transformers/all-MiniLM-L6-v2` on CPU. `SchemeVectorStore` stores the resulting chunks in a persistent Chroma collection under `chroma_db`.

## Answering

Both application entry points call `create_rag_pipeline()` in `src/bootstrap.py`. It loads the saved Chroma collection if its directory exists; otherwise it indexes the local PDFs.

For each question, `Retriever` asks Chroma for up to three matches and drops results below the configured score threshold. `RAGPipeline` sends the remaining chunk text, metadata, and question to `ChatOpenAI`, configured with the DeepSeek base URL and credentials from `.env`. It returns the model text together with the retrieval metadata and scores.

The Streamlit app stores its pipeline and conversation in Streamlit session state. The command-line app keeps one pipeline for its interactive loop.

## Configuration

`src/config.py` contains paths, model names, chunk settings, and retrieval settings. `DEEPSEEK_API_KEY` and the optional `DEEPSEEK_MODEL` are read from `.env`.
