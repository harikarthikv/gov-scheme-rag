# Government Scheme Finder — vector store builder
# Reads processed/schemes_rag_chunks.json, embeds each scheme,
# stores in ChromaDB at server/vectorstore/chroma_db/
# Embedding model: paraphrase-multilingual-MiniLM-L12-v2 (local, no API key required)
# Run once: python -m server.vectorstore.build_vectorstore
# Prerequisite: run_pipeline.sh (downloads and prepares the scheme data)

import json
import logging
import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "processed", "schemes_rag_chunks.json")

def build():
    with open(DATA_PATH, encoding="utf-8") as f:
        schemes = json.load(f)

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Delete existing collection if rebuilding
    try:
        client.delete_collection("schemes")
    except Exception:
        pass

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    collection = client.create_collection("schemes", embedding_function=embedding_fn)

    documents, metadatas, ids = [], [], []

    for scheme in schemes:
        doc = scheme["chunk_text"]
        documents.append(doc)

        meta = scheme["metadata"]
        meta["url"] = scheme.get("application_link", "") # Keep url for compatibility
        if "beneficiaries" in meta and isinstance(meta["beneficiaries"], list):
            meta["beneficiaries"] = ", ".join(meta["beneficiaries"])

        metadatas.append(meta)
        ids.append(scheme["scheme_id"])

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    logger.info("Stored %d schemes in ChromaDB at %s", len(documents), CHROMA_PATH)

if __name__ == "__main__":
    build()
