"""Minimal test: just import langchain_huggingface and load the model."""
import os
import sys
print("1. Starting", flush=True)

# Skip online checks
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

print("2. Importing langchain_huggingface...", flush=True)
from langchain_huggingface import HuggingFaceEmbeddings
print("3. Import OK, creating embeddings object...", flush=True)

emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
print("4. Embeddings object created! Testing...", flush=True)

result = emb.embed_query("test query")
print(f"5. Embedding works! Dimension: {len(result)}", flush=True)
