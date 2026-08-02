"""Quick debug: test imports and initialisation."""
import sys
print("Python started", flush=True)

print("Importing config...", flush=True)
from src.config import DEEPSEEK_API_KEY
print(f"API key: {'SET' if DEEPSEEK_API_KEY else 'MISSING'}", flush=True)

print("Importing EmbeddingPipeline...", flush=True)
from src.embedding import EmbeddingPipeline
print("Creating EmbeddingPipeline...", flush=True)
ep = EmbeddingPipeline()
print("EmbeddingPipeline created!", flush=True)

print("Importing vector store...", flush=True)
from src.vector_store import SchemeVectorStore
store = SchemeVectorStore(embedding_pipeline=ep)
print(f"DB exists: {store.db_exists()}", flush=True)

print("Loading DB...", flush=True)
db = store.load()
print(f"DB loaded, count: {db._collection.count()}", flush=True)

print("Creating retriever...", flush=True)
from src.retriever import Retriever
retriever = Retriever(db=db)

print("Creating RAG pipeline...", flush=True)
from src.rag import RAGPipeline
rag = RAGPipeline(retriever=retriever)

print("ALL IMPORTS OK!", flush=True)
