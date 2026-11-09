"""Build and query an isolated real-PDF index without touching the main collection."""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    # Keep the inspection index outside the repository and the source corpus.
    work = Path(tempfile.mkdtemp(prefix="gov-scheme-rag-"))
    pdfs = work / "pdfs"
    pdfs.mkdir()
    for name in ["25-ciss.pdf", "aaelss.pdf"]:
        shutil.copy2(next((root / "data/text_data").rglob(name)), pdfs / name)
    os.environ.update(RAG_DATA_DIR=str(pdfs), RAG_DB_PATH=str(work / "index"),
                      RAG_COLLECTION="smoke_v1")
    from gov_scheme_rag.data_loader import SchemeDataLoader
    from gov_scheme_rag.embedding import EmbeddingPipeline
    from gov_scheme_rag.vector_store import SchemeVectorStore
    from gov_scheme_rag.retriever import Retriever
    from gov_scheme_rag.config import RetrievalConfig
    loader = SchemeDataLoader(pdfs)
    pages = loader.load()
    store = SchemeVectorStore(EmbeddingPipeline())
    database = store.build_from_documents(pages)
    loaded = store.load()
    assert loaded._collection.count() == database._collection.count() > 0
    question = "What is the maximum subsidy under Assam Abhinandan Education Loan Subsidy Scheme?"
    retriever = Retriever(loaded, RetrievalConfig(top_k=3))
    retrieved = retriever.retrieve(question)
    assert any(item["metadata"]["filename"] == "aaelss.pdf" for item in retrieved)
    assert all(item["metadata"].get("chunk_id") for item in retrieved)
    report = {"status": "ok", "loader": loader.stats, "chunks": loaded._collection.count(),
              "metadata": loaded._collection.metadata, "retrieved": [i["metadata"] for i in retrieved]}
    if args.generate:
        from gov_scheme_rag.rag import RAGPipeline, FALLBACK_ANSWER
        result = RAGPipeline(retriever).answer_from_retrieved(question, retrieved)
        report["answer"] = result["answer"]
        report["citation_validation"] = result["citation_validation"]
        assert result["answer"] != FALLBACK_ANSWER and result["citation_validation"]["valid"]
    destination = root / "evaluation/results/smoke.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Isolated index retained for inspection: {work}")


if __name__ == "__main__":
    main()
