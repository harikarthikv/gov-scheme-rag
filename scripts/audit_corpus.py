"""Verify every indexed chunk against every readable local PDF page, without re-embedding."""

import hashlib
import json
from pathlib import Path

import chromadb

from gov_scheme_rag.config import DATA_DIR, VECTOR_DB_PATH, COLLECTION_NAME
from gov_scheme_rag.data_loader import SchemeDataLoader
from gov_scheme_rag.identity import normalized_text, canonical_filename
from gov_scheme_rag.index_io import iter_documents


def main():
    loader = SchemeDataLoader()
    documents = loader.load()
    pages = {(doc.metadata["filename"], doc.metadata["page"]): normalized_text(doc.page_content)
             for doc in documents}
    collection = chromadb.PersistentClient(path=str(VECTOR_DB_PATH)).get_collection(COLLECTION_NAME)
    indexed_files, indexed_pages, unmatched, errors = set(), set(), [], []
    total = 0
    for doc in iter_documents(collection):
        total += 1
        key = (doc.metadata.get("filename"), doc.metadata.get("page"))
        indexed_files.add(key[0])
        indexed_pages.add(key)
        if key not in pages:
            errors.append({"filename": key[0], "page": key[1], "reason": "missing_pdf_page"})
        elif normalized_text(doc.page_content) not in pages[key]:
            unmatched.append({"filename": key[0], "page": key[1]})
    source_files = {key[0] for key in pages}
    pdfs = sorted(DATA_DIR.rglob("*.pdf"))
    file_hashes = [f"{p.relative_to(DATA_DIR).as_posix()}:{hashlib.sha256(p.read_bytes()).hexdigest()}" for p in pdfs]
    report = {"status": "ok" if not (errors or unmatched or source_files - indexed_files) else "mismatch",
              "loader": loader.stats, "source_pdf_files": len(pdfs),
              "canonical_pdf_names": len({canonical_filename(p.name) for p in pdfs}),
              "readable_pages": len(pages), "indexed_chunks": total,
              "indexed_files": len(indexed_files), "indexed_pages": len(indexed_pages),
              "unindexed_readable_files": sorted(source_files - indexed_files),
              "unindexed_readable_pages": len(set(pages) - indexed_pages),
              "missing_source_errors": errors, "chunk_text_mismatches": unmatched,
              "source_corpus_sha256": hashlib.sha256("\n".join(file_hashes).encode()).hexdigest(),
              "collection_configuration": collection.configuration}
    destination = Path("evaluation/results/corpus_audit.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "ok" or loader.stats["failed_files"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
