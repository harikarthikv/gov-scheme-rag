"""Bounded Chroma reads avoid SQLite's variable limit on full-corpus get()."""

from langchain_core.documents import Document


def iter_documents(database, batch_size: int = 1000):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    offset = 0
    while True:
        rows = database.get(limit=batch_size, offset=offset, include=["documents", "metadatas"])
        texts = rows["documents"]
        for text, metadata in zip(texts, rows["metadatas"]):
            yield Document(page_content=text, metadata=metadata or {})
        if len(texts) < batch_size:
            break
        offset += len(texts)
