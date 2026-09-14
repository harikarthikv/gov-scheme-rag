"""Build the local Chroma index from PDFs under data/text_data."""

def main() -> None:
    from src.data_loader import load_all_documents
    from src.embedding import EmbeddingPipeline
    from src.vector_store import SchemeVectorStore

    documents = load_all_documents()
    store = SchemeVectorStore(EmbeddingPipeline())
    if store.db_exists():
        raise FileExistsError(
            "The index already exists. Use a new collection name to rebuild it."
        )
    database = store.build_from_documents(documents)
    print(f"Indexed {database._collection.count()} chunks.")


if __name__ == "__main__":
    main()
