"""Build the local Chroma index from PDFs under data/text_data."""

def main() -> None:
    from gov_scheme_rag.data_loader import SchemeDataLoader
    from gov_scheme_rag.embedding import EmbeddingPipeline
    from gov_scheme_rag.vector_store import SchemeVectorStore

    documents = SchemeDataLoader().load()
    store = SchemeVectorStore(EmbeddingPipeline())
    if store.db_exists():
        raise FileExistsError(
            "The index already exists. Use a new collection name to rebuild it."
        )
    database = store.build_from_documents(documents)
    print(f"Indexed {database._collection.count()} chunks.")


if __name__ == "__main__":
    main()
