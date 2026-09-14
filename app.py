"""Command-line interface for asking questions about the indexed PDFs."""

import logging

from src.bootstrap import create_rag_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def display_result(result: dict) -> None:
    """Print an answer and the PDF pages used to retrieve it."""
    print(f"\n{result['answer']}")
    for source, score in zip(result["sources"], result["similarity_scores"]):
        name = source.get("scheme_name", source.get("filename", "Unknown"))
        print(f"- {name}, page {source.get('page', '?')} (relevance: {score:.4f})")


def main() -> None:
    try:
        rag = create_rag_pipeline()
    except Exception as error:
        print(f"Could not start the application: {error}")
        return

    print("Government scheme PDF assistant. Type 'exit' to quit.")
    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if question.lower() in {"exit", "quit", "q"}:
            return
        if not question:
            continue
        try:
            display_result(rag.query(question))
        except Exception as error:
            print(f"Could not answer the question: {error}")


if __name__ == "__main__":
    main()
