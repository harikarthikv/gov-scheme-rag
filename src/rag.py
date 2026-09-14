"""Retrieve relevant chunks and ask the configured language model to answer."""

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from src.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, MAX_TOKENS, MODEL_NAME, TEMPERATURE
from src.prompt import get_prompt_template
from src.retriever import Retriever

FALLBACK_ANSWER = (
    "I cannot find sufficient information about this in the available scheme "
    "details. Please try rephrasing your question or ask about a specific scheme name."
)


class RAGPipeline:
    """Combine retrieval, prompting, and a DeepSeek-compatible chat model."""

    def __init__(self, retriever: Retriever) -> None:
        if not DEEPSEEK_API_KEY:
            raise ValueError("DEEPSEEK_API_KEY is not set. Add it to the .env file.")
        self.retriever = retriever
        self._prompt = get_prompt_template()
        self._llm = ChatOpenAI(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )
        self._output_parser = StrOutputParser()

    def query(self, question: str) -> dict:
        """Return an answer, its retrieved source metadata, and relevance scores."""
        retrieved = self.retriever.retrieve(question)
        if not retrieved:
            return {
                "answer": FALLBACK_ANSWER,
                "sources": [],
                "similarity_scores": [],
                "average_relevance_score": 0.0,
            }

        context = "\n\n---\n\n".join(item["chunk"] for item in retrieved)
        sources = "\n".join(
            "[{number}] {name} - Page {page} (score: {score:.4f})".format(
                number=index,
                name=item["metadata"].get("scheme_name", item["metadata"].get("filename", "Unknown")),
                page=item["metadata"].get("page", "?"),
                score=item["score"],
            )
            for index, item in enumerate(retrieved, start=1)
        )
        messages = self._prompt.format_messages(context=context, sources=sources, query=question)
        answer = self._output_parser.invoke(self._llm.invoke(messages))
        scores = [item["score"] for item in retrieved]
        return {
            "answer": answer,
            "sources": [item["metadata"] for item in retrieved],
            "similarity_scores": scores,
            "average_relevance_score": round(sum(scores) / len(scores), 4),
        }
