"""Retrieve relevant chunks and ask the configured language model to answer."""

import re

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from gov_scheme_rag.config import (DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, MAX_TOKENS, MODEL_NAME,
                        TEMPERATURE, DEEPSEEK_THINKING)
from gov_scheme_rag.prompt import get_prompt_template
from gov_scheme_rag.retriever import Retriever

FALLBACK_ANSWER = (
    "I cannot find sufficient information about this in the available scheme "
    "details. Please try rephrasing your question or ask about a specific scheme name."
)


class RAGPipeline:
    """Combine retrieval, prompting, and a DeepSeek-compatible chat model."""

    def __init__(self, retriever: Retriever, llm=None) -> None:
        if llm is None and not DEEPSEEK_API_KEY:
            raise ValueError("DEEPSEEK_API_KEY is not set. Add it to the .env file.")
        self.retriever = retriever
        self._prompt = get_prompt_template()
        self._llm = llm or ChatOpenAI(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            timeout=90,
            max_retries=2,
            extra_body={"thinking": {"type": DEEPSEEK_THINKING}},
        )
        self._output_parser = StrOutputParser()

    def query(self, question: str) -> dict:
        """Return an answer, its retrieved source metadata, and relevance scores."""
        retrieved = self.retriever.retrieve(question)
        return self.answer_from_retrieved(question, retrieved)

    def answer_from_retrieved(self, question: str, retrieved: list[dict]) -> dict:
        """Generate from exactly the contexts evaluated; never retrieve twice."""
        if not retrieved:
            return {
                "answer": FALLBACK_ANSWER,
                "sources": [],
                "similarity_scores": [],
                "average_relevance_score": 0.0,
                "contexts": [],
                "citation_validation": {"valid": True, "reason": "no_context"},
                "cited_sources": [],
            }

        context = "\n\n---\n\n".join(
            f"[{index}] {item['metadata'].get('filename', 'Unknown')}, "
            f"page {item['metadata'].get('page', '?')}\n{item['chunk']}"
            for index, item in enumerate(retrieved, 1)
        )
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
        response = self._llm.invoke(messages)
        answer = self._output_parser.invoke(response)
        citations = [int(value) for value in re.findall(r"\[(\d+)\]", answer)]
        abstained = answer.strip() == FALLBACK_ANSWER
        valid = abstained or (bool(citations) and all(1 <= value <= len(retrieved) for value in citations))
        raw_answer = answer
        if not valid:
            answer = FALLBACK_ANSWER
        scores = [item["score"] for item in retrieved]
        return {
            "answer": answer,
            "sources": [dict(item["metadata"], citation_id=index, excerpt=item["chunk"])
                        for index, item in enumerate(retrieved, 1)],
            "similarity_scores": scores,
            "average_relevance_score": round(sum(scores) / len(scores), 4),
            "score_type": retrieved[0].get("score_type", "dense_relevance"),
            "contexts": retrieved,
            "raw_answer": raw_answer,
            "usage": response.usage_metadata,
            "response_model": response.response_metadata.get("model_name"),
            "finish_reason": response.response_metadata.get("finish_reason"),
            "citation_validation": {"valid": valid, "reason": "ok" if valid else "missing_or_out_of_range"},
            "cited_sources": [dict(retrieved[index - 1]["metadata"], citation_id=index)
                              for index in sorted(set(citations)) if valid and not abstained],
        }
