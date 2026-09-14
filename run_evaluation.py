"""Run the small, API-backed evaluation set for the RAG pipeline."""

import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.bootstrap import create_rag_pipeline
from src.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, MODEL_NAME
from test.gold_standard_dataset import gold_standard

JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Grade the generated answer against the expected answer. Return only one integer from 1 to 5, where 5 is fully correct.",
        ),
        (
            "human",
            "Question: {question}\nExpected answer: {expected}\nGenerated answer: {generated}\nGrade:",
        ),
    ]
)


def parse_grade(response: str) -> int:
    """Read a 1-to-5 grade from the judge response."""
    match = re.search(r"\b([1-5])\b", response)
    if not match:
        raise ValueError(f"Judge returned an invalid grade: {response!r}")
    return int(match.group(1))


def create_judge():
    """Create the separate model call used to grade generated answers."""
    return JUDGE_PROMPT | ChatOpenAI(
        model=MODEL_NAME,
        temperature=0,
        max_tokens=10,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    ) | StrOutputParser()


def run_evaluation() -> tuple[float, float]:
    """Print retrieval recall and the mean model-judge grade for ten cases."""
    if not DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY is not set. Add it to the .env file.")

    rag = create_rag_pipeline()
    judge = create_judge()
    retrieval_hits = 0
    grades = []

    for index, case in enumerate(gold_standard, start=1):
        retrieved = rag.retriever.retrieve(case["question"])
        hit = any(
            item["metadata"].get("filename") == case["source_document"]
            and item["metadata"].get("page") == case["page_number"]
            for item in retrieved
        )
        retrieval_hits += hit

        answer = rag.query(case["question"])["answer"]
        grade = parse_grade(
            judge.invoke(
                {
                    "question": case["question"],
                    "expected": case["expected_answer"],
                    "generated": answer,
                }
            )
        )
        grades.append(grade)
        print(f"{index}. retrieval {'hit' if hit else 'miss'}; judge grade {grade}/5")

    recall_percent = retrieval_hits / len(gold_standard) * 100
    average_grade = sum(grades) / len(grades)
    print(f"Retrieval recall: {retrieval_hits}/{len(gold_standard)} ({recall_percent:.1f}%)")
    print(f"Average judge grade: {average_grade:.2f}/5")
    return recall_percent, average_grade


if __name__ == "__main__":
    run_evaluation()
