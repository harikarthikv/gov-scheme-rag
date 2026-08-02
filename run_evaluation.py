"""
run_evaluation.py
─────────────────
Automated evaluation of the MyScheme RAG pipeline against a gold-standard
dataset of 10 benchmark question-answer pairs.

Metrics computed
────────────────
  1. Retrieval Recall @ 3 — does at least one of the top-3 retrieved chunks
     come from the correct source PDF and page?
  2. LLM-as-a-Judge Correctness — DeepSeek grades the generated answer
     against the expected answer on a 1–5 scale.

Usage
─────
    python run_evaluation.py
"""

import importlib.util
import logging
import re
import sys
import time
from pathlib import Path

# ── Early progress indicator (before slow torch / sentence-transformers import) ─
print("⏳  Initialising evaluation environment …", flush=True)

# ── Configure logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Heavy imports (torch + sentence-transformers take time on first load) ─────
print("   → Importing LangChain + DeepSeek (lightweight) …", flush=True)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    MAX_TOKENS,
    MODEL_NAME,
)

print("   → Importing embedding pipeline (torch + sentence-transformers) …", flush=True)
from src.embedding import EmbeddingPipeline

print("   → Importing RAG components …", flush=True)
from src.rag import RAGPipeline
from src.retriever import Retriever
from src.vector_store import SchemeVectorStore

print("   ✅  All imports complete.", flush=True)

# ── Load gold_standard from test/ (no __init__.py, so use importlib) ─────────
_GS_PATH = Path(__file__).resolve().parent / "test" / "gold_standard_dataset.py"
_spec = importlib.util.spec_from_file_location("gold_standard_dataset", _GS_PATH)
_gs_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gs_module)
gold_standard = _gs_module.gold_standard

# ── LLM Judge prompt ─────────────────────────────────────────────────────────

_JUDGE_SYSTEM = """\
You are an expert evaluator for a Government Schemes Q&A system. \
Your task is to grade the correctness of a generated answer against the \
expected (ground-truth) answer.

Scoring criteria:
  • 5 — Excellent: Completely accurate, all key facts match the expected answer.
  • 4 — Good: Mostly accurate, minor omissions or slight wording differences only.
  • 3 — Acceptable: Partially correct but misses some key details or includes minor errors.
  • 2 — Poor: Significant inaccuracies or missing most key information.
  • 1 — Incorrect: Completely wrong, irrelevant, or hallucinated.

Output ONLY a single integer (1, 2, 3, 4, or 5). Do NOT include any \
other text, punctuation, or explanation.
"""

_JUDGE_HUMAN = """\
Question:
{question}

Expected (Ground-Truth) Answer:
{expected_answer}

Generated Answer:
{generated_answer}

Grade (1–5):"""

JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _JUDGE_SYSTEM),
    ("human", _JUDGE_HUMAN),
])


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_grade(text: str) -> int:
    """Extract an integer 1–5 from the judge's response string."""
    text = text.strip()
    # Direct parse attempt
    try:
        grade = int(text)
        if 1 <= grade <= 5:
            return grade
    except ValueError:
        pass
    # Fallback: find first lone digit 1–5
    match = re.search(r"\b([1-5])\b", text)
    if match:
        return int(match.group(1))
    logger.warning(f"Could not parse grade from judge response: '{text[:80]}…'")
    return 3  # safe default


def _init_system() -> tuple[EmbeddingPipeline, RAGPipeline]:
    """Initialise embedding pipeline, vector store, retriever, and RAG."""
    logger.info("Loading embedding model …")
    embed_pipeline = EmbeddingPipeline()

    store = SchemeVectorStore(embedding_pipeline=embed_pipeline)
    if store.db_exists():
        logger.info("Existing ChromaDB found — loading from disk …")
        db = store.load()
    else:
        logger.info("No ChromaDB found — building from source documents …")
        from src.data_loader import load_all_documents
        documents = load_all_documents()
        db = store.build_from_documents(documents)

    retriever = Retriever(db=db)
    rag = RAGPipeline(retriever=retriever)
    return embed_pipeline, rag


def _create_judge_llm() -> ChatOpenAI:
    """Return a low-temperature LLM for deterministic judging."""
    return ChatOpenAI(
        model=MODEL_NAME,
        temperature=0.0,
        max_tokens=10,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )


# ── Main evaluation entry point ──────────────────────────────────────────────

def run_evaluation() -> tuple[float, float]:
    """Run all 10 test cases and print a detailed report.

    Returns:
        (recall_pct, avg_grade) for use by the README updater.
    """
    print("\n" + "═" * 72)
    print("  🧪  MyScheme RAG — Automated Evaluation")
    print("═" * 72)

    if not DEEPSEEK_API_KEY:
        logger.error(
            "DEEPSEEK_API_KEY is not set. Please add it to your .env file "
            "and retry."
        )
        sys.exit(1)

    # ── Init components ──────────────────────────────────────────────────────
    print("\n[1/2] Initialising RAG pipeline …")
    t0 = time.time()
    _, rag = _init_system()
    judge_llm = _create_judge_llm()
    judge_chain = JUDGE_PROMPT | judge_llm | StrOutputParser()
    print(f"       Done in {time.time() - t0:.1f}s\n")

    # ── Run test cases ───────────────────────────────────────────────────────
    n = len(gold_standard)
    retrieval_matches = 0
    grades: list[int] = []

    print(f"[2/2] Running {n} test cases …\n")

    for idx, case in enumerate(gold_standard, start=1):
        question = case["question"]
        expected = case["expected_answer"]
        src_doc = case["source_document"]
        src_page = case["page_number"]

        print(f"{'─' * 72}")
        print(f"\n📋  Test Case {idx}/{n}")
        print(f"    Question: {question}")

        # ─── Metric 1: Retrieval Recall @ 3 ──────────────────────────────────
        retrieved = rag.retriever.retrieve(question)
        retrieved_metas = [r["metadata"] for r in retrieved]

        # metadata uses "filename" (e.g. "25-ciss.pdf") and "page" (1-indexed)
        hit = any(
            m.get("filename") == src_doc and m.get("page") == src_page
            for m in retrieved_metas
        )
        if hit:
            retrieval_matches += 1

        print(f"\n    📂  Top-{len(retrieved)} Retrieved Chunks:")
        for i, r in enumerate(retrieved):
            m = r["metadata"]
            print(
                f"        [{i+1}] {m.get('filename', '?')}  "
                f"p.{m.get('page', '?')}  "
                f"(score: {r['score']:.4f})"
            )
        print(f"    🎯  Expected: {src_doc}  p.{src_page}")
        print(f"    {'✅  HIT' if hit else '❌  MISS'}  |  Retrieval Recall @3")

        # ─── Metric 2: Full RAG generation + LLM Judge ───────────────────────
        result = rag.query(question)
        generated = result["answer"]

        print(f"\n    💬  Generated Answer:")
        for line in generated.splitlines():
            print(f"        {line}")

        print(f"\n    📝  Expected Answer:")
        for line in expected.splitlines():
            print(f"        {line}")

        # LLM Judge
        try:
            judge_raw = judge_chain.invoke({
                "question": question,
                "expected_answer": expected,
                "generated_answer": generated,
            })
            grade = _parse_grade(judge_raw)
        except Exception as exc:
            logger.warning(f"Judge LLM call failed for case {idx}: {exc}")
            grade = 3

        grades.append(grade)
        grade_bar = "█" * grade + "░" * (5 - grade)
        print(f"\n    ⭐  LLM Judge Grade: {grade}/5  [{grade_bar}]")

        # Small delay between cases to be polite to the API
        if idx < n:
            time.sleep(1.0)

    # ── Aggregate results ────────────────────────────────────────────────────
    recall_pct = (retrieval_matches / n) * 100
    avg_grade = sum(grades) / n

    print(f"\n{'═' * 72}")
    print(f"\n  📊  FINAL RESULTS")
    print(f"  {'─' * 52}")
    print(f"  │  Retrieval Recall @ 3:       {retrieval_matches:>3}/{n}  "
          f"({recall_pct:5.1f}%){'':>5}│")
    print(f"  │  Avg. LLM Judge Score:        {avg_grade:.2f} / 5     "
          f"{'':>15}│")
    print(f"  {'─' * 52}")
    print(f"\n  Grade Distribution:  "
          f"{'  '.join(f'{i}→{grades.count(i)}' for i in range(1, 6))}")
    print(f"\n{'═' * 72}\n")

    return recall_pct, avg_grade


if __name__ == "__main__":
    run_evaluation()
