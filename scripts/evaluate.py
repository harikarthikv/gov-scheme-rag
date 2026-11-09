"""Reusable, resumable retrieval and generation evaluation; see --help."""

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from gov_scheme_rag.config import (COLLECTION_NAME, DATA_DIR, EMBEDDING_MODEL, MODEL_NAME,
                        JUDGE_MODEL, RetrievalConfig, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL,
                        DEEPSEEK_THINKING, TEMPERATURE, MAX_TOKENS)
from gov_scheme_rag.prompt import _SYSTEM_TEMPLATE
from gov_scheme_rag.evaluation import (JUDGE_SYSTEM, load_cases, parse_judgment,
                            retrieval_metrics, validate_sources)


def aggregate(rows: list[dict], key: str) -> dict:
    names = sorted({name for row in rows for name in row.get(key, {})})
    return {name: {"mean": statistics.mean(values) if values else None, "n": len(values)}
            for name in names
            for values in [[row[key][name] for row in rows
                            if row.get(key, {}).get(name) is not None]]}


def fingerprint_database(database) -> dict:
    from gov_scheme_rag.identity import chunk_id
    from gov_scheme_rag.index_io import iter_documents
    hashes = sorted(chunk_id(doc.page_content, doc.metadata) for doc in iter_documents(database))
    return {"collection": COLLECTION_NAME, "chunks": len(hashes),
            "content_sha256": hashlib.sha256("\n".join(hashes).encode()).hexdigest(),
            "metadata": database._collection.metadata}


def judge_answer(llm, case: dict, result: dict) -> dict:
    payload = {"question": case["question"], "reference_answer": case["reference_answer"],
               "reference_evidence": case["supporting_evidence"], "answer": result["answer"],
               "contexts": [{"citation_id": index, "text": item["chunk"]}
                            for index, item in enumerate(result["contexts"], 1)]}
    response = llm.invoke([("system", JUDGE_SYSTEM), ("human", json.dumps(payload, ensure_ascii=False))])
    raw = response.content
    try:
        judgment, metrics = parse_judgment(raw, len(result["contexts"]))
    except (ValueError, KeyError, TypeError):
        return {"judge_raw": raw, "judge_error": "InvalidJudgment"}
    return {"judgment": judgment, "generation_metrics": metrics,
            "judge_usage": response.usage_metadata}


def evaluate_one(case, mode, current, args, llm, judge, retrieval_lock) -> dict:
    row = {"id": case["id"], "mode": mode, "split": case["split"],
           "question": case["question"], "reference_answer": case["reference_answer"], "status": "ok"}
    try:
        # Bound local model work to one thread; API calls may run concurrently.
        with retrieval_lock:
            start = time.perf_counter()
            retrieved = current.retrieve(case["question"])
            row["retrieval_seconds"] = time.perf_counter() - start
        row["retrieved"] = retrieved
        row["retrieval_metrics"] = {}
        for k in args.ks:
            row["retrieval_metrics"].update(retrieval_metrics(case, retrieved, k))
        if args.generate:
            start = time.perf_counter()
            row["generation"] = llm.answer_from_retrieved(case["question"], retrieved[:args.generation_k])
            row["generation_seconds"] = time.perf_counter() - start
            start = time.perf_counter()
            row.update(judge_answer(judge, case, row["generation"]))
            row["judge_seconds"] = time.perf_counter() - start
            if "generation_metrics" not in row:
                raise ValueError("Invalid judgment")
    except Exception as error:
        row.update(status="error", error_type=type(error).__name__)
    return row


def run(args) -> dict:
    all_cases = load_cases(args.dataset)
    validation = validate_sources(all_cases, DATA_DIR) if not args.skip_source_validation else {"skipped": True}
    cases = [case for case in all_cases if args.split == "all" or case["split"] == args.split]
    if args.limit:
        cases = cases[:args.limit]
    if not cases:
        raise ValueError("No evaluation cases selected")
    from gov_scheme_rag.bootstrap import create_retriever
    from gov_scheme_rag.retriever import Retriever
    base_config = RetrievalConfig(candidate_k=max(args.candidates, max(args.ks)),
                                  dense_weight=args.dense_weight, reranker_model=args.reranker)
    retriever = create_retriever(base_config)
    db = retriever._database
    corpus = fingerprint_database(db)
    started = time.perf_counter()
    lexical = retriever.lexical if any(mode in {"bm25", "hybrid"} for mode in args.modes) else None
    lexical_seconds = time.perf_counter() - started
    metadata = {"dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                "corpus": corpus, "embedding_model": EMBEDDING_MODEL,
                "model": MODEL_NAME if args.generate else None,
                "judge_model": JUDGE_MODEL if args.generate else None,
                "retrieval_config": asdict(base_config), "modes": args.modes, "ks": args.ks,
                "split": args.split, "case_ids": [c["id"] for c in cases],
                "generate": args.generate, "generation_k": args.generation_k,
                "thinking": DEEPSEEK_THINKING if args.generate else None,
                "generation_settings": {"temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
                                        "base_url": DEEPSEEK_BASE_URL} if args.generate else None,
                "answer_prompt_sha256": hashlib.sha256(_SYSTEM_TEMPLATE.encode()).hexdigest() if args.generate else None,
                "judge_prompt_sha256": hashlib.sha256(JUDGE_SYSTEM.encode()).hexdigest(),
                "python": platform.python_version(),
                "packages": {p: version(p) for p in ["chromadb", "langchain-chroma", "sentence-transformers", "langchain-openai"]}}
    signature = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if previous["signature"] != signature:
            raise ValueError("Output has a different dataset/corpus/config. Choose a new --output directory.")
    else:
        manifest.write_text(json.dumps(dict(metadata, signature=signature,
                                          started_at=datetime.now(timezone.utc).isoformat(),
                                          source_validation=validation,
                                          workers=args.workers,
                                          bm25_build_seconds=lexical_seconds), indent=2), encoding="utf-8")
    output = args.output / "cases.jsonl"
    previous_rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()] if output.exists() else []
    done = {(row["id"], row["mode"]) for row in previous_rows if row.get("status") == "ok"}
    llm = judge = None
    if args.generate:
        if not DEEPSEEK_API_KEY:
            raise ValueError("Set DEEPSEEK_API_KEY for --generate")
        from langchain_openai import ChatOpenAI
        from gov_scheme_rag.rag import RAGPipeline
        llm = RAGPipeline(retriever)
        judge = ChatOpenAI(model=JUDGE_MODEL, temperature=0, max_tokens=4096,
                           api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL,
                           timeout=120, max_retries=1,
                           model_kwargs={"response_format": {"type": "json_object"}},
                           extra_body={"thinking": {"type": "disabled"}})
    consecutive_errors = 0
    retrieval_lock = threading.Lock()
    with output.open("a", encoding="utf-8") as stream:
        for mode in args.modes:
            cfg = replace(base_config, mode=mode, top_k=max(args.ks))
            current = Retriever(db, cfg, lexical)
            written = set()
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(evaluate_one, case, mode, current, args, llm, judge, retrieval_lock)
                           for case in cases if (case["id"], mode) not in done]
                for future in as_completed(futures):
                    row = future.result()
                    consecutive_errors = consecutive_errors + 1 if row["status"] == "error" else 0
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    print(f"{mode}: {row['id']} {row['status']}", flush=True)
                    previous_rows.append(row)
                    written.add((row["id"], row["mode"]))
                    if consecutive_errors >= 3:
                        for pending in futures:
                            pending.cancel()
                        # Preserve any already-running work before returning a partial result.
                        for pending in futures:
                            if not pending.cancelled() and pending is not future:
                                extra = pending.result()
                                if (extra["id"], extra["mode"]) not in written:
                                    stream.write(json.dumps(extra, ensure_ascii=False) + "\n")
                                    previous_rows.append(extra)
                                    written.add((extra["id"], extra["mode"]))
                        print("Stopped after three consecutive failures; fix service/config and rerun to resume.", flush=True)
                        break
            if consecutive_errors >= 3:
                break
    # Latest attempt wins; failed rows are explicit and never counted as successes.
    latest = {(row["id"], row["mode"]): row for row in previous_rows}
    summary = {"signature": signature, "completed_at": datetime.now(timezone.utc).isoformat(),
               "requested_per_mode": len(cases), "modes": {}}
    for mode in args.modes:
        rows = [row for row in latest.values() if row["mode"] == mode]
        good = [row for row in rows if row.get("status") == "ok"]
        # Retrieval remains measurable even if the external generation call failed.
        retrieved_rows = [row for row in rows if "retrieval_metrics" in row]
        summary["modes"][mode] = {"attempted": len(rows), "successful": len(good),
                                   "errors": len(rows) - len(good),
                                   "retrieval": aggregate(retrieved_rows, "retrieval_metrics"),
                                   "generation": aggregate(good, "generation_metrics")}
        times = [row["retrieval_seconds"] for row in retrieved_rows]
        if times:
            summary["modes"][mode]["retrieval_latency_seconds"] = {
                "mean": statistics.mean(times), "p50": statistics.median(times),
                "p95": sorted(times)[min(len(times) - 1, int(0.95 * len(times)))]}
        if args.generate:
            results = [row["generation"] for row in good]
            summary["modes"][mode]["abstentions"] = sum(r["answer"].startswith("I cannot find sufficient") for r in results)
            summary["modes"][mode]["invalid_citations"] = sum(not r["citation_validation"]["valid"] for r in results)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("evaluation/schemes_v1.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/retrieval"))
    parser.add_argument("--modes", nargs="+", choices=["baseline", "dense", "bm25", "hybrid"],
                        default=["baseline", "dense", "bm25", "hybrid"])
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--candidates", type=int, default=30)
    parser.add_argument("--dense-weight", type=float, default=0.5)
    parser.add_argument("--reranker", default="")
    parser.add_argument("--split", choices=["all", "dev", "test"], default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--generate", action="store_true", help="Calls DeepSeek and an LLM judge; incurs API charges")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent API tasks; local retrieval is serialized")
    parser.add_argument("--generation-k", type=int, default=5)
    parser.add_argument("--skip-source-validation", action="store_true", help="For portable datasets whose PDFs are unavailable")
    args = parser.parse_args()
    if min(args.ks) < 1 or args.generation_k < 1 or args.generation_k > max(args.ks):
        parser.error("Require positive ks and 1 <= generation-k <= max(ks)")
    if args.limit < 0 or args.workers < 1:
        parser.error("limit must be nonnegative; workers must be positive")
    summary = run(args)
    if any(result["errors"] or result["attempted"] < summary["requested_per_mode"]
           for result in summary["modes"].values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
