"""Render saved retrieval/generation measurements without calling any models."""

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from gov_scheme_rag.evaluation import load_cases
from gov_scheme_rag.identity import canonical_filename


def read_rows(path: Path) -> list[dict]:
    latest = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        latest[(row["id"], row["mode"])] = row
    return list(latest.values())


def cluster_interval(rows: list[dict], cases: dict, split: str) -> tuple | None:
    paired = defaultdict(dict)
    for row in rows:
        score = row.get("retrieval_metrics", {}).get("recall@3")
        if (score is not None and row["mode"] in {"baseline", "hybrid"}
                and (split == "all" or row["split"] == split)):
            paired[row["id"]][row["mode"]] = score
    groups = defaultdict(list)
    for identifier, scores in paired.items():
        if len(scores) == 2:
            group = canonical_filename(cases[identifier]["supporting_evidence"][0]["filename"])
            groups[group].append(scores["hybrid"] - scores["baseline"])
    clusters = list(groups.values())
    if not clusters:
        return None
    rng = random.Random(42)
    differences = sorted(statistics.mean(value for cluster in rng.choices(clusters, k=len(clusters))
                                        for value in cluster) for _ in range(2000))
    return differences[50], differences[1949]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-run", type=Path, default=Path("evaluation/results/retrieval"))
    parser.add_argument("--generation-run", type=Path)
    parser.add_argument("--dataset", type=Path, default=Path("evaluation/schemes_v1.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/report.md"))
    args = parser.parse_args()
    lines = ["# Evaluation results", ""]
    for label, directory in [("Retrieval", args.retrieval_run), ("Generation", args.generation_run)]:
        if directory is None:
            continue
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        lines.extend([f"## {label}", "", f"Completed: {summary['completed_at']}", ""])
        for mode, result in summary["modes"].items():
            lines.extend([f"### {mode}", "",
                          f"Successful: {result['successful']}/{summary['requested_per_mode']}; "
                          f"attempted: {result['attempted']}; errors: {result['errors']}.", "",
                          "| Metric | Mean | Scored cases |", "|---|---:|---:|"])
            for metric, score in result[label.lower()].items():
                value = f"{score['mean']:.4f}" if score["mean"] is not None else "undefined"
                lines.append(f"| {metric} | {value} | {score['n']} |")
            if label == "Generation":
                lines.extend(["", f"Abstentions: {result.get('abstentions', 0)}; "
                              f"invalid citation IDs: {result.get('invalid_citations', 0)}."])
            lines.append("")
    rows = read_rows(args.retrieval_run / "cases.jsonl")
    cases = {case["id"]: case for case in load_cases(args.dataset)}
    for split in ("all", "test"):
        interval = cluster_interval(rows, cases, split)
        if interval:
            low, high = interval
            lines.append(f"{split}: hybrid minus baseline Recall@3, document-cluster bootstrap "
                         f"95% interval [{100 * low:.2f}, {100 * high:.2f}] percentage points "
                         "(2,000 samples, seed 42).")
    lines.extend(["", "Read the benchmark methodology and metric denominators before interpreting scores.", ""])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
