# Evaluation

[schemes_v1.jsonl](schemes_v1.jsonl) contains 110 cases from 59 PDFs: 10 original
examples, 69 source-authored FAQs and 31 reviewed paraphrases. The document-grouped
split has 25 development and 85 test cases. Each case includes a reference answer,
verbatim evidence, filename, 1-based page and PDF hash. The original ten examples
are preserved in [seeds.json](seeds.json).

The fixture repairs the original `aag_copy.pdf` filename reference and records the
AABY premium contradiction. Malformed and unrelated FAQs were rejected. Evidence
remains verbatim; display text repairs common encoding errors. It is a benchmark
of source fidelity, not an independent verification of current government policy.

## Commands

Run from the repository root after `uv sync --frozen`:

```powershell
uv run python -m scripts.evaluate --output evaluation/results/retrieval
uv run python -m scripts.evaluate --generate --modes hybrid --ks 1 3 5 --workers 4 --output evaluation/results/generation
uv run python -m scripts.report_results --retrieval-run evaluation/results/retrieval --generation-run evaluation/results/generation
```

`--generate` makes paid answer and judge calls. Three consecutive failures stop the
run with a nonzero exit; failures are reported, not assigned neutral scores. The
same command/output directory resumes successful case/mode pairs and retries
failures. Changed data or settings require a fresh directory. Raw case outputs,
manifests and generated reports are ignored by Git and retained locally.

To reproduce the reviewed fixture without overwriting it:

```powershell
uv run python -m scripts.build_eval_dataset --output evaluation/results/rebuilt_dataset.jsonl
```

The FAQ extractor is collection-specific. Review generated candidates before
using them as a benchmark. The runner, metrics and judge can use other collections.

## Another collection

Supply UTF-8 JSONL using this schema and configure the PDF directory/index:

```json
{"id":"example-001","question":"What is the benefit?","reference_answer":"The benefit is ...","split":"test","supporting_evidence":[{"filename":"document.pdf","page":1,"quote":"Verbatim PDF text ...","pdf_sha256":"optional source hash"}]}
```

Use `--dataset path/to/cases.jsonl`. Filenames must be unique within the collection;
adapt `page_key` and copy-alias canonicalization if the collection needs another
identity convention. Source validation checks PDF hashes and verbatim evidence.
`--skip-source-validation` is available when PDFs cannot be accessed, and that
limitation must be reported. Keep evidence independent of retrieval output and
keep questions from the same document in the same split.

## Metrics

- Recall@K: unique labeled pages retrieved / labeled pages.
- Precision@K: first occurrences of relevant pages / K; missing ranks and duplicate
  pages receive no credit. With one labeled page, Precision@3 is capped at 1/3.
- MRR@K: reciprocal of the first relevant rank, or zero when absent.
- Evidence-token recall: lexical coverage of evidence tokens from relevant pages;
  a diagnostic, not semantic context recall.
- Faithfulness and citation support: supported atomic answer claims / answer
  claims, judged against all contexts or the cited contexts respectively.
- Answer relevancy and correctness: separate judge rubric scores in [0,1].
- Context precision: average precision over ranked context-usefulness judgments.
  Context recall: reference claims supported by retrieved context / reference claims.

Metrics use the exact contexts passed to generation and are macro-averaged over
valid cases, with denominators. Abstentions have undefined faithfulness/citation
support (`null`), not perfect scores. The structured DeepSeek judge is implemented
locally; these are not RAGAS or human ratings. The judge shares a model family with
the answerer unless `RAG_JUDGE_MODEL` is changed.

## Interpretation

The page relevance labels are incomplete. A page hit can miss the actual answer
span, while an unlabeled page can still be useful. FAQ vocabulary and scheme names
make many cases easier than real queries. There are no multilingual, unanswerable,
adversarial or exhaustive multi-document eligibility cases. Do not tune repeatedly
on the test split or describe these results as universal accuracy.

[RESULTS.md](RESULTS.md) preserves the previous measurements. Fresh reports read
saved results rather than embedding run-specific scores or test claims.
