# Recorded results

Measured on 2026-11-09 before repository cleanup, using the original 34,844-chunk
Chroma collection and unchanged MiniLM embeddings/chunking. Dataset: 110 cases
from 59 PDFs, with 25 development and 85 document-grouped test cases.

Hybrid uses 30 candidates per branch, 50:50 RRF with constant 60 and no re-ranker.
The baseline preserves duplicate chunks and its 0.3 dense-relevance cutoff.

| Split | Method | Recall@3 | Precision@3 | MRR@3 | Recall@5 |
|---|---|---:|---:|---:|---:|
| All 110 | Baseline | 0.8727 | 0.2909 | 0.7848 | 0.9273 |
| All 110 | Dense, deduplicated | 0.8818 | 0.2939 | 0.7909 | 0.9091 |
| All 110 | BM25 | 0.8727 | 0.2909 | 0.8167 | 0.9182 |
| All 110 | Hybrid | 0.9091 | 0.3030 | 0.8606 | 0.9273 |
| Test 85 | Baseline | 0.8824 | 0.2941 | 0.8059 | 0.9412 |
| Test 85 | Hybrid | 0.9176 | 0.3059 | 0.8725 | 0.9412 |

Recall@3 improved by four cases overall and three on test; Recall@5 was unchanged.
A document-cluster bootstrap (2,000 samples, seed 42) gives a 95% interval of
[-5.45, 12.73] percentage points overall and [-7.06, 13.95] on test. The small
benchmark does not establish statistical significance or universal answer accuracy.

## Generation

All 110 hybrid answer-and-judge cases completed without errors, using five contexts
per answer. DeepSeek resolved the configured alias to `deepseek-flash`; thinking
was disabled. The answerer used temperature 0.1 / 1,024 tokens, the judge 0 / 4,096.

| Custom judge metric | Mean | Scored cases |
|---|---:|---:|
| Faithfulness | 0.9882 | 99 |
| Citation support | 0.9882 | 99 |
| Answer relevancy | 0.8664 | 110 |
| Answer correctness | 0.8336 | 110 |
| Context precision | 0.8554 | 110 |
| Context recall | 0.8348 | 110 |

There were 11 abstentions and no responses rejected for invalid citation IDs.
Abstentions have undefined faithfulness, so its smaller denominator matters.
These are model-judge scores, not RAGAS or human ratings. No generation improvement
relative to the old prompt was measured. Some page hits missed the required answer
span, and some useful contexts still produced abstentions.

## Re-ranking and verification

On 25 development cases, `cross-encoder/ms-marco-MiniLM-L6-v2` left Recall@3 at
0.8800 and reduced MRR@3 from 0.8200 to 0.7067. Median CPU retrieval with re-ranking
was 3.24 seconds. It remains disabled; no test/generation benefit is claimed.

The original corpus audit matched all 2,876 PDFs, 4,050 pages and 34,844 chunks to
source text. A separate two-PDF, 25-chunk ingestion/reload/generation test and live
Streamlit check passed. New indexes use cosine; the full benchmark used legacy L2.

The reviewed benchmark remains committed. Historical manifests and raw judgments
are retained locally under ignored `evaluation/results/`; they contain machine
paths and are not included in GitHub. To reproduce measurements and generate a
fresh report, use the commands in [the evaluation guide](README.md).
