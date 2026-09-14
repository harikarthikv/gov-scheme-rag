# Limitations and operational notes

- The code only reads PDF text layers with PyMuPDF. Scanned or malformed PDFs can be skipped, and tables or multi-column layouts may extract poorly.
- The current vector collection is reused when it already exists. It does not detect whether the source PDFs or embedding settings have changed.
- A DeepSeek-compatible API key and network access are required to initialize the answer-generation pipeline. Retrieval alone is not exposed as a separate user interface.
- The language model is instructed to use retrieved content and cite sources, but the code does not verify claims or citations. Do not use answers as eligibility, legal, financial, or application advice without checking the original material.
- `run_evaluation.py` is an API-backed script, not a repeatable offline test: its grades depend on model responses and it assigns a neutral score when the judge call fails.
- `test/gold_standard_dataset.py` is data consumed by the evaluation script, not an automated test suite.
