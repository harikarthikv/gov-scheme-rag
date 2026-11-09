"""Check resumption and failure accounting without a model or real Chroma."""

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document

from scripts.evaluate import run
from gov_scheme_rag.evaluation import load_cases


class FakeDatabase:
    def __init__(self):
        self._collection = SimpleNamespace(metadata=None)
        self.calls = 0
        self.fail = False

    def get(self, **kwargs):
        return {"documents": ["The grant is fifty rupees."],
                "metadatas": [{"filename": "example.pdf", "page": 1}]}

    def similarity_search_with_relevance_scores(self, query, k):
        self.calls += 1
        if self.fail:
            raise RuntimeError("Simulated search failure")
        return [(Document(page_content="The grant is fifty rupees.",
                          metadata={"filename": "example.pdf", "page": 1}), 0.8)]


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        root = Path(self.work.name)
        self.dataset = root / "cases.jsonl"
        self.case = {"id": "one", "question": "What is the grant?", "reference_answer": "Fifty rupees.",
                     "split": "test", "supporting_evidence": [{"filename": "example.pdf", "page": 1,
                                                               "quote": "The grant is fifty rupees."}]}
        self.dataset.write_text(json.dumps(self.case) + "\n", encoding="utf-8")
        self.args = argparse.Namespace(dataset=self.dataset, skip_source_validation=True, split="all",
                                       limit=0, candidates=30, ks=[1], dense_weight=0.5, reranker="",
                                       modes=["baseline"], generate=False, generation_k=1, workers=1,
                                       output=root / "output")
        self.database = FakeDatabase()

    def evaluate(self):
        with patch("gov_scheme_rag.bootstrap.create_retriever", return_value=SimpleNamespace(_database=self.database)):
            with contextlib.redirect_stdout(io.StringIO()):
                return run(self.args)

    def test_resume_skips_success_and_rejects_changed_config(self):
        self.evaluate()
        summary = self.evaluate()
        self.assertEqual(self.database.calls, 1)
        self.assertEqual(summary["modes"]["baseline"]["successful"], 1)
        self.args.dense_weight = 0.75
        with self.assertRaises(ValueError):
            self.evaluate()

    def test_failed_search_is_recorded_and_retried(self):
        self.database.fail = True
        summary = self.evaluate()
        self.assertEqual(summary["modes"]["baseline"]["errors"], 1)
        self.assertEqual(summary["modes"]["baseline"]["retrieval"], {})
        self.database.fail = False
        summary = self.evaluate()
        self.assertEqual(summary["modes"]["baseline"]["successful"], 1)
        attempts = (self.args.output / "cases.jsonl").read_text().splitlines()
        self.assertEqual(len(attempts), 2)

    def test_bad_evidence_page_is_rejected_before_search(self):
        self.case["supporting_evidence"][0]["page"] = 0
        self.dataset.write_text(json.dumps(self.case) + "\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_cases(self.dataset)


if __name__ == "__main__":
    unittest.main()
