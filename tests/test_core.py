"""Behavioral regression tests; offline and independent of the real index/API."""

import json
import unittest
from unittest.mock import Mock

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from gov_scheme_rag.config import RetrievalConfig
from gov_scheme_rag.evaluation import parse_judgment, retrieval_metrics
from gov_scheme_rag.identity import canonical_filename, chunk_id
from gov_scheme_rag.lexical import BM25Index
from gov_scheme_rag.rag import FALLBACK_ANSWER, RAGPipeline
from gov_scheme_rag.retriever import Retriever, record
from gov_scheme_rag.index_io import iter_documents


def doc(text, filename="scheme.pdf", page=1):
    return Document(page_content=text, metadata={"filename": filename, "page": page})


class RetrievalTests(unittest.TestCase):
    def test_chroma_reads_are_paginated(self):
        database = Mock()
        database.get.side_effect = [
            {"documents": ["first", "second"], "metadatas": [{}, {}]},
            {"documents": ["third"], "metadatas": [{}]}]
        self.assertEqual([d.page_content for d in iter_documents(database, 2)], ["first", "second", "third"])
        self.assertEqual(database.get.call_args.kwargs["offset"], 2)

    def test_copy_alias_and_stable_identity(self):
        self.assertEqual(canonical_filename("aag_copy.pdf"), "aag.pdf")
        self.assertEqual(chunk_id("hello world", {"filename": "aag.pdf", "page": 1}),
                         chunk_id("hello\nworld", {"filename": "aag copy.pdf", "page": 1}))

    def test_bm25_exact_term_and_empty_query(self):
        index = BM25Index([record(doc("AABCS subsidy for entrepreneurs")),
                           record(doc("agriculture pension"))])
        self.assertIn("AABCS", index.search("AABCS", 3)[0][0]["chunk"])
        self.assertEqual(index.search("absent_token", 3), [])
        self.assertEqual(BM25Index([]).search("x", 3), [])

    def test_baseline_threshold_and_dense_deduplication(self):
        database = Mock()
        database.similarity_search_with_relevance_scores.return_value = [
            (doc("same text"), 0.8), (doc("same text", "scheme copy.pdf"), 0.7),
            (doc("second"), 0.6), (doc("weak"), 0.2)]
        baseline = Retriever(database, RetrievalConfig(mode="baseline", top_k=3))
        self.assertEqual(len(baseline.retrieve("query")), 3)
        dense = Retriever(database, RetrievalConfig(mode="dense", top_k=3))
        self.assertEqual([i["chunk"] for i in dense.retrieve("query")], ["same text", "second"])
        self.assertEqual(dense.retrieve("  "), [])

    def test_hybrid_keeps_sparse_match_below_dense_threshold(self):
        database = Mock()
        database.similarity_search_with_relevance_scores.return_value = [(doc("unrelated"), 0.01)]
        lexical = BM25Index([record(doc("AABCS capital subsidy"))])
        retriever = Retriever(database, RetrievalConfig(top_k=2), lexical)
        self.assertIn("AABCS capital subsidy", [i["chunk"] for i in retriever.retrieve("AABCS")])
        self.assertTrue(all(i["score_type"] == "rrf" for i in retriever.retrieve("AABCS")))

    def test_rrf_weights_can_select_only_one_branch(self):
        database = Mock()
        database.similarity_search_with_relevance_scores.return_value = [(doc("dense"), 0.8)]
        lexical = BM25Index([record(doc("sparse"))])
        cfg = RetrievalConfig(top_k=1, dense_weight=0)
        self.assertEqual(Retriever(database, cfg, lexical).retrieve("sparse")[0]["chunk"], "sparse")

    def test_page_metrics_do_not_reward_duplicates(self):
        case = {"supporting_evidence": [{"filename": "a.pdf", "page": 1, "quote": "fact"},
                                         {"filename": "b.pdf", "page": 2, "quote": "other"}]}
        retrieved = [record(doc("wrong", "x.pdf")), record(doc("fact", "a copy.pdf")),
                     record(doc("fact", "a.pdf"))]
        metrics = retrieval_metrics(case, retrieved, 3)
        self.assertEqual(metrics["recall@3"], 0.5)
        self.assertEqual(metrics["precision@3"], 1 / 3)
        self.assertEqual(metrics["mrr@3"], 0.5)


class GenerationTests(unittest.TestCase):
    def test_no_context_never_calls_provider(self):
        model = Mock()
        rag = RAGPipeline(Mock(), llm=model)
        self.assertEqual(rag.answer_from_retrieved("question", [])["answer"], FALLBACK_ANSWER)
        model.invoke.assert_not_called()

    def test_citations_are_inline_and_out_of_range_is_rejected(self):
        model = Mock()
        model.invoke.return_value = AIMessage(content="The benefit is 50 rupees [2].")
        rag = RAGPipeline(Mock(), llm=model)
        contexts = [record(doc("The benefit is 50 rupees."))]
        result = rag.answer_from_retrieved("amount?", contexts)
        self.assertEqual(result["answer"], FALLBACK_ANSWER)
        self.assertFalse(result["citation_validation"]["valid"])
        messages = model.invoke.call_args.args[0]
        self.assertIn("[1] scheme.pdf, page 1", messages[1].content)
        model.invoke.return_value = AIMessage(content="The benefit is 50 rupees [1].")
        result = rag.answer_from_retrieved("amount?", contexts)
        self.assertEqual(result["cited_sources"][0]["citation_id"], 1)

    def test_missing_citation_is_rejected(self):
        model = Mock()
        model.invoke.return_value = AIMessage(content="Uncited answer.")
        result = RAGPipeline(Mock(), llm=model).answer_from_retrieved("?", [record(doc("fact"))])
        self.assertFalse(result["citation_validation"]["valid"])

    def test_judge_strictness_and_abstention_is_not_perfect_faithfulness(self):
        judgment = {"claims": [], "answer_relevancy": 0.2, "answer_correctness": 0,
                    "context_relevance": [False], "reference_claims": [{"claim": "fact", "covered": False}]}
        _, metrics = parse_judgment(json.dumps(judgment), 1)
        self.assertIsNone(metrics["faithfulness"])
        self.assertEqual(metrics["context_precision"], 0)
        with self.assertRaises(ValueError):
            parse_judgment(json.dumps(judgment), 2)
        judgment["answer_relevancy"] = True
        with self.assertRaises(ValueError):
            parse_judgment(json.dumps(judgment), 1)


if __name__ == "__main__":
    unittest.main()
