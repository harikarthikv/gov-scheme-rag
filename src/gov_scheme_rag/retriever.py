"""Dense/BM25 retrieval, reciprocal-rank fusion and optional cross-encoding."""

import logging

from gov_scheme_rag.config import RetrievalConfig
from gov_scheme_rag.identity import chunk_id, content_id
from gov_scheme_rag.lexical import BM25Index

logger = logging.getLogger(__name__)


def record(document, score: float = 0.0, score_type: str = "dense_relevance") -> dict:
    metadata = dict(document.metadata)
    metadata.setdefault("chunk_id", chunk_id(document.page_content, metadata))
    return {"chunk": document.page_content, "metadata": metadata,
            "score": float(score), "score_type": score_type}


def unique(items: list[dict]) -> list[dict]:
    seen, result = set(), []
    for item in items:
        identity = content_id(item["chunk"])
        if identity not in seen:
            result.append(item)
            seen.add(identity)
    return result


class Retriever:
    """Baseline intentionally preserves the old top-k / threshold behavior."""

    def __init__(self, database, config: RetrievalConfig | None = None,
                 lexical: BM25Index | None = None) -> None:
        self._database = database
        self.config = config or RetrievalConfig()
        self._lexical = lexical
        self._reranker = None

    @property
    def lexical(self) -> BM25Index:
        if self._lexical is None:
            from gov_scheme_rag.index_io import iter_documents
            records = unique(sorted([record(doc) for doc in iter_documents(self._database)],
                                    key=lambda item: item["metadata"]["chunk_id"]))
            self._lexical = BM25Index(records)
            logger.info("BM25 ready: %s unique chunks", len(records))
        return self._lexical

    def retrieve(self, query: str) -> list[dict]:
        if not query.strip():
            return []
        cfg = self.config
        if cfg.mode == "baseline":
            matches = self._database.similarity_search_with_relevance_scores(query, k=cfg.top_k)
            return [record(doc, score) for doc, score in matches if score >= cfg.similarity_threshold]
        dense = []
        if cfg.mode != "bm25":
            matches = self._database.similarity_search_with_relevance_scores(query, k=cfg.candidate_k)
            dense = unique([record(doc, score) for doc, score in matches
                            if cfg.mode == "hybrid" or score >= cfg.similarity_threshold])
        sparse = []
        if cfg.mode in {"bm25", "hybrid"}:
            sparse = [dict(item, score=score, score_type="bm25")
                      for item, score in self.lexical.search(query, cfg.candidate_k)]
        if cfg.mode == "hybrid":
            fused = {}
            for weight, ranking in [(cfg.dense_weight, dense), (1 - cfg.dense_weight, sparse)]:
                if not weight:
                    continue
                for rank, item in enumerate(ranking, 1):
                    identity = content_id(item["chunk"])
                    fused.setdefault(identity, dict(item, score=0.0, score_type="rrf"))
                    fused[identity]["score"] += weight / (cfg.rrf_constant + rank)
            candidates = sorted(fused.values(), key=lambda item: (-item["score"], item["metadata"]["chunk_id"]))
        else:
            candidates = dense if cfg.mode == "dense" else sparse
        if cfg.reranker_model and candidates:
            if self._reranker is None:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder(cfg.reranker_model, device="cpu")
            candidates = candidates[:cfg.candidate_k]
            scores = self._reranker.predict([(query, item["chunk"]) for item in candidates])
            candidates = sorted([dict(item, score=float(score), score_type="cross_encoder")
                                 for item, score in zip(candidates, scores)],
                                key=lambda item: -item["score"])
        return candidates[:cfg.top_k]
