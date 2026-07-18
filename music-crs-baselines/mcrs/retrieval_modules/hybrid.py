"""Hybrid retrieval via reciprocal rank fusion (RRF)."""

from typing import List

from .rrf import reciprocal_rank_fusion


class HybridRetriever:
    """Fuse sparse and dense retrievers with RRF."""

    def __init__(self, sparse_retriever, dense_retriever, rrf_k: int = 60, pool_topk: int = 100) -> None:
        self.sparse_retriever = sparse_retriever
        self.dense_retriever = dense_retriever
        self.rrf_k = rrf_k
        self.pool_topk = pool_topk

    @property
    def track_ids(self):
        return self.dense_retriever.track_ids

    def text_to_item_retrieval(self, query: str, topk: int) -> List[str]:
        sparse_hits = self.sparse_retriever.text_to_item_retrieval(query, topk=self.pool_topk)
        dense_hits = self.dense_retriever.text_to_item_retrieval(query, topk=self.pool_topk)
        fused = reciprocal_rank_fusion([sparse_hits, dense_hits], k=self.rrf_k)
        return fused[:topk]

    def batch_text_to_item_retrieval(self, queries: List[str], topk: int) -> List[List[str]]:
        sparse_hits = self.sparse_retriever.batch_text_to_item_retrieval(queries, topk=self.pool_topk)
        dense_hits = self.dense_retriever.batch_text_to_item_retrieval(queries, topk=self.pool_topk)
        return [
            reciprocal_rank_fusion([sparse, dense], k=self.rrf_k)[:topk]
            for sparse, dense in zip(sparse_hits, dense_hits)
        ]
