"""Three-way recall: dense context + BM25 dialogue + CF history, fused with RRF."""

from typing import List, Optional

from .cf_retrieve import CFRetriever
from .rrf import reciprocal_rank_fusion_with_scores


class FullStackRetriever:
    """Fuse dense, BM25, and CF retrievers with reciprocal rank fusion."""

    def __init__(
        self,
        dense_retriever,
        sparse_retriever,
        cf_retriever: CFRetriever,
        rrf_k: int = 60,
        pool_topk: int = 500,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.cf_retriever = cf_retriever
        self.rrf_k = rrf_k
        self.pool_topk = pool_topk
        self.last_batch_meta: List[dict] = []

    def _fuse(
        self,
        dense_hits: List[str],
        sparse_hits: List[str],
        cf_hits: List[str],
        topk: int,
    ) -> tuple[List[str], dict]:
        fused, rrf_scores = reciprocal_rank_fusion_with_scores(
            [dense_hits, sparse_hits, cf_hits],
            k=self.rrf_k,
        )
        dense_rank = {track_id: rank for rank, track_id in enumerate(dense_hits)}
        sparse_rank = {track_id: rank for rank, track_id in enumerate(sparse_hits)}
        cf_rank = {track_id: rank for rank, track_id in enumerate(cf_hits)}
        meta = {
            "dense_rank": dense_rank,
            "sparse_rank": sparse_rank,
            "cf_rank": cf_rank,
            "rrf_score": rrf_scores,
        }
        return fused[:topk], meta

    def retrieve(
        self,
        full_query: str,
        dialogue_query: str,
        user_id: Optional[str],
        topk: int,
    ) -> List[str]:
        dense_hits = self.dense_retriever.text_to_item_retrieval(full_query, topk=self.pool_topk)
        sparse_hits = self.sparse_retriever.text_to_item_retrieval(dialogue_query, topk=self.pool_topk)
        cf_hits = self.cf_retriever.retrieve(user_id, topk=self.pool_topk)
        candidates, meta = self._fuse(dense_hits, sparse_hits, cf_hits, topk)
        self.last_batch_meta = [meta]
        return candidates

    def batch_retrieve(
        self,
        full_queries: List[str],
        dialogue_queries: List[str],
        user_ids: List[Optional[str]],
        topk: int,
    ) -> List[List[str]]:
        dense_hits = self.dense_retriever.batch_text_to_item_retrieval(full_queries, topk=self.pool_topk)
        sparse_hits = self.sparse_retriever.batch_text_to_item_retrieval(dialogue_queries, topk=self.pool_topk)
        cf_hits = self.cf_retriever.batch_retrieve(user_ids, topk=self.pool_topk)

        results: List[List[str]] = []
        self.last_batch_meta = []
        for dense, sparse, cf, user_id in zip(dense_hits, sparse_hits, cf_hits, user_ids):
            candidates, meta = self._fuse(dense, sparse, cf, topk)
            meta["user_id"] = user_id
            self.last_batch_meta.append(meta)
            results.append(candidates)
        return results

    def text_to_item_retrieval(self, query: str, topk: int) -> List[str]:
        return self.retrieve(full_query=query, dialogue_query=query, user_id=None, topk=topk)

    def batch_text_to_item_retrieval(self, queries: List[str], topk: int) -> List[List[str]]:
        return self.batch_retrieve(
            full_queries=queries,
            dialogue_queries=queries,
            user_ids=[None] * len(queries),
            topk=topk,
        )
