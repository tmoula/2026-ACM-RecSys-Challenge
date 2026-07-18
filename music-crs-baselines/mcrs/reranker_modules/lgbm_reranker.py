"""LightGBM reranker with cross-encoder and retrieval features."""

import os
from typing import List, Optional

import lightgbm as lgb

from .cross_encoder_reranker import CrossEncoderReranker
from .rerank_features import FEATURE_COLUMNS, build_candidate_features, load_popularity_scores


class LGBMReranker:
    """Rerank RRF candidates using trained LightGBM + optional CE features."""

    needs_item_db = True
    needs_retrieval = True

    def __init__(
        self,
        model_path: str = "./cache/reranker/lgbm_model.txt",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cuda",
        batch_size: int = 32,
        ce_topk: int = 200,
        cache_dir: str = "./cache",
        use_cross_encoder: bool = True,
    ) -> None:
        self.model_path = model_path
        self.cache_dir = cache_dir
        self.ce_topk = ce_topk
        self.use_cross_encoder = use_cross_encoder
        self.popularity_scores = load_popularity_scores(cache_dir)
        self.model = lgb.Booster(model_file=model_path) if os.path.exists(model_path) else None
        self.cross_encoder = (
            CrossEncoderReranker(
                model_name=cross_encoder_model,
                device=device,
                batch_size=batch_size,
                ce_topk=ce_topk,
            )
            if use_cross_encoder
            else None
        )

    def _predict_scores(self, feature_rows: List[dict]) -> List[float]:
        if self.model is None:
            return [row["ce_score"] * 0.5 + row["rrf_score"] * 0.5 for row in feature_rows]
        matrix = [[row[name] for name in FEATURE_COLUMNS] for row in feature_rows]
        return self.model.predict(matrix).tolist()

    def rerank(
        self,
        candidates: List[str],
        user_id: Optional[str] = None,
        topk: int = 20,
        query: Optional[str] = None,
        dialogue_query: Optional[str] = None,
        item_db=None,
        retrieval=None,
        retrieval_meta: Optional[dict] = None,
    ) -> List[str]:
        if not candidates:
            return []

        pool = candidates[: max(self.ce_topk, topk)]
        full_query = query or dialogue_query or ""
        dialogue_query = dialogue_query or full_query

        ce_scores: dict[str, float] = {}
        if self.cross_encoder is not None and item_db is not None and full_query:
            ce_scores = self.cross_encoder._score_pairs(full_query, pool[: self.ce_topk], item_db)

        if retrieval is None:
            feature_rows = [
                {
                    "track_id": track_id,
                    "rrf_rank": rank,
                    "rrf_score": 1.0 / (rank + 1),
                    "dense_rank": 9999,
                    "sparse_rank": 9999,
                    "cf_rank": 9999,
                    "dense_score": 0.0,
                    "cf_score": 0.0,
                    "popularity": self.popularity_scores.get(track_id, 0.0),
                    "ce_score": ce_scores.get(track_id, 0.0),
                }
                for rank, track_id in enumerate(pool)
            ]
        else:
            feature_rows = build_candidate_features(
                candidates=pool,
                full_query=full_query,
                dialogue_query=dialogue_query,
                user_id=user_id,
                retrieval_meta=retrieval_meta,
                dense_retriever=retrieval.dense_retriever,
                cf_retriever=retrieval.cf_retriever,
                popularity_scores=self.popularity_scores,
            )
            for row in feature_rows:
                row["ce_score"] = ce_scores.get(row["track_id"], 0.0)

        scored = list(zip(pool, self._predict_scores(feature_rows)))
        scored.sort(key=lambda item: item[1], reverse=True)

        reranked = [track_id for track_id, _ in scored]
        seen = set(reranked)
        for track_id in candidates:
            if track_id not in seen:
                reranked.append(track_id)
                seen.add(track_id)
        return reranked[:topk]

    def batch_rerank(
        self,
        batch_candidates: List[List[str]],
        user_ids: List[Optional[str]],
        topk: int = 20,
        queries: Optional[List[str]] = None,
        dialogue_queries: Optional[List[str]] = None,
        item_db=None,
        retrieval=None,
    ) -> List[List[str]]:
        retrieval_meta_list = getattr(retrieval, "last_batch_meta", [None] * len(batch_candidates))
        return [
            self.rerank(
                candidates,
                user_id=user_id,
                topk=topk,
                query=query,
                dialogue_query=dialogue_query,
                item_db=item_db,
                retrieval=retrieval,
                retrieval_meta=meta,
            )
            for candidates, user_id, query, dialogue_query, meta in zip(
                batch_candidates,
                user_ids,
                queries or [None] * len(batch_candidates),
                dialogue_queries or queries or [None] * len(batch_candidates),
                retrieval_meta_list,
            )
        ]
