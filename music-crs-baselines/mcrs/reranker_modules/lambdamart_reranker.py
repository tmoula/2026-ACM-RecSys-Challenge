"""LambdaMART reranker (LightGBM lambdarank) for nDCG@20 optimization."""

import os
from typing import List, Optional, Tuple

import lightgbm as lgb
import numpy as np

from .rank_features import (
    LAMBDAMART_FEATURE_COLUMNS,
    build_lambdamart_features,
    extract_recent_dialogue,
)
from .rerank_features import load_popularity_scores


class LambdaMARTReRanker:
    """Rerank dense retrieval candidates with a trained LambdaMART model."""

    needs_item_db = True
    needs_retrieval = True

    def __init__(
        self,
        model_path: str = "./cache/reranker/lambdamart_model.txt",
        cache_dir: str = "./cache",
        popularity_penalty: float = 0.0,
    ) -> None:
        self.model_path = model_path
        self.cache_dir = cache_dir
        self.popularity_penalty = popularity_penalty
        self.popularity_scores = load_popularity_scores(cache_dir)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"LambdaMART model not found at {model_path}. "
                "Run build_lambdamart_features.py and train_lambdamart.py first."
            )
        self.model = lgb.Booster(model_file=model_path)
        model_features = self.model.feature_name()
        if model_features:
            self.feature_columns = [name for name in LAMBDAMART_FEATURE_COLUMNS if name in model_features]
        else:
            self.feature_columns = list(LAMBDAMART_FEATURE_COLUMNS)

    def _dense_retriever(self, retrieval):
        if hasattr(retrieval, "dense_retriever"):
            return retrieval.dense_retriever
        return retrieval

    def _predict_scores(self, feature_rows: List[dict]) -> np.ndarray:
        matrix = [[row[name] for name in self.feature_columns] for row in feature_rows]
        return self.model.predict(matrix)

    def _build_feature_rows(
        self,
        candidates: List[str],
        full_query: str,
        dialogue_query: str,
        item_db,
        retrieval,
        user_id: Optional[str] = None,
        user_db=None,
        session_memory: Optional[list] = None,
    ) -> List[dict]:
        dense = self._dense_retriever(retrieval)
        user_profile_str = None
        if user_id and user_db is not None:
            user_profile_str = user_db.id_to_profile_str(user_id)
        recent_dialogue = extract_recent_dialogue(session_memory, num_turns=2) if session_memory else dialogue_query
        pool_topk = getattr(retrieval, "pool_topk", None) if retrieval is not None else None
        return build_lambdamart_features(
            candidates=candidates,
            full_query=full_query,
            dialogue_query=dialogue_query,
            dense_retriever=dense,
            item_db=item_db,
            popularity_scores=self.popularity_scores,
            user_profile_str=user_profile_str,
            recent_dialogue=recent_dialogue,
            retrieval=retrieval,
            pool_topk=pool_topk,
        )

    def rerank_with_scores(
        self,
        candidates: List[str],
        user_id: Optional[str] = None,
        query: Optional[str] = None,
        dialogue_query: Optional[str] = None,
        item_db=None,
        retrieval=None,
        user_db=None,
        session_memory: Optional[list] = None,
    ) -> List[Tuple[str, float]]:
        if not candidates:
            return []

        full_query = query or dialogue_query or ""
        dialogue_query = dialogue_query or full_query

        feature_rows = self._build_feature_rows(
            candidates,
            full_query,
            dialogue_query,
            item_db,
            retrieval,
            user_id=user_id,
            user_db=user_db,
            session_memory=session_memory,
        )
        scores = self._predict_scores(feature_rows)
        if self.popularity_penalty:
            for index, row in enumerate(feature_rows):
                scores[index] -= self.popularity_penalty * row.get("popularity", 0.0)

        ranked_indices = np.argsort(-scores)
        reranked = [(candidates[index], float(scores[index])) for index in ranked_indices]

        seen = {track_id for track_id, _ in reranked}
        for track_id in candidates:
            if track_id not in seen:
                reranked.append((track_id, 0.0))
                seen.add(track_id)
        return reranked

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
        user_db=None,
        session_memory: Optional[list] = None,
    ) -> List[str]:
        scored = self.rerank_with_scores(
            candidates,
            user_id=user_id,
            query=query,
            dialogue_query=dialogue_query,
            item_db=item_db,
            retrieval=retrieval,
            user_db=user_db,
            session_memory=session_memory,
        )
        return [track_id for track_id, _ in scored[:topk]]

    def batch_rerank(
        self,
        batch_candidates: List[List[str]],
        user_ids: List[Optional[str]],
        topk: int = 20,
        queries: Optional[List[str]] = None,
        dialogue_queries: Optional[List[str]] = None,
        item_db=None,
        retrieval=None,
        session_memories: Optional[List[list]] = None,
        user_db=None,
    ) -> List[List[str]]:
        return [
            self.rerank(
                candidates,
                user_id=user_id,
                topk=topk,
                query=query,
                dialogue_query=dialogue_query,
                item_db=item_db,
                retrieval=retrieval,
                user_db=user_db,
                session_memory=session_memory,
            )
            for candidates, user_id, query, dialogue_query, session_memory in zip(
                batch_candidates,
                user_ids,
                queries or [None] * len(batch_candidates),
                dialogue_queries or queries or [None] * len(batch_candidates),
                session_memories or [None] * len(batch_candidates),
            )
        ]
