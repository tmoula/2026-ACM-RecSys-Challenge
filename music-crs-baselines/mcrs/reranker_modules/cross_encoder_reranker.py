"""Cross-encoder reranker for retrieval candidates."""

from typing import List, Optional

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """Rerank candidates using a cross-encoder relevance model."""

    needs_item_db = True

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cuda",
        batch_size: int = 32,
        ce_topk: int = 200,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.ce_topk = ce_topk
        self.model = CrossEncoder(model_name, device=device)

    def _score_pairs(self, query: str, track_ids: List[str], item_db) -> dict[str, float]:
        if not track_ids:
            return {}
        pairs = [(query, item_db.id_to_metadata(track_id)) for track_id in track_ids]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return {track_id: float(score) for track_id, score in zip(track_ids, scores)}

    def rerank(
        self,
        candidates: List[str],
        user_id: Optional[str] = None,
        topk: int = 20,
        query: Optional[str] = None,
        item_db=None,
    ) -> List[str]:
        if not candidates or not query or item_db is None:
            return candidates[:topk]

        pool = candidates[: self.ce_topk]
        scores = self._score_pairs(query, pool, item_db)
        reranked = sorted(pool, key=lambda track_id: scores.get(track_id, 0.0), reverse=True)
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
        item_db=None,
    ) -> List[List[str]]:
        return [
            self.rerank(
                candidates,
                user_id=user_id,
                topk=topk,
                query=query,
                item_db=item_db,
            )
            for candidates, user_id, query in zip(batch_candidates, user_ids, queries or [None] * len(batch_candidates))
        ]

    def batch_score(
        self,
        batch_candidates: List[List[str]],
        queries: List[str],
        item_db,
    ) -> List[dict[str, float]]:
        return [
            self._score_pairs(query, candidates[: self.ce_topk], item_db)
            for candidates, query in zip(batch_candidates, queries)
        ]
