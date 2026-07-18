"""Reciprocal rank fusion utilities."""

from typing import List


def reciprocal_rank_fusion(ranked_lists: List[List[str]], k: int = 60) -> List[str]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item_id in enumerate(ranked):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda item_id: scores[item_id], reverse=True)


def reciprocal_rank_fusion_with_scores(
    ranked_lists: List[List[str]], k: int = 60
) -> tuple[List[str], dict[str, float]]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item_id in enumerate(ranked):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.keys(), key=lambda item_id: scores[item_id], reverse=True)
    return fused, scores
