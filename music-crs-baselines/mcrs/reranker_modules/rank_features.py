"""Feature extraction for LambdaMART learning-to-rank.

AUDIT (pre-fix behavior on hybrid fused pools):
- ``dense_score``: cosine from ``dense_retriever.score_tracks()`` for every candidate id.
  BM25-only tracks still received a real embedding cosine (not NaN/zero), even when absent
  from the dense top-k pool.
- ``dense_rank``: was ``enumerate(candidates)`` — the fused RRF list position, NOT the
  dense retriever rank. Mislabeled and misleading for tree splits.
- No BM25 or RRF features existed; the reranker could not use lexical retrieval signal.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from mcrs.retrieval_modules.rrf import reciprocal_rank_fusion_with_scores

# Sentinel values when a track is missing from a retriever's top-k pool.
MISSING_RANK_OFFSET = 1  # rank_sentinel = pool_depth + MISSING_RANK_OFFSET


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", (text or "").lower()))


def jaccard_similarity(left: str, right: str) -> float:
    left_tokens = _token_set(left)
    right_tokens = _token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = left_tokens & right_tokens
    union = left_tokens | right_tokens
    return len(intersection) / len(union)


def word_overlap_count(left: str, right: str) -> float:
    left_tokens = _token_set(left)
    right_tokens = _token_set(right)
    return float(len(left_tokens & right_tokens))


def _flatten_field(value) -> str:
    if isinstance(value, list):
        return " ".join(str(part) for part in value)
    return str(value or "")


def _track_tag_tokens(track_fields: dict) -> set[str]:
    tags = track_fields.get("tag_list") or []
    tokens: set[str] = set()
    for tag in tags:
        tokens.update(_token_set(str(tag)))
    return tokens


def _user_preference_tokens(user_profile_str: Optional[str], dialogue_query: str) -> set[str]:
    combined = f"{user_profile_str or ''} {dialogue_query or ''}"
    return _token_set(combined)


def user_likes_genre_match(
    track_fields: dict,
    user_profile_str: Optional[str],
    dialogue_query: str,
) -> float:
    """Binary: 1 if any track tag overlaps user profile or dialogue tokens."""
    tag_tokens = _track_tag_tokens(track_fields)
    if not tag_tokens:
        return 0.0
    preference_tokens = _user_preference_tokens(user_profile_str, dialogue_query)
    return float(bool(tag_tokens & preference_tokens))


def recent_dialogue_overlap(track_fields: dict, recent_dialogue: str) -> float:
    """Count overlapping tokens between recent dialogue and track title/artist."""
    track_text = " ".join(
        [
            _flatten_field(track_fields.get("track_name")),
            _flatten_field(track_fields.get("artist_name")),
        ]
    )
    return word_overlap_count(recent_dialogue, track_text)


def extract_recent_dialogue(session_memory: Optional[Sequence[dict]], num_turns: int = 2) -> str:
    """Return text from the last N conversation turns."""
    if not session_memory:
        return ""
    recent = list(session_memory)[-num_turns:]
    return " ".join(turn.get("content", "") for turn in recent)


def _rank_map(ranked_ids: Sequence[str]) -> Dict[str, int]:
    return {track_id: rank for rank, track_id in enumerate(ranked_ids)}


def _score_sentinel(score_map: Dict[str, float]) -> float:
    if not score_map:
        return -1.0
    return min(score_map.values()) - 1.0


def compute_hybrid_retrieval_signals(
    full_query: str,
    retrieval,
    fused_candidates: Sequence[str],
) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, float], Dict[str, int], Dict[str, float]]:
    """
    Build per-candidate dense/BM25/RRF signals for a fused candidate list.

    Returns:
        dense_scores, dense_ranks, bm25_scores, bm25_ranks, rrf_scores
        (rank dicts use 0-based ranks; missing pool entries get rank sentinel)
    """
    pool_topk = getattr(retrieval, "pool_topk", 1000)
    rank_sentinel = float(pool_topk + MISSING_RANK_OFFSET)
    sparse = retrieval.sparse_retriever
    dense = retrieval.dense_retriever
    rrf_k = getattr(retrieval, "rrf_k", 60)

    bm25_ids, bm25_score_map = sparse.retrieve_with_scores(full_query, topk=pool_topk)
    dense_ids = dense.text_to_item_retrieval(full_query, topk=pool_topk)
    dense_score_map = dense.score_tracks(full_query, dense_ids)

    _, rrf_score_map = reciprocal_rank_fusion_with_scores(
        [bm25_ids, dense_ids],
        k=rrf_k,
    )

    bm25_rank_map = _rank_map(bm25_ids)
    dense_rank_map = _rank_map(dense_ids)
    dense_score_floor = _score_sentinel(dense_score_map)
    bm25_score_floor = _score_sentinel(bm25_score_map)

    dense_scores: Dict[str, float] = {}
    dense_ranks: Dict[str, float] = {}
    bm25_scores: Dict[str, float] = {}
    bm25_ranks: Dict[str, float] = {}

    for track_id in fused_candidates:
        if track_id in dense_rank_map:
            dense_ranks[track_id] = float(dense_rank_map[track_id])
            dense_scores[track_id] = float(dense_score_map[track_id])
        else:
            dense_ranks[track_id] = rank_sentinel
            dense_scores[track_id] = dense_score_floor

        if track_id in bm25_rank_map:
            bm25_ranks[track_id] = float(bm25_rank_map[track_id])
            bm25_scores[track_id] = float(bm25_score_map[track_id])
        else:
            bm25_ranks[track_id] = rank_sentinel
            bm25_scores[track_id] = bm25_score_floor

    rrf_scores = {track_id: float(rrf_score_map.get(track_id, 0.0)) for track_id in fused_candidates}
    return dense_scores, dense_ranks, bm25_scores, bm25_ranks, rrf_scores


def compute_dense_only_retrieval_signals(
    full_query: str,
    dense_retriever,
    fused_candidates: Sequence[str],
    pool_topk: int = 200,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Dense-only path: BM25/RRF sentinels, true dense ranks from dense top-k pool."""
    rank_sentinel = float(pool_topk + MISSING_RANK_OFFSET)
    dense_ids = dense_retriever.text_to_item_retrieval(full_query, topk=pool_topk)
    dense_score_map = dense_retriever.score_tracks(full_query, dense_ids)
    dense_rank_map = _rank_map(dense_ids)
    dense_score_floor = _score_sentinel(dense_score_map)
    bm25_score_floor = -1.0

    dense_scores: Dict[str, float] = {}
    dense_ranks: Dict[str, float] = {}
    bm25_scores = {track_id: bm25_score_floor for track_id in fused_candidates}
    bm25_ranks = {track_id: rank_sentinel for track_id in fused_candidates}
    rrf_scores = {track_id: 1.0 / (rank + 1 + 60) for rank, track_id in enumerate(fused_candidates)}

    for track_id in fused_candidates:
        if track_id in dense_rank_map:
            dense_ranks[track_id] = float(dense_rank_map[track_id])
            dense_scores[track_id] = float(dense_score_map[track_id])
        else:
            dense_ranks[track_id] = rank_sentinel
            dense_scores[track_id] = dense_score_floor

    return dense_scores, dense_ranks, bm25_scores, bm25_ranks, rrf_scores


def build_lambdamart_features(
    candidates: Sequence[str],
    full_query: str,
    dialogue_query: str,
    dense_retriever,
    item_db,
    popularity_scores: Dict[str, float],
    user_profile_str: Optional[str] = None,
    recent_dialogue: Optional[str] = None,
    retrieval=None,
    pool_topk: Optional[int] = None,
) -> List[dict]:
    user_text = dialogue_query or full_query
    recent_dialogue = recent_dialogue if recent_dialogue is not None else dialogue_query

    if retrieval is not None and hasattr(retrieval, "sparse_retriever"):
        dense_scores, dense_ranks, bm25_scores, bm25_ranks, rrf_scores = compute_hybrid_retrieval_signals(
            full_query,
            retrieval,
            candidates,
        )
    else:
        pool = pool_topk or getattr(dense_retriever, "pool_topk", None) or 200
        dense_scores, dense_ranks, bm25_scores, bm25_ranks, rrf_scores = compute_dense_only_retrieval_signals(
            full_query,
            dense_retriever,
            candidates,
            pool_topk=pool,
        )

    rows: List[dict] = []
    for fused_rank, track_id in enumerate(candidates):
        metadata = item_db.id_to_metadata(track_id)
        track_fields = item_db.id_to_fields(track_id)
        rows.append(
            {
                "track_id": track_id,
                "dense_rank": dense_ranks[track_id],
                "dense_score": dense_scores[track_id],
                "bm25_rank": bm25_ranks[track_id],
                "bm25_score": bm25_scores[track_id],
                "rrf_score": rrf_scores[track_id],
                "retrieval_prior": 1.0 / (fused_rank + 1),
                "popularity": popularity_scores.get(track_id, 0.0),
                "text_jaccard": jaccard_similarity(user_text, metadata),
                "word_overlap": word_overlap_count(user_text, metadata),
                "query_len": float(len(_token_set(user_text))),
                "user_likes_genre_match": user_likes_genre_match(
                    track_fields, user_profile_str, dialogue_query
                ),
                "recent_dialogue_overlap": recent_dialogue_overlap(track_fields, recent_dialogue),
            }
        )
    return rows


# Legacy v7 model (7 features) — auto-detected at inference from model.feature_name().
LAMBDAMART_FEATURE_COLUMNS_LEGACY = [
    "dense_rank",
    "dense_score",
    "retrieval_prior",
    "popularity",
    "text_jaccard",
    "word_overlap",
    "query_len",
]

# Full feature set for hybrid-pool training (dense + BM25 + RRF + metadata).
LAMBDAMART_FEATURE_COLUMNS = [
    "dense_rank",
    "dense_score",
    "bm25_rank",
    "bm25_score",
    "rrf_score",
    "retrieval_prior",
    "popularity",
    "text_jaccard",
    "word_overlap",
    "query_len",
    "user_likes_genre_match",
    "recent_dialogue_overlap",
]
