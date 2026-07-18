"""Shared feature extraction for reranking."""

import json
import os
from collections import Counter
from typing import Dict, List, Optional

from datasets import load_dataset


def load_popularity_scores(cache_dir: str = "./cache") -> Dict[str, float]:
    cache_path = os.path.join(cache_dir, "reranker", "popularity_scores.json")
    if os.path.exists(cache_path):
        return json.load(open(cache_path, "r", encoding="utf-8"))

    db = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="train")
    counts: Counter[str] = Counter()
    for item in db:
        for turn in item["conversations"]:
            if turn["role"] == "music":
                counts[turn["content"]] += 1

    if not counts:
        popularity = {}
    else:
        max_count = max(counts.values())
        popularity = {track_id: count / max_count for track_id, count in counts.items()}

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as file:
        json.dump(popularity, file)
    return popularity


def build_candidate_features(
    candidates: List[str],
    full_query: str,
    dialogue_query: str,
    user_id: Optional[str],
    retrieval_meta: Optional[dict],
    dense_retriever,
    cf_retriever,
    popularity_scores: Dict[str, float],
    ce_score: Optional[float] = None,
) -> List[dict]:
    dense_scores = dense_retriever.score_tracks(full_query, candidates)
    cf_scores = cf_retriever.score_tracks(user_id, candidates)

    dense_rank = (retrieval_meta or {}).get("dense_rank", {})
    sparse_rank = (retrieval_meta or {}).get("sparse_rank", {})
    cf_rank = (retrieval_meta or {}).get("cf_rank", {})
    rrf_scores = (retrieval_meta or {}).get("rrf_score", {})

    rows = []
    for rrf_rank, track_id in enumerate(candidates):
        rows.append(
            {
                "track_id": track_id,
                "rrf_rank": rrf_rank,
                "rrf_score": rrf_scores.get(track_id, 0.0),
                "dense_rank": dense_rank.get(track_id, 9999),
                "sparse_rank": sparse_rank.get(track_id, 9999),
                "cf_rank": cf_rank.get(track_id, 9999),
                "dense_score": dense_scores.get(track_id, 0.0),
                "cf_score": cf_scores.get(track_id, 0.0),
                "popularity": popularity_scores.get(track_id, 0.0),
                "ce_score": ce_score if ce_score is not None else 0.0,
            }
        )
    return rows


FEATURE_COLUMNS = [
    "rrf_rank",
    "rrf_score",
    "dense_rank",
    "sparse_rank",
    "cf_rank",
    "dense_score",
    "cf_score",
    "popularity",
    "ce_score",
]
