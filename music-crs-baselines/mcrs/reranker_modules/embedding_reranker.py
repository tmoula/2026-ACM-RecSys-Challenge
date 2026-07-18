"""Rerank retrieval candidates using precomputed user/track embeddings."""

import json
import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from datasets import concatenate_datasets, load_dataset


class EmbeddingReranker:
    """Rerank candidates by blending retrieval rank with user-item embedding similarity."""

    def __init__(
        self,
        track_embedding_name: str = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
        user_embedding_name: str = "talkpl-ai/TalkPlayData-Challenge-User-Embeddings",
        track_split_types: Optional[List[str]] = None,
        user_split_types: Optional[List[str]] = None,
        embedding_field: str = "cf-bpr",
        cache_dir: str = "./cache",
        retrieval_weight: float = 0.4,
        user_weight: float = 0.6,
    ) -> None:
        self.embedding_field = embedding_field
        self.retrieval_weight = retrieval_weight
        self.user_weight = user_weight
        track_split_types = track_split_types or ["all_tracks"]
        user_split_types = user_split_types or ["train", "test_warm", "test_cold"]
        cache_path = os.path.join(
            cache_dir,
            "reranker",
            f"{embedding_field}_{'_'.join(track_split_types)}_{'_'.join(user_split_types)}.pt",
        )

        if os.path.exists(cache_path):
            cached = torch.load(cache_path, map_location="cpu")
            self.track_embeddings = cached["track_embeddings"]
            self.user_embeddings = cached["user_embeddings"]
            return

        track_ds = concatenate_datasets(
            [load_dataset(track_embedding_name, split=split_type) for split_type in track_split_types]
        )
        user_ds = concatenate_datasets(
            [load_dataset(user_embedding_name, split=split_type) for split_type in user_split_types]
        )

        self.track_embeddings: Dict[str, torch.Tensor] = {}
        for row in track_ds:
            raw = row.get(self.embedding_field) or []
            if not raw:
                continue
            emb = torch.tensor(raw, dtype=torch.float32)
            if emb.numel() == 0:
                continue
            self.track_embeddings[row["track_id"]] = F.normalize(emb, p=2, dim=0)

        self.user_embeddings: Dict[str, torch.Tensor] = {}
        for row in user_ds:
            raw = row.get(self.embedding_field) or []
            if not raw:
                continue
            emb = torch.tensor(raw, dtype=torch.float32)
            if emb.numel() == 0:
                continue
            self.user_embeddings[row["user_id"]] = F.normalize(emb, p=2, dim=0)

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(
            {"track_embeddings": self.track_embeddings, "user_embeddings": self.user_embeddings},
            cache_path,
        )

    def rerank(
        self,
        candidates: List[str],
        user_id: Optional[str] = None,
        topk: int = 20,
    ) -> List[str]:
        if not candidates:
            return []

        user_emb = self.user_embeddings.get(user_id) if user_id else None
        scored = []
        for rank, track_id in enumerate(candidates):
            retrieval_score = 1.0 / (rank + 1)
            user_score = 0.0
            if user_emb is not None:
                track_emb = self.track_embeddings.get(track_id)
                if track_emb is not None and track_emb.numel() == user_emb.numel():
                    user_score = torch.dot(user_emb, track_emb).item()
            score = self.retrieval_weight * retrieval_score + self.user_weight * user_score
            scored.append((track_id, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        reranked = []
        seen = set()
        for track_id, _ in scored:
            if track_id in seen:
                continue
            seen.add(track_id)
            reranked.append(track_id)
            if len(reranked) == topk:
                break
        return reranked

    def batch_rerank(
        self,
        batch_candidates: List[List[str]],
        user_ids: List[Optional[str]],
        topk: int = 20,
    ) -> List[List[str]]:
        return [
            self.rerank(candidates, user_id, topk=topk)
            for candidates, user_id in zip(batch_candidates, user_ids)
        ]
