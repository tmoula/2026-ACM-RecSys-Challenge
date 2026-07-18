"""Collaborative filtering retrieval via precomputed user/track embeddings."""

import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from datasets import concatenate_datasets, load_dataset


class CFRetriever:
    """Top-k retrieval by dot product between user and track CF embeddings."""

    def __init__(
        self,
        track_embedding_name: str = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
        user_embedding_name: str = "talkpl-ai/TalkPlayData-Challenge-User-Embeddings",
        track_split_types: Optional[List[str]] = None,
        user_split_types: Optional[List[str]] = None,
        embedding_field: str = "cf-bpr",
        cache_dir: str = "./cache",
    ) -> None:
        track_split_types = track_split_types or ["all_tracks"]
        user_split_types = user_split_types or ["train", "test_warm", "test_cold"]
        cache_path = os.path.join(
            cache_dir,
            "cf",
            f"{embedding_field}_{'_'.join(track_split_types)}_{'_'.join(user_split_types)}.pt",
        )

        if os.path.exists(cache_path):
            cached = torch.load(cache_path, map_location="cpu")
            self.track_ids: List[str] = cached["track_ids"]
            self.track_matrix: torch.Tensor = cached["track_matrix"]
            self.user_embeddings: Dict[str, torch.Tensor] = cached["user_embeddings"]
            self.track_id_to_index = {track_id: index for index, track_id in enumerate(self.track_ids)}
            return

        track_ds = concatenate_datasets(
            [load_dataset(track_embedding_name, split=split_type) for split_type in track_split_types]
        )
        user_ds = concatenate_datasets(
            [load_dataset(user_embedding_name, split=split_type) for split_type in user_split_types]
        )

        track_ids: List[str] = []
        track_rows: List[torch.Tensor] = []
        for row in track_ds:
            raw = row.get(embedding_field) or []
            if not raw:
                continue
            emb = torch.tensor(raw, dtype=torch.float32)
            if emb.numel() == 0:
                continue
            track_ids.append(row["track_id"])
            track_rows.append(F.normalize(emb, p=2, dim=0))

        self.track_ids = track_ids
        self.track_matrix = torch.stack(track_rows, dim=0).contiguous()
        self.track_id_to_index = {track_id: index for index, track_id in enumerate(self.track_ids)}

        self.user_embeddings: Dict[str, torch.Tensor] = {}
        for row in user_ds:
            raw = row.get(embedding_field) or []
            if not raw:
                continue
            emb = torch.tensor(raw, dtype=torch.float32)
            if emb.numel() == 0:
                continue
            self.user_embeddings[row["user_id"]] = F.normalize(emb, p=2, dim=0)

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(
            {
                "track_ids": self.track_ids,
                "track_matrix": self.track_matrix,
                "user_embeddings": self.user_embeddings,
            },
            cache_path,
        )

    def retrieve(self, user_id: Optional[str], topk: int) -> List[str]:
        if not user_id or user_id not in self.user_embeddings:
            return []
        user_emb = self.user_embeddings[user_id]
        scores = torch.matmul(self.track_matrix, user_emb)
        k = min(topk, scores.numel())
        top_indices = torch.topk(scores, k=k).indices.tolist()
        return [self.track_ids[index] for index in top_indices]

    def batch_retrieve(self, user_ids: List[Optional[str]], topk: int) -> List[List[str]]:
        return [self.retrieve(user_id, topk) for user_id in user_ids]

    def score_tracks(self, user_id: Optional[str], track_ids: List[str]) -> Dict[str, float]:
        if not user_id or user_id not in self.user_embeddings:
            return {track_id: 0.0 for track_id in track_ids}
        user_emb = self.user_embeddings[user_id]
        scores: Dict[str, float] = {}
        for track_id in track_ids:
            index = self.track_id_to_index.get(track_id)
            if index is None:
                scores[track_id] = 0.0
            else:
                scores[track_id] = torch.dot(user_emb, self.track_matrix[index]).item()
        return scores
