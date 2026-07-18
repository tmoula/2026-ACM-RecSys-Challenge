"""Dense embedding retrieval over track metadata."""

import json
import os
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from datasets import concatenate_datasets, load_dataset
from transformers import AutoModel, AutoTokenizer

# Query/passage prefixes for common embedding models.
MODEL_PREFIXES: Dict[str, Tuple[str, str]] = {
    "bert-base-uncased": ("", ""),
    "BAAI/bge-small-en-v1.5": (
        "Represent this sentence for searching relevant passages: ",
        "",
    ),
    "intfloat/e5-base-v2": ("query: ", "passage: "),
}


class DenseRetriever:
    """Dense retriever using a transformer encoder and cosine similarity."""

    def __init__(
        self,
        dataset_name: str,
        split_types: List[str],
        corpus_types: List[str],
        cache_dir: str = "./cache",
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 512,
        query_prefix: str | None = None,
        passage_prefix: str | None = None,
        pool_topk: int | None = None,
        mmr_lambda: float | None = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.split_types = split_types
        self.corpus_types = corpus_types
        self.corpus_name = "_".join(corpus_types)
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.model_slug = (
            model_name.replace("/", "_").replace(".", "_").replace(" ", "_").strip("_")
        )
        self.index_dir = os.path.join(self.cache_dir, "dense", self.model_slug, self.corpus_name)
        self.batch_size = batch_size
        self.max_length = max_length

        default_query_prefix, default_passage_prefix = MODEL_PREFIXES.get(model_name, ("", ""))
        self.query_prefix = default_query_prefix if query_prefix is None else query_prefix
        self.passage_prefix = default_passage_prefix if passage_prefix is None else passage_prefix
        self.pool_topk = pool_topk
        self.mmr_lambda = mmr_lambda

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.metadata_dict = self._load_corpus()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device).eval()

        if os.path.exists(os.path.join(self.index_dir, "embeddings.pt")) and os.path.exists(
            os.path.join(self.index_dir, "track_ids.json")
        ):
            self.embeddings, self.track_ids = self._load_index()
        else:
            self.build_index()
            self.embeddings, self.track_ids = self._load_index()

    def _load_index(self) -> Tuple[torch.Tensor, List[str]]:
        embeddings = torch.load(os.path.join(self.index_dir, "embeddings.pt"), map_location="cpu")
        track_ids = json.load(open(os.path.join(self.index_dir, "track_ids.json"), "r"))
        return embeddings, track_ids

    def _load_corpus(self) -> Dict[str, dict]:
        metadata_dataset = load_dataset(self.dataset_name)
        metadata_concat_dataset = concatenate_datasets(
            [metadata_dataset[split_type] for split_type in self.split_types]
        )
        return {item["track_id"]: item for item in metadata_concat_dataset}

    def _stringify_metadata(self, metadata: dict) -> str:
        metadata_str = ""
        for corpus_type in self.corpus_types:
            entity = metadata[corpus_type]
            if isinstance(entity, list):
                entity = ", ".join(str(x) for x in entity)
            metadata_str += f"{corpus_type}: {entity}\n"
        return self.passage_prefix + metadata_str.strip()

    def _mean_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
        summed = torch.sum(last_hidden_states * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        self.model.eval()
        embeddings: List[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch_texts = texts[start : start + self.batch_size]
                batch = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                batch = {key: value.to(self.device) for key, value in batch.items()}
                outputs = self.model(**batch)
                pooled = self._mean_pool(outputs.last_hidden_state, batch["attention_mask"])
                pooled = F.normalize(pooled, p=2, dim=1)
                embeddings.append(pooled.detach().cpu())
        return torch.cat(embeddings, dim=0).contiguous()

    def build_index(self) -> None:
        track_ids = list(self.metadata_dict.keys())
        corpus_texts = [self._stringify_metadata(self.metadata_dict[track_id]) for track_id in track_ids]
        os.makedirs(self.index_dir, exist_ok=True)
        embedding_mat = self._encode_texts(corpus_texts)
        torch.save(embedding_mat, os.path.join(self.index_dir, "embeddings.pt"))
        with open(os.path.join(self.index_dir, "track_ids.json"), "w", encoding="utf-8") as file:
            json.dump(track_ids, file, indent=2)

    def _encode_queries(self, queries: List[str]) -> torch.Tensor:
        prefixed = [self.query_prefix + query for query in queries]
        return self._encode_texts(prefixed)

    def _resolve_pool_topk(self, topk: int) -> int:
        if self.pool_topk is not None:
            return min(self.pool_topk, len(self.track_ids))
        if self.mmr_lambda is not None:
            return min(max(topk * 5, 100), len(self.track_ids))
        return min(topk, len(self.track_ids))

    def _mmr_select(
        self,
        query_emb: torch.Tensor,
        candidate_indices: List[int],
        topk: int,
    ) -> List[str]:
        """Maximal marginal relevance over a candidate pool."""
        if self.mmr_lambda is None or len(candidate_indices) <= topk:
            return [self.track_ids[index] for index in candidate_indices[:topk]]

        lambda_relevance = float(self.mmr_lambda)
        candidate_embs = self.embeddings[candidate_indices]
        relevance = torch.matmul(candidate_embs, query_emb)

        selected_local: List[int] = []
        remaining = list(range(len(candidate_indices)))

        while remaining and len(selected_local) < topk:
            best_local = None
            best_score = float("-inf")
            for local_index in remaining:
                rel = relevance[local_index].item()
                if selected_local:
                    selected_embs = candidate_embs[selected_local]
                    redundancy = torch.max(
                        torch.matmul(selected_embs, candidate_embs[local_index])
                    ).item()
                else:
                    redundancy = 0.0
                score = lambda_relevance * rel - (1.0 - lambda_relevance) * redundancy
                if score > best_score:
                    best_score = score
                    best_local = local_index
            selected_local.append(best_local)
            remaining.remove(best_local)

        return [self.track_ids[candidate_indices[index]] for index in selected_local]

    def _rank_candidates(self, query_emb: torch.Tensor, topk: int) -> List[str]:
        pool_topk = self._resolve_pool_topk(topk)
        scores = torch.matmul(self.embeddings, query_emb)
        pool_indices = torch.topk(scores, k=pool_topk).indices.tolist()
        if self.mmr_lambda is None:
            return [self.track_ids[index] for index in pool_indices[:topk]]
        return self._mmr_select(query_emb, pool_indices, topk)

    def text_to_item_retrieval(self, query: str, topk: int) -> List[str]:
        query_emb = self._encode_queries([query]).squeeze(0)
        return self._rank_candidates(query_emb, topk)

    def batch_text_to_item_retrieval(self, queries: List[str], topk: int) -> List[List[str]]:
        query_embs = self._encode_queries(queries)
        return [self._rank_candidates(query_embs[index], topk) for index in range(len(queries))]

    def score_tracks(self, query: str, track_ids: List[str]) -> Dict[str, float]:
        if not track_ids:
            return {}
        query_emb = self._encode_queries([query]).squeeze(0)
        track_id_to_index = {track_id: index for index, track_id in enumerate(self.track_ids)}
        scores: Dict[str, float] = {}
        for track_id in track_ids:
            index = track_id_to_index.get(track_id)
            if index is None:
                scores[track_id] = 0.0
            else:
                scores[track_id] = torch.dot(query_emb, self.embeddings[index]).item()
        return scores


# Backward-compatible alias used by older configs.
BERT_MODEL = DenseRetriever
