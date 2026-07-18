"""
Measure retrieval recall@K on the devset (pre-rerank diagnostic).

Uses the same config, query construction, and retrieval module as inference.

Usage (Colab):
  # Baseline dense pool 200 (original diagnostic)
  !python recall_at_k.py --tid llama1b_lambdamart_devset

  # Wider dense pool 1000 (works on any zip with llama1b_lambdamart_devset.yaml)
  !python recall_at_k.py --tid llama1b_lambdamart_devset --retrieve-topk 1000

  # BM25 + dense RRF merged pool 1000
  !python recall_at_k.py --tid llama1b_lambdamart_devset --retrieve-topk 1000 --hybrid

  # Quick pool size sweep on dense
  !python recall_at_k.py --tid llama1b_lambdamart_devset --retrieve-topk 500
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm

from mcrs import load_crs_baseline
from mcrs.utils.ground_truth import load_or_build_ground_truth
from run_inference_blindset import build_turn_examples, optional_config_dict, resolve_device

RECALL_AT_K_VERSION = "2026-06-22-pool"

K_VALUES: Tuple[int, ...] = (20, 50, 100, 200, 500, 1000)
MAX_K = max(K_VALUES)
REPO_ROOT = Path(__file__).resolve().parent


def resolve_checkpoint_path(model_name: str) -> Path:
    path = Path(model_name).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def require_local_checkpoint(model_name: str) -> str:
    """Fail fast if the FT checkpoint directory is missing (avoids HF hub repo-id error)."""
    path = resolve_checkpoint_path(model_name)
    config_file = path / "config.json"
    if path.is_dir() and config_file.is_file():
        return str(path)

    print("ERROR: Fine-tuned checkpoint not found on disk.", file=sys.stderr)
    print(f"  Expected directory: {path}", file=sys.stderr)
    print(f"  Missing file:       {config_file}", file=sys.stderr)
    print(file=sys.stderr)
    print("Copy the checkpoint from Colab / Google Drive, e.g.:", file=sys.stderr)
    print("  /content/drive/MyDrive/recsys-checkpoints/bge-talkplay-v5-bs32", file=sys.stderr)
    raise SystemExit(1)


def _checkpoint_path_from_config(config) -> str:
    retrieval_kwargs = optional_config_dict(config, "retrieval_kwargs") or {}
    if config.retrieval_type == "hybrid":
        dense_kwargs = retrieval_kwargs.get("dense_kwargs") or {}
        return dense_kwargs.get("model_name", "")
    return retrieval_kwargs.get("model_name", "")


def apply_checkpoint_override(config, checkpoint: str | None) -> str | None:
    if not checkpoint and config.retrieval_type != "hybrid":
        model_name = _checkpoint_path_from_config(config)
        if model_name and not model_name.startswith(("BAAI/", "bert-", "intfloat/")):
            require_local_checkpoint(model_name)
        return model_name or None

    model_name = checkpoint or _checkpoint_path_from_config(config)
    if not model_name:
        print("ERROR: No dense checkpoint in config and no --checkpoint given.", file=sys.stderr)
        raise SystemExit(1)
    model_name = require_local_checkpoint(model_name)

    if config.retrieval_kwargs is None:
        config.retrieval_kwargs = {}
    if config.retrieval_type == "hybrid":
        if config.retrieval_kwargs.get("dense_kwargs") is None:
            config.retrieval_kwargs.dense_kwargs = {}
        config.retrieval_kwargs.dense_kwargs.model_name = model_name
    else:
        config.retrieval_kwargs.model_name = model_name
    return model_name


def apply_retrieve_topk_override(config, retrieve_topk: int | None) -> int:
    pool_k = retrieve_topk if retrieve_topk is not None else int(config.get("retrieve_topk", 200))
    config.retrieve_topk = pool_k

    retrieval_kwargs = optional_config_dict(config, "retrieval_kwargs") or {}
    if config.retrieval_type == "hybrid":
        if config.retrieval_kwargs is None:
            config.retrieval_kwargs = {}
        current_pool = retrieval_kwargs.get("pool_topk")
        if current_pool is None or current_pool < pool_k:
            config.retrieval_kwargs.pool_topk = pool_k
    return pool_k


def apply_hybrid_override(config, use_hybrid: bool, pool_k: int, rrf_k: int) -> None:
    """Enable BM25+dense RRF from a dense devset config (no extra yaml required)."""
    if not use_hybrid:
        return
    if config.retrieval_type == "hybrid":
        return

    retrieval_kwargs = optional_config_dict(config, "retrieval_kwargs") or {}
    dense_kwargs = {
        key: retrieval_kwargs[key]
        for key in ("model_name", "max_length", "query_prefix", "passage_prefix")
        if key in retrieval_kwargs
    }
    config.retrieval_type = "hybrid"
    config.retrieval_kwargs = {
        "pool_topk": pool_k,
        "rrf_k": rrf_k,
        "dense_kwargs": dense_kwargs,
    }


def retrieval_track_ids(retrieval) -> set[str]:
    if hasattr(retrieval, "track_ids"):
        return set(retrieval.track_ids)
    if hasattr(retrieval, "dense_retriever"):
        return set(retrieval.dense_retriever.track_ids)
    raise RuntimeError(f"Cannot resolve track ids from {type(retrieval).__name__}")


def retrieval_label(retrieval_type: str) -> str:
    if retrieval_type == "hybrid":
        return "BM25 + dense RRF merged pool"
    if retrieval_type in ("dense", "bert"):
        return "dense retriever"
    return retrieval_type


def load_music_crs(config, device: str):
    retrieval_kwargs = optional_config_dict(config, "retrieval_kwargs")
    return load_crs_baseline(
        lm_type=config.lm_type,
        retrieval_type=config.retrieval_type,
        item_db_name=config.item_db_name,
        user_db_name=config.user_db_name,
        track_split_types=config.track_split_types,
        user_split_types=config.user_split_types,
        corpus_types=config.corpus_types,
        cache_dir=config.cache_dir,
        device=device,
        attn_implementation=config.attn_implementation,
        dtype=torch.bfloat16,
        retrieve_topk=config.get("retrieve_topk", 200),
        final_topk=config.get("final_topk", 20),
        retrieval_kwargs=retrieval_kwargs,
        include_user_profile_in_retrieval=config.get("include_user_profile_in_retrieval", False),
        load_lm=False,
    )


def build_retrieval_query(music_crs, example: dict) -> str:
    session_memory = example["session_memory"].copy()
    session_memory.append({"role": "user", "content": example["user_query"]})
    return music_crs._build_retrieval_input(session_memory, user_id=example["user_id"])


def retrieve_candidates(retrieval, queries: List[str], topk: int) -> List[List[str]]:
    if hasattr(retrieval, "batch_text_to_item_retrieval"):
        return retrieval.batch_text_to_item_retrieval(queries, topk=topk)
    return [retrieval.text_to_item_retrieval(query, topk=topk) for query in queries]


def rank_in_list(track_id: str, candidates: Sequence[str]) -> int | None:
    try:
        return candidates.index(track_id) + 1
    except ValueError:
        return None


def print_recall_table(hits: Dict[int, int], evaluated: int, label: str) -> None:
    print(f"\n=== Recall@K ({label}, pre-rerank) ===")
    print(f"Evaluated turns: {evaluated}")
    print(f"{'K':>6}  {'Hits':>8}  {'Recall':>10}")
    print("-" * 28)
    for k in K_VALUES:
        recall = hits[k] / evaluated if evaluated else 0.0
        print(f"{k:>6}  {hits[k]:>8}  {recall:>10.4f}")


def print_rank_distribution(ranks: List[int], pool_k: int) -> None:
    print(f"\n=== Ground-truth rank when in top-{pool_k} ===")
    print(f"Turns with GT in top-{pool_k}: {len(ranks)}")
    if not ranks:
        print(f"(no hits in top-{pool_k} — retrieval is missing GT entirely)")
        return

    ranks_sorted = sorted(ranks)
    median = statistics.median(ranks_sorted)
    mean = statistics.mean(ranks_sorted)
    in_top_20 = sum(1 for rank in ranks if rank <= 20)
    in_top_50 = sum(1 for rank in ranks if rank <= 50)
    in_top_100 = sum(1 for rank in ranks if rank <= 100)
    pct = lambda count: 100.0 * count / len(ranks)

    print(f"  Median rank:     {median:.1f}")
    print(f"  Mean rank:       {mean:.1f}")
    print(f"  Min / Max rank:  {ranks_sorted[0]} / {ranks_sorted[-1]}")
    print(f"  In top-20:       {in_top_20:>5}  ({pct(in_top_20):.1f}%)")
    print(f"  In top-50:       {in_top_50:>5}  ({pct(in_top_50):.1f}%)")
    print(f"  In top-100:      {in_top_100:>5}  ({pct(in_top_100):.1f}%)")


def main(args: argparse.Namespace) -> None:
    config = OmegaConf.load(f"config/{args.tid}.yaml")
    dense_checkpoint = apply_checkpoint_override(config, args.checkpoint)
    pool_k = apply_retrieve_topk_override(config, args.retrieve_topk)
    apply_hybrid_override(config, args.hybrid, pool_k, args.rrf_k)

    device = resolve_device(config.device)
    if device != config.device:
        print(f"Device {config.device!r} unavailable, using {device!r}")

    music_crs = load_music_crs(config, device)
    retrieval = music_crs.retrieval
    if not hasattr(retrieval, "text_to_item_retrieval"):
        raise RuntimeError(f"Unsupported retriever for recall@K: {type(retrieval).__name__}")

    label = retrieval_label(config.retrieval_type)
    index_ids = retrieval_track_ids(retrieval)
    eval_topk = min(MAX_K, pool_k)

    print("=== Recall@K diagnostic ===")
    print(f"Script version:          {RECALL_AT_K_VERSION}")
    print(f"Config tid:              {args.tid}")
    print(f"Retrieval:               {config.retrieval_type} ({label})")
    print(f"Dataset:                 {config.test_dataset_name} (split=test)")
    print(f"Candidate pool (topk):   {pool_k}")
    if dense_checkpoint:
        print(f"Dense checkpoint:        {dense_checkpoint}")
    if config.retrieval_type == "hybrid":
        rk = optional_config_dict(config, "retrieval_kwargs") or {}
        print(f"RRF pool_topk / k:       {rk.get('pool_topk')} / {rk.get('rrf_k', 60)}")
    print(f"Index tracks:            {len(index_ids)}")
    print(f"include_user_profile:    {config.get('include_user_profile_in_retrieval', False)}")
    print(f"Max eval depth:          {eval_topk}")

    ground_truth = load_or_build_ground_truth(
        args.ground_truth,
        dataset_name=config.test_dataset_name,
        split="test",
    )
    catalog_ids = set(music_crs.item_db.metadata_dict.keys())

    db = load_dataset(config.test_dataset_name, split="test")
    examples = build_turn_examples(db, music_crs)
    print(f"Sessions in devset:      {len(db)}")
    print(f"Turn examples:           {len(examples)}")

    missing_label = 0
    gt_not_in_catalog = 0
    gt_not_in_index = 0
    evaluated_examples: List[dict] = []

    for example in examples:
        key = (example["session_id"], example["turn_number"])
        gt_id = ground_truth.get(key)
        if gt_id is None:
            missing_label += 1
            continue
        if gt_id not in catalog_ids:
            gt_not_in_catalog += 1
            continue
        if gt_id not in index_ids:
            gt_not_in_index += 1
            continue
        evaluated_examples.append({**example, "ground_truth_track_id": gt_id})

    print("\n=== Ground-truth coverage ===")
    print(f"Missing label:           {missing_label}")
    print(f"GT not in metadata:      {gt_not_in_catalog}")
    print(f"GT not in index:         {gt_not_in_index}")
    print(f"Evaluated turns:         {len(evaluated_examples)}")

    hits = {k: 0 for k in K_VALUES}
    ranks_in_pool: List[int] = []

    for start in tqdm(range(0, len(evaluated_examples), args.batch_size), desc="Retrieving"):
        batch = evaluated_examples[start : start + args.batch_size]
        queries = [build_retrieval_query(music_crs, row) for row in batch]
        batch_candidates = retrieve_candidates(retrieval, queries, topk=eval_topk)

        for row, candidates in zip(batch, batch_candidates):
            gt_id = row["ground_truth_track_id"]
            for k in K_VALUES:
                if gt_id in candidates[: min(k, eval_topk)]:
                    hits[k] += 1

            rank = rank_in_list(gt_id, candidates[:pool_k])
            if rank is not None:
                ranks_in_pool.append(rank)

        if device == "cuda":
            torch.cuda.empty_cache()

    evaluated = len(evaluated_examples)
    print_recall_table(hits, evaluated, label)
    print_rank_distribution(ranks_in_pool, pool_k=min(pool_k, eval_topk))

    pool_hits = len(ranks_in_pool)
    missed_at_pool = evaluated - pool_hits
    print("\n=== Bottleneck hint ===")
    if evaluated:
        print(
            f"GT missing from top-{min(pool_k, eval_topk)}: "
            f"{missed_at_pool} turns ({100.0 * missed_at_pool / evaluated:.1f}%)"
        )
        if ranks_in_pool:
            not_top_20 = sum(1 for rank in ranks_in_pool if rank > 20)
            print(
                f"GT in pool but rank>20: {not_top_20} turns "
                f"({100.0 * not_top_20 / len(ranks_in_pool):.1f}% of pool hits) "
                "→ reranker can help"
            )
        if missed_at_pool:
            print("→ widen pool, add hybrid RRF, or improve dense FT")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieval recall@K on devset (pre-rerank).")
    parser.add_argument(
        "--tid",
        type=str,
        default="llama1b_lambdamart_devset",
        help="Config under config/ (dense or hybrid).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Override dense checkpoint (dense_kwargs.model_name for hybrid).",
    )
    parser.add_argument(
        "--retrieve-topk",
        type=int,
        default=None,
        help="Override retrieve_topk / hybrid pool_topk (e.g. 500 vs 1000).",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Fuse BM25 + dense with RRF (uses dense fields from config yaml).",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF constant k when --hybrid is set.",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default="./cache/reranker/devset_ground_truth.json",
        help="Ground-truth JSON; built automatically if missing.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for retrieval encoding.",
    )
    main(parser.parse_args())
