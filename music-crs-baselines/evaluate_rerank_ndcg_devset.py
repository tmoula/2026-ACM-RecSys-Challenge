"""
End-to-end rerank-only nDCG@20 on devset (no LM / GPT).

Compares hybrid+retrained LambdaMART vs legacy dense-200 pipeline.

Usage:
  python evaluate_rerank_ndcg_devset.py \
    --baseline-tid llama1b_v5_lam   bdamart_blindset_A \
    --candidate-tid llama1b_hybrid_lambdamart_devset
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm

from mcrs import load_crs_baseline
from mcrs.utils.conversation import parse_conversation_history
from run_inference_blindset import optional_config_dict, resolve_device


def load_ground_truth(path: str) -> dict[tuple[str, int], str]:
    rows = json.load(open(path, "r", encoding="utf-8"))
    return {(row["session_id"], row["turn_number"]): row["ground_truth_track_id"] for row in rows}


def ndcg_at_k(relevance: list[int], k: int) -> float:
    rel = np.asarray(relevance[:k], dtype=np.float64)
    if rel.sum() == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, len(rel) + 2))
    dcg = float((rel * discounts).sum())
    ideal = np.sort(rel)[::-1]
    idcg = float((ideal * discounts).sum())
    return dcg / idcg if idcg > 0 else 0.0


def load_pipeline(tid: str, device: str, load_lm: bool = False):
    config = OmegaConf.load(f"config/{tid}.yaml")
    config.device = device
    retrieval_kwargs = optional_config_dict(config, "retrieval_kwargs")
    reranker_kwargs = optional_config_dict(config, "reranker_kwargs")
    return config, load_crs_baseline(
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
        reranker_type=config.get("reranker_type"),
        reranker_kwargs=reranker_kwargs,
        retrieve_topk=config.get("retrieve_topk", 200),
        final_topk=config.get("final_topk", 20),
        retrieval_kwargs=retrieval_kwargs,
        include_user_profile_in_retrieval=config.get("include_user_profile_in_retrieval", False),
        load_lm=load_lm,
    )


def evaluate_tid(tid: str, ground_truth: dict, device: str, batch_size: int) -> float:
    config, music_crs = load_pipeline(tid, device)
    if music_crs.reranker is None:
        raise ValueError(f"Config {tid} has no reranker — cannot compute rerank nDCG.")

    db = load_dataset(config.test_dataset_name, split="test")
    ndcg_scores: list[float] = []

    examples = []
    for item in db:
        for target_turn_number in range(1, 9):
            chat_history, user_query = parse_conversation_history(
                item["conversations"],
                music_crs,
                target_turn_number,
            )
            key = (item["session_id"], target_turn_number)
            gt_id = ground_truth.get(key)
            if gt_id is None:
                continue
            examples.append(
                {
                    "session_id": item["session_id"],
                    "user_id": item["user_id"],
                    "turn_number": target_turn_number,
                    "user_query": user_query,
                    "session_memory": chat_history,
                    "ground_truth_track_id": gt_id,
                }
            )

    for start in tqdm(range(0, len(examples), batch_size), desc=f"Eval {tid}"):
        batch = examples[start : start + batch_size]
        queries = []
        dialogue_queries = []
        session_memories = []
        for row in batch:
            session_memory = row["session_memory"].copy()
            session_memory.append({"role": "user", "content": row["user_query"]})
            queries.append(music_crs._build_retrieval_input(session_memory, user_id=row["user_id"]))
            dialogue_queries.append(music_crs._build_dialogue_input(session_memory))
            session_memories.append(session_memory)

        if hasattr(music_crs.retrieval, "batch_text_to_item_retrieval"):
            batch_candidates = music_crs.retrieval.batch_text_to_item_retrieval(
                queries,
                topk=music_crs.retrieve_topk,
            )
        else:
            batch_candidates = [
                music_crs.retrieval.text_to_item_retrieval(query, topk=music_crs.retrieve_topk)
                for query in queries
            ]

        reranked = music_crs.reranker.batch_rerank(
            batch_candidates,
            [row["user_id"] for row in batch],
            topk=music_crs.final_topk,
            queries=queries,
            dialogue_queries=dialogue_queries,
            item_db=music_crs.item_db,
            retrieval=music_crs.retrieval,
            session_memories=session_memories,
            user_db=music_crs.user_db,
        )

        for row, preds in zip(batch, reranked):
            rel = [int(track_id == row["ground_truth_track_id"]) for track_id in preds]
            ndcg_scores.append(ndcg_at_k(rel, music_crs.final_topk))

        if device == "cuda":
            torch.cuda.empty_cache()

    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0


def main(args: argparse.Namespace) -> None:
    device = resolve_device("cuda")
    gt_path = args.ground_truth
    if not os.path.exists(gt_path):
        raise FileNotFoundError(gt_path)
    ground_truth = load_ground_truth(gt_path)

    print("=== Rerank-only devset nDCG@20 (retrieval → LambdaMART → top-20) ===")
    baseline_ndcg = evaluate_tid(args.baseline_tid, ground_truth, device, args.batch_size)
    candidate_ndcg = evaluate_tid(args.candidate_tid, ground_truth, device, args.batch_size)

    print("\n=== Side-by-side ===")
    print(f"{'Pipeline':<50} {'nDCG@20':>10}")
    print("-" * 62)
    print(f"{args.baseline_tid:<50} {baseline_ndcg:>10.4f}")
    print(f"{args.candidate_tid:<50} {candidate_ndcg:>10.4f}")
    delta = candidate_ndcg - baseline_ndcg
    print(f"{'Delta (candidate - baseline)':<50} {delta:>+10.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rerank-only devset nDCG@20 comparison.")
    parser.add_argument(
        "--baseline-tid",
        type=str,
        default="llama1b_v5_lambdamart_devset",
        help="Legacy dense-200 + v7 LambdaMART config (devset yaml).",
    )
    parser.add_argument(
        "--candidate-tid",
        type=str,
        default="llama1b_hybrid_lambdamart_devset",
        help="Hybrid pool + retrained LambdaMART config.",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default="../music-crs-evaluator/exp/ground_truth/devset.json",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    main(parser.parse_args())
