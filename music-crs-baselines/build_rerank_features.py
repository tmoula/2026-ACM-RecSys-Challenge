"""
Build reranker training features from devset recall pools.

Usage:
  python build_rerank_features.py --tid llama1b_fullstack_devset
"""

import argparse
import json
import os

import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm

from mcrs import load_crs_baseline
from mcrs.reranker_modules.rerank_features import FEATURE_COLUMNS, build_candidate_features, load_popularity_scores
from mcrs.utils.conversation import parse_conversation_history
from run_inference_blindset import optional_config_dict, resolve_device


def load_ground_truth(path: str) -> dict[tuple[str, int], str]:
    rows = json.load(open(path, "r", encoding="utf-8"))
    return {(row["session_id"], row["turn_number"]): row["ground_truth_track_id"] for row in rows}


def main(args):
    config = OmegaConf.load(f"config/{args.tid}.yaml")
    device = resolve_device(config.device)
    reranker_kwargs = optional_config_dict(config, "reranker_kwargs") or {}
    retrieval_kwargs = optional_config_dict(config, "retrieval_kwargs")
    cross_encoder = None
    if args.with_ce:
        from mcrs.reranker_modules.cross_encoder_reranker import CrossEncoderReranker

        cross_encoder = CrossEncoderReranker(
            model_name=reranker_kwargs.get("cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            device=device,
            batch_size=reranker_kwargs.get("batch_size", 32),
            ce_topk=reranker_kwargs.get("ce_topk", 200),
        )

    music_crs = load_crs_baseline(
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
        retrieve_topk=config.get("retrieve_topk", 500),
        final_topk=config.get("retrieve_topk", 500),
        retrieval_kwargs=retrieval_kwargs,
        include_user_profile_in_retrieval=config.get("include_user_profile_in_retrieval", False),
    )

    gt_path = args.ground_truth
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth not found: {gt_path}")
    ground_truth = load_ground_truth(gt_path)
    popularity_scores = load_popularity_scores(config.cache_dir)

    db = load_dataset(config.test_dataset_name, split="test")
    feature_rows = []

    for item in tqdm(db, desc="Building rerank features"):
        for target_turn_number in range(1, 9):
            chat_history, user_query = parse_conversation_history(
                item["conversations"],
                music_crs,
                target_turn_number,
            )
            session_memory = chat_history + [{"role": "user", "content": user_query}]
            full_query = music_crs._build_retrieval_input(session_memory, user_id=item["user_id"])
            dialogue_query = music_crs._build_dialogue_input(session_memory)

            candidates = music_crs.retrieval.retrieve(
                full_query=full_query,
                dialogue_query=dialogue_query,
                user_id=item["user_id"],
                topk=config.get("retrieve_topk", 500),
            )
            meta = music_crs.retrieval.last_batch_meta[0] if music_crs.retrieval.last_batch_meta else {}

            ce_scores = {}
            if cross_encoder is not None:
                ce_scores = cross_encoder._score_pairs(
                    full_query,
                    candidates[: cross_encoder.ce_topk],
                    music_crs.item_db,
                )

            positive_id = ground_truth.get((item["session_id"], target_turn_number))
            candidate_features = build_candidate_features(
                candidates=candidates[: args.max_candidates],
                full_query=full_query,
                dialogue_query=dialogue_query,
                user_id=item["user_id"],
                retrieval_meta=meta,
                dense_retriever=music_crs.retrieval.dense_retriever,
                cf_retriever=music_crs.retrieval.cf_retriever,
                popularity_scores=popularity_scores,
            )
            for row in candidate_features:
                row["ce_score"] = ce_scores.get(row["track_id"], 0.0)
                row["label"] = int(row["track_id"] == positive_id)
                row["session_id"] = item["session_id"]
                row["turn_number"] = target_turn_number
                feature_rows.append(row)

            if device == "cuda":
                torch.cuda.empty_cache()

    os.makedirs(os.path.join(config.cache_dir, "reranker"), exist_ok=True)
    output_path = os.path.join(config.cache_dir, "reranker", "train_features.jsonl")
    with open(output_path, "w", encoding="utf-8") as file:
        for row in feature_rows:
            file.write(json.dumps(row) + "\n")

    print(f"Wrote {len(feature_rows)} rows to {output_path}")
    print(f"Feature columns: {FEATURE_COLUMNS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build LightGBM reranker training features.")
    parser.add_argument("--tid", type=str, default="llama1b_fullstack_devset")
    parser.add_argument(
        "--ground-truth",
        type=str,
        default="../music-crs-evaluator/exp/ground_truth/devset.json",
    )
    parser.add_argument("--max-candidates", type=int, default=200, help="Cap candidates per turn for CE speed.")
    parser.add_argument("--with-ce", action="store_true", help="Compute cross-encoder scores (slow).")
    main(parser.parse_args())
