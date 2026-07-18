"""
Build LambdaMART training features from devset dense retrieval pools.

Usage:
  python build_lambdamart_features.py --tid llama1b_lambdamart_devset
"""

import argparse
import json
import os

import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm

from mcrs import load_crs_baseline
from mcrs.reranker_modules.rank_features import (
    LAMBDAMART_FEATURE_COLUMNS,
    build_lambdamart_features,
    extract_recent_dialogue,
)
from mcrs.reranker_modules.rerank_features import load_popularity_scores
from mcrs.utils.conversation import parse_conversation_history
from run_inference_blindset import optional_config_dict, resolve_device


def _dense_retriever(retrieval):
    if hasattr(retrieval, "dense_retriever"):
        return retrieval.dense_retriever
    return retrieval


def load_ground_truth(path: str) -> dict[tuple[str, int], str]:
    rows = json.load(open(path, "r", encoding="utf-8"))
    return {(row["session_id"], row["turn_number"]): row["ground_truth_track_id"] for row in rows}


def main(args):
    config = OmegaConf.load(f"config/{args.tid}.yaml")
    device = resolve_device(config.device)
    retrieval_kwargs = optional_config_dict(config, "retrieval_kwargs")

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
        retrieve_topk=config.get("retrieve_topk", 200),
        final_topk=config.get("retrieve_topk", 200),
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
    groups = []

    for item in tqdm(db, desc="Building LambdaMART features"):
        for target_turn_number in range(1, 9):
            chat_history, user_query = parse_conversation_history(
                item["conversations"],
                music_crs,
                target_turn_number,
            )
            session_memory = chat_history + [{"role": "user", "content": user_query}]
            full_query = music_crs._build_retrieval_input(session_memory, user_id=item["user_id"])
            dialogue_query = music_crs._build_dialogue_input(session_memory)
            user_profile_str = music_crs.user_db.id_to_profile_str(item["user_id"])
            recent_dialogue = extract_recent_dialogue(session_memory, num_turns=2)

            candidates = music_crs.retrieval.text_to_item_retrieval(
                full_query,
                topk=config.get("retrieve_topk", 200),
            )
            max_candidates = args.max_candidates or config.get("retrieve_topk", 200)
            positive_id = ground_truth.get((item["session_id"], target_turn_number))

            candidate_features = build_lambdamart_features(
                candidates=candidates[:max_candidates],
                full_query=full_query,
                dialogue_query=dialogue_query,
                dense_retriever=_dense_retriever(music_crs.retrieval),
                item_db=music_crs.item_db,
                popularity_scores=popularity_scores,
                user_profile_str=user_profile_str,
                recent_dialogue=recent_dialogue,
                retrieval=music_crs.retrieval,
                pool_topk=config.get("retrieve_topk", 200),
            )

            group_key = f"{item['session_id']}:{target_turn_number}"
            groups.append(len(candidate_features))

            for row in candidate_features:
                row["label"] = int(row["track_id"] == positive_id)
                row["group_key"] = group_key
                row["session_id"] = item["session_id"]
                row["turn_number"] = target_turn_number
                feature_rows.append(row)

            if device == "cuda":
                torch.cuda.empty_cache()

    os.makedirs(os.path.join(config.cache_dir, "reranker"), exist_ok=True)
    output_path = args.output or os.path.join(config.cache_dir, "reranker", "lambdamart_features.jsonl")
    with open(output_path, "w", encoding="utf-8") as file:
        for row in feature_rows:
            file.write(json.dumps(row) + "\n")

    meta_path = output_path.replace(".jsonl", "_groups.json")
    with open(meta_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "feature_columns": LAMBDAMART_FEATURE_COLUMNS,
                "num_groups": len(groups),
                "num_rows": len(feature_rows),
                "groups": groups,
            },
            file,
            indent=2,
        )

    positives = sum(row["label"] for row in feature_rows)
    print(f"Wrote {len(feature_rows)} rows ({positives} positives) to {output_path}")
    print(f"Groups: {len(groups)}, features: {LAMBDAMART_FEATURE_COLUMNS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build LambdaMART training features.")
    parser.add_argument("--tid", type=str, default="llama1b_lambdamart_devset")
    parser.add_argument(
        "--ground-truth",
        type=str,
        default="../music-crs-evaluator/exp/ground_truth/devset.json",
    )
    parser.add_argument("--max-candidates", type=int, default=None, help="Cap candidates per group (default: full retrieve_topk).")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output jsonl path (default: cache/reranker/lambdamart_features.jsonl).",
    )
    args = parser.parse_args()
    main(args)
