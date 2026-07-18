"""
Batch inference script for Music CRS development set.
"""

import argparse
import json
import os

import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm

from mcrs import load_crs_baseline
from mcrs.utils.conversation import parse_conversation_history
from run_inference_blindset import resolve_device, optional_config_dict


def main(args):
    if not args.keep_cache:
        print("Removing cache directory for preventing memory issues...")
        os.system("rm -rf cache")

    config = OmegaConf.load(f"config/{args.tid}.yaml")
    device = resolve_device(config.device)
    if device != config.device:
        print(f"Device {config.device!r} unavailable, using {device!r}")

    reranker_kwargs = optional_config_dict(config, "reranker_kwargs")
    retrieval_kwargs = optional_config_dict(config, "retrieval_kwargs")
    lm_kwargs = optional_config_dict(config, "lm_kwargs")
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
        reranker_type=config.get("reranker_type"),
        reranker_kwargs=reranker_kwargs,
        retrieve_topk=config.get("retrieve_topk", 20),
        final_topk=config.get("final_topk", 20),
        retrieval_kwargs=retrieval_kwargs,
        include_user_profile_in_retrieval=config.get("include_user_profile_in_retrieval", False),
        response_topk=config.get("response_topk", 1),
        generation_mode=config.get("generation_mode", "default"),
        max_response_tokens=config.get("max_response_tokens", 64),
        lm_kwargs=lm_kwargs,
    )

    db = load_dataset(config.test_dataset_name, split="test")
    batch_data, metadata = [], []
    for item in db:
        for target_turn_number in range(1, 9):
            chat_history, user_query = parse_conversation_history(
                item["conversations"],
                music_crs,
                target_turn_number,
            )
            batch_data.append(
                {
                    "user_query": user_query,
                    "user_id": item["user_id"],
                    "session_memory": chat_history,
                }
            )
            metadata.append(
                {
                    "session_id": item["session_id"],
                    "user_id": item["user_id"],
                    "turn_number": target_turn_number,
                }
            )

    inference_results = []
    for index in tqdm(range(0, len(batch_data), args.batch_size), desc="Batch inference"):
        batch = batch_data[index : index + args.batch_size]
        batch_metadata = metadata[index : index + args.batch_size]
        results = music_crs.batch_chat(batch)
        if device == "cuda":
            torch.cuda.empty_cache()
        for result_index, result in enumerate(results):
            inference_results.append(
                {
                    "session_id": batch_metadata[result_index]["session_id"],
                    "user_id": batch_metadata[result_index]["user_id"],
                    "turn_number": batch_metadata[result_index]["turn_number"],
                    "predicted_track_ids": result["retrieval_items"],
                    "predicted_response": result["response"],
                }
            )

    os.makedirs("exp/inference/devset", exist_ok=True)
    output_path = f"exp/inference/devset/{args.tid}.json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(inference_results, file, ensure_ascii=False)
    print(f"Wrote {len(inference_results)} predictions to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch inference on TalkPlay devset")
    parser.add_argument("--tid", type=str, default="llama1b_bm25_devset")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--save_path", type=str, default="./exp/inference")
    parser.add_argument("--keep-cache", action="store_true")
    main(parser.parse_args())
