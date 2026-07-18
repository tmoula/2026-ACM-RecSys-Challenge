"""
Batch inference script for Music CRS blind sets.
"""

import argparse
import json
import os
from collections import defaultdict

import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm

from mcrs import load_crs_baseline
from mcrs.utils.conversation import parse_conversation_history
from mcrs.utils.diversity import select_diverse_topk


def resolve_device(requested: str) -> str:
    """Use the requested device when available, otherwise pick the best fallback."""
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    if requested == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def optional_config_dict(config, key: str) -> dict | None:
    """Convert an optional config section to a plain dict."""
    value = config.get(key)
    if value is None:
        return None
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True) or None
    if isinstance(value, dict):
        return value or None
    return None


def build_turn_examples(db, music_crs):
    """Expand dataset rows into one example per conversation turn."""
    examples = []
    for item in db:
        turn_numbers = sorted({turn["turn_number"] for turn in item["conversations"]})
        for target_turn_number in turn_numbers:
            chat_history, user_query = parse_conversation_history(
                item["conversations"],
                music_crs,
                target_turn_number,
            )
            examples.append(
                {
                    "user_query": user_query,
                    "user_id": item["user_id"],
                    "session_memory": chat_history,
                    "session_id": item["session_id"],
                    "turn_number": target_turn_number,
                }
            )
    examples.sort(key=lambda row: (row["session_id"], row["turn_number"]))
    return examples


def run_sequential_diverse_inference(music_crs, examples, config, device):
    """Sequential inference with global frequency masking and session de-duplication."""
    diversity_cfg = optional_config_dict(config, "diversity_kwargs") or {}
    global_penalty = diversity_cfg.get("global_penalty", 2.0)
    max_global_allowed = diversity_cfg.get("max_global_allowed", 3)
    final_topk = config.get("final_topk", 20)

    global_track_counts: dict[str, int] = {}
    session_used_tracks: dict[str, set[str]] = defaultdict(set)
    inference_results = []

    for example in tqdm(examples, desc="Sequential inference (diversity)"):
        session_memory = example["session_memory"].copy()
        session_memory.append({"role": "user", "content": example["user_query"]})
        user_id = example["user_id"]
        session_id = example["session_id"]

        retrieval_input = music_crs._build_retrieval_input(session_memory, user_id=user_id)
        dialogue_input = music_crs._build_dialogue_input(session_memory)
        candidates = music_crs.retrieval.text_to_item_retrieval(
            retrieval_input,
            topk=music_crs.retrieve_topk,
        )

        if music_crs.reranker is not None and hasattr(music_crs.reranker, "rerank_with_scores"):
            scored = music_crs.reranker.rerank_with_scores(
                candidates,
                user_id=user_id,
                query=retrieval_input,
                dialogue_query=dialogue_input,
                item_db=music_crs.item_db,
                retrieval=music_crs.retrieval,
                user_db=music_crs.user_db,
                session_memory=session_memory,
            )
            track_ids = select_diverse_topk(
                scored,
                session_used_tracks[session_id],
                global_track_counts,
                topk=final_topk,
                global_penalty=global_penalty,
                max_global_allowed=max_global_allowed,
            )
        else:
            track_ids = music_crs._postprocess_retrieval(
                candidates,
                user_id=user_id,
                query=retrieval_input,
                dialogue_query=dialogue_input,
                session_memory=session_memory,
            )

        system_prompt = music_crs._get_system_prompt(user_id)
        recommend_item = music_crs._build_generation_context(
            track_ids,
            example["user_query"],
            session_memory,
        )
        response = music_crs.lm.response_generation(
            system_prompt,
            session_memory,
            recommend_item,
            max_new_tokens=music_crs.max_response_tokens,
            generation_mode=music_crs.generation_mode,
        )

        if device == "cuda":
            torch.cuda.empty_cache()

        inference_results.append(
            {
                "session_id": session_id,
                "user_id": user_id,
                "turn_number": example["turn_number"],
                "predicted_track_ids": track_ids,
                "predicted_response": response,
            }
        )

    unique_tracks = len({tid for row in inference_results for tid in row["predicted_track_ids"]})
    print(
        f"Diversity stats: {unique_tracks} unique tracks across "
        f"{len(inference_results)} turns (~{unique_tracks / 47071:.4f} catalog diversity)"
    )
    return inference_results


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
    use_diversity = config.get("global_diversity", False)
    examples = build_turn_examples(db, music_crs)

    if use_diversity:
        if args.batch_size != 1:
            print("Forcing batch_size=1 for sequential global diversity masking.")
        inference_results = run_sequential_diverse_inference(music_crs, examples, config, device)
    else:
        batch_data = [
            {
                "user_query": row["user_query"],
                "user_id": row["user_id"],
                "session_memory": row["session_memory"],
            }
            for row in examples
        ]
        metadata = [
            {
                "session_id": row["session_id"],
                "user_id": row["user_id"],
                "turn_number": row["turn_number"],
            }
            for row in examples
        ]

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

    os.makedirs(f"exp/inference/{args.eval_dataset}", exist_ok=True)
    output_path = f"exp/inference/{args.eval_dataset}/{args.tid}.json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(inference_results, file, ensure_ascii=False)
    print(f"Wrote {len(inference_results)} predictions to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run batch inference on TalkPlayData-2 blind sets for Music CRS evaluation."
    )
    parser.add_argument("--tid", type=str, default="llama1b_bm25_blindset_A")
    parser.add_argument("--eval_dataset", type=str, default="blindset_A")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of queries to process in parallel. Use 1 on Colab if you hit GPU OOM.",
    )
    parser.add_argument("--save_path", type=str, default="./exp/inference")
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Keep retrieval indices between runs (saves time for BERT/BGE/hybrid).",
    )
    main(parser.parse_args())
