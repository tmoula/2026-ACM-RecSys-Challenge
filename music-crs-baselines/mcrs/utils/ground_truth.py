"""Build and load devset ground-truth labels from TalkPlay conversations."""

from __future__ import annotations

import json
import os
from typing import Dict, Tuple

from datasets import load_dataset
from tqdm import tqdm


def _gt_track_for_turn(conversations: list, target_turn_number: int) -> str | None:
    for turn in conversations:
        if turn.get("turn_number") != target_turn_number:
            continue
        if turn.get("role") == "music":
            return turn["content"]
    return None


def build_ground_truth_file(
    dataset_name: str,
    split: str,
    output_path: str,
) -> list[dict]:
    db = load_dataset(dataset_name, split=split)
    rows = []
    for item in tqdm(db, desc="Building ground truth"):
        for target_turn_number in range(1, 9):
            gt_track_id = _gt_track_for_turn(item["conversations"], target_turn_number)
            if not gt_track_id:
                continue
            rows.append(
                {
                    "session_id": item["session_id"],
                    "user_id": item["user_id"],
                    "turn_number": target_turn_number,
                    "ground_truth_track_id": gt_track_id,
                }
            )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2)
    return rows


def load_or_build_ground_truth(
    path: str,
    dataset_name: str = "talkpl-ai/TalkPlayData-Challenge-Dataset",
    split: str = "test",
) -> Dict[Tuple[str, int], str]:
    if not os.path.exists(path):
        print(f"Ground truth not found at {path} — building from {dataset_name} ({split})...")
        build_ground_truth_file(dataset_name, split, path)
    rows = json.load(open(path, "r", encoding="utf-8"))
    return {(row["session_id"], row["turn_number"]): row["ground_truth_track_id"] for row in rows}
