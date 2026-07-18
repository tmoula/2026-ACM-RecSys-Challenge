"""
Build a pool of diverse assistant-style examples for dynamic few-shot prompting.

Usage:
  python build_fewshot_pool.py
"""

import argparse
import json
import os
import random

from datasets import load_dataset
from tqdm import tqdm


def track_metadata_string(track: dict) -> str:
    parts = []
    for field in ["track_name", "artist_name", "album_name", "tag_list"]:
        value = track.get(field, "")
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value)
        parts.append(f"{field}: {value}")
    return ", ".join(parts)


def synthesize_curator_response(user_request: str, track: dict) -> str:
    track_name = ", ".join(track.get("track_name", ["Unknown"]))
    artist = ", ".join(track.get("artist_name", ["Unknown"]))
    tags = ", ".join(track.get("tag_list", [])[:3]) or "its own distinct mood"
    return (
        f"Given your ask about \"{user_request.strip()}\", I'd start with \"{track_name}\" by {artist}. "
        f"It carries {tags}, which lines up with the energy you described without feeling generic. "
        f"If you want to stay in that lane, we can push deeper into similar artists next."
    )


def main(args):
    conversations = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="train")
    tracks = {
        row["track_id"]: row
        for row in load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks")
    }

    pool = []
    for session in tqdm(conversations, desc="Building few-shot pool"):
        history = []
        for turn in session["conversations"]:
            if turn["role"] == "user":
                user_request = turn["content"]
            elif turn["role"] == "music":
                track = tracks.get(turn["content"])
                if track is None or not history:
                    continue
                latest_user = next(
                    (item["content"] for item in reversed(history) if item["role"] == "user"),
                    "",
                )
                if not latest_user.strip():
                    continue
                pool.append(
                    {
                        "user_context": "\n".join(
                            f"{item['role']}: {item['content']}" for item in history[-4:]
                        ),
                        "user_request": latest_user,
                        "track_metadata": track_metadata_string(track),
                        "assistant_response": synthesize_curator_response(latest_user, track),
                    }
                )
            history.append({"role": turn["role"], "content": turn["content"]})

    random.seed(args.seed)
    random.shuffle(pool)
    pool = pool[: args.max_examples]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(pool, file, ensure_ascii=False, indent=2)
    print(f"Wrote {len(pool)} few-shot examples to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build dynamic few-shot pool.")
    parser.add_argument(
        "--output",
        type=str,
        default="./cache/generation/fewshot_pool.json",
    )
    parser.add_argument("--max-examples", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
