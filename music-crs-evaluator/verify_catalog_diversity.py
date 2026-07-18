"""Verify catalog diversity and submission quality on a prediction file."""

import argparse
import json
from collections import Counter
from pathlib import Path


def load_predictions(path: str) -> list[dict]:
    return json.load(open(path, encoding="utf-8"))


def check_turn_uniqueness(rows: list[dict]) -> int:
    duplicate_turns = 0
    for row in rows:
        ids = row["predicted_track_ids"]
        if len(ids) != len(set(ids)):
            duplicate_turns += 1
    return duplicate_turns


def check_session_repeats(rows: list[dict]) -> list[tuple[str, int]]:
    by_session: dict[str, list[str]] = {}
    for row in rows:
        by_session.setdefault(row["session_id"], []).extend(row["predicted_track_ids"])
    repeats = []
    for session_id, ids in by_session.items():
        counts = Counter(ids)
        repeat_count = sum(count - 1 for count in counts.values() if count > 1)
        if repeat_count:
            repeats.append((session_id, repeat_count))
    repeats.sort(key=lambda item: item[1], reverse=True)
    return repeats


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify catalog diversity computation")
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--catalog-size", type=int, default=47071)
    parser.add_argument("--top-repeats", type=int, default=15)
    args = parser.parse_args()

    path = Path(args.predictions)
    rows = load_predictions(str(path))
    all_ids = []
    per_turn_unique = []
    for row in rows:
        ids = row["predicted_track_ids"]
        all_ids.extend(ids)
        per_turn_unique.append(len(set(ids)))

    unique = len(set(all_ids))
    slots = len(all_ids)
    diversity = unique / args.catalog_size
    theoretical_max = min(slots, args.catalog_size) / args.catalog_size
    global_counts = Counter(all_ids)
    duplicate_turns = check_turn_uniqueness(rows)
    session_repeats = check_session_repeats(rows)

    print("=== Catalog diversity check ===")
    print(f"predictions file: {path}")
    print(f"turns: {len(rows)}")
    print(f"recommendation slots (turns x 20): {slots}")
    print(f"unique tracks recommended: {unique}")
    print(f"catalog size (denominator): {args.catalog_size}")
    print(f"catalog_diversity = unique / catalog = {diversity:.6f}")
    print(f"theoretical max for this submission: {theoretical_max:.6f}")
    print(f"avg unique tracks per turn: {sum(per_turn_unique) / len(per_turn_unique):.2f}")
    print(f"turns with duplicate track IDs in top-20: {duplicate_turns}")
    print(f"sessions with cross-turn repeats: {len(session_repeats)}")
    print()
    print(f"Top {args.top_repeats} globally repeated tracks:")
    for track_id, count in global_counts.most_common(args.top_repeats):
        print(f"  {track_id}: {count} slots")
    if session_repeats:
        print()
        print("Sessions with most cross-turn repeats (first 5):")
        for session_id, repeat_count in session_repeats[:5]:
            print(f"  {session_id}: {repeat_count} repeated slots")
    print()
    print("Note: with 80 turns x 20 tracks, diversity cannot exceed ~1600/47071 ≈ 0.034")
    print("even if every slot is a different track. A leaderboard value near 1.0 would")
    print("use a different definition or display scale — not this evaluator formula.")


if __name__ == "__main__":
    main()
