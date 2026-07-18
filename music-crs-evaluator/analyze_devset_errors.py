"""
Error analysis for devset predictions.

Classifies failures as retrieval (GT not in top-k) vs ranking (GT in list but low rank).
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from metrics.metrics_recsys import get_hit, get_ndcg


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze devset retrieval failures")
    parser.add_argument("--tid", type=str, required=True, help="Prediction file stem")
    parser.add_argument(
        "--ground-truth",
        type=str,
        default="exp/ground_truth/devset.json",
        help="Path to devset ground truth JSON",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default=None,
        help="Path to predictions JSON (default: exp/inference/devset/<tid>.json)",
    )
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--sample-worst", type=int, default=20)
    args = parser.parse_args()

    predictions_path = Path(args.predictions or f"exp/inference/devset/{args.tid}.json")
    ground_truth = pd.DataFrame(json.load(open(args.ground_truth)))
    predictions = pd.DataFrame(json.load(open(predictions_path)))

    merged = ground_truth.merge(
        predictions,
        on=["session_id", "turn_number"],
        suffixes=("_gt", "_pred"),
    )

    rows = []
    for _, row in merged.iterrows():
        gold = [row["ground_truth_track_id"]]
        preds = row["predicted_track_ids"]
        hit = get_hit(gold, preds, args.topk)
        rank = preds.index(gold[0]) + 1 if gold[0] in preds else None
        rows.append(
            {
                "session_id": row["session_id"],
                "turn_number": row["turn_number"],
                "hit_at_k": hit,
                "gt_rank": rank,
                "ndcg_at_k": get_ndcg(gold, preds, args.topk),
                "failure_type": (
                    "hit"
                    if hit
                    else "retrieval_miss"
                ),
                "ranking_issue": bool(hit and rank and rank > 1),
            }
        )

    results = pd.DataFrame(rows)
    summary = {
        "tid": args.tid,
        "examples": len(results),
        f"hit@{args.topk}": results["hit_at_k"].mean(),
        f"ndcg@{args.topk}": results["ndcg_at_k"].mean(),
        "retrieval_miss_rate": (results["failure_type"] == "retrieval_miss").mean(),
        "ranking_issue_rate": results["ranking_issue"].mean(),
    }

    print("=== Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    worst = results.sort_values("ndcg_at_k").head(args.sample_worst)
    print("\n=== Worst sessions (lowest nDCG) ===")
    print(
        worst.merge(merged[["session_id", "turn_number", "ground_truth_track_id"]], on=["session_id", "turn_number"])[
            ["session_id", "turn_number", "ground_truth_track_id", "gt_rank", "ndcg_at_k", "failure_type"]
        ].to_string(index=False)
    )

    out_dir = Path("exp/analysis/devset")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.tid}.json"
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump({"summary": summary, "per_example": rows}, file, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
