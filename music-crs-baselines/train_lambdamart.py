"""
Train LambdaMART ranker for nDCG@20 optimization.

Usage:
  # Hybrid pool (1000 candidates, 12 features):
  python build_lambdamart_features.py --tid llama1b_lambdamart_hybrid_pool1000_devset
  python train_lambdamart.py --features ./cache/reranker/lambdamart_hybrid_features.jsonl \
      --output ./cache/reranker/lambdamart_hybrid_model.txt --num-boost-round 300

  # Legacy v7 dense pool (fallback model untouched at lambdamart_model.txt):
  python train_lambdamart.py --output ./cache/reranker/lambdamart_model.txt
"""

from __future__ import annotations

import argparse
import json
import os

import lightgbm as lgb
import numpy as np

from mcrs.reranker_modules.rank_features import LAMBDAMART_FEATURE_COLUMNS


def load_feature_columns(features_path: str) -> list[str]:
    meta_path = features_path.replace(".jsonl", "_groups.json")
    legacy_path = os.path.join(os.path.dirname(features_path), "lambdamart_groups.json")
    for path in (meta_path, legacy_path):
        if os.path.exists(path):
            meta = json.load(open(path, "r", encoding="utf-8"))
            columns = meta.get("feature_columns")
            if columns:
                return columns
    return list(LAMBDAMART_FEATURE_COLUMNS)


def load_features(
    path: str,
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, list[int], np.ndarray, np.ndarray, list[int]]:
    rows = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            rows.append(json.loads(line))

    groups_map: dict[str, list[dict]] = {}
    for row in rows:
        groups_map.setdefault(row["group_key"], []).append(row)

    session_ids = sorted({row["session_id"] for row in rows})
    rng = np.random.default_rng(42)
    rng.shuffle(session_ids)
    split_at = int(len(session_ids) * 0.8)
    train_sessions = set(session_ids[:split_at])

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    for group_rows in groups_map.values():
        if group_rows[0]["session_id"] in train_sessions:
            train_rows.extend(group_rows)
        else:
            val_rows.extend(group_rows)

    def pack(grouped_rows: list[dict]) -> tuple[np.ndarray, np.ndarray, list[int]]:
        grouped: dict[str, list[dict]] = {}
        for row in grouped_rows:
            grouped.setdefault(row["group_key"], []).append(row)
        x_parts, y_parts, group_sizes = [], [], []
        for group_key in sorted(grouped.keys()):
            group = grouped[group_key]
            x_parts.append([[row[name] for name in feature_columns] for row in group])
            y_parts.append([row["label"] for row in group])
            group_sizes.append(len(group))
        return (
            np.array([value for part in x_parts for value in part], dtype=np.float32),
            np.array([value for part in y_parts for value in part], dtype=np.float32),
            group_sizes,
        )

    x_train, y_train, train_groups = pack(train_rows)
    x_val, y_val, val_groups = pack(val_rows)
    return x_train, y_train, train_groups, x_val, y_val, val_groups


def main(args: argparse.Namespace) -> None:
    features_path = args.features
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features not found: {features_path}. Run build_lambdamart_features.py first.")

    feature_columns = load_feature_columns(features_path)
    print(f"Training with {len(feature_columns)} features: {feature_columns}")

    x_train, y_train, train_groups, x_val, y_val, val_groups = load_features(
        features_path,
        feature_columns,
    )

    train_set = lgb.Dataset(
        x_train,
        label=y_train,
        group=train_groups,
        feature_name=feature_columns,
    )
    val_set = lgb.Dataset(
        x_val,
        label=y_val,
        group=val_groups,
        feature_name=feature_columns,
        reference=train_set,
    )

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [20],
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "verbose": -1,
    }

    evals_result: dict = {}
    model = lgb.train(
        params,
        train_set,
        num_boost_round=args.num_boost_round,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[
            lgb.record_evaluation(evals_result),
            lgb.log_evaluation(period=args.log_period),
        ],
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    model.save_model(args.output)
    print(f"Saved LambdaMART model to {args.output}")
    print(f"Train rows: {len(y_train)}, val rows: {len(y_val)}")

    val_ndcg = evals_result.get("val", {}).get("ndcg@20", [])
    if val_ndcg:
        best_round = int(np.argmax(val_ndcg)) + 1
        print(f"Best val ndcg@20: {max(val_ndcg):.6f} at round {best_round}")
        print(f"Final val ndcg@20: {val_ndcg[-1]:.6f} (round {len(val_ndcg)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LambdaMART ranker.")
    parser.add_argument(
        "--features",
        type=str,
        default="./cache/reranker/lambdamart_features.jsonl",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./cache/reranker/lambdamart_model.txt",
    )
    parser.add_argument("--num-boost-round", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-data-in-leaf", type=int, default=20)
    parser.add_argument("--log-period", type=int, default=10, help="Print val ndcg every N rounds.")
    main(parser.parse_args())
