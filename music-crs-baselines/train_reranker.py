"""
Train LightGBM reranker from build_rerank_features.py output.

Usage:
  python train_reranker.py
"""

import argparse
import json
import os

import lightgbm as lgb
import numpy as np

from mcrs.reranker_modules.rerank_features import FEATURE_COLUMNS


def load_features(path: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rows = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            rows.append(json.loads(line))
    x = np.array([[row[name] for name in FEATURE_COLUMNS] for row in rows], dtype=np.float32)
    y = np.array([row["label"] for row in rows], dtype=np.float32)
    return x, y, rows


def main(args):
    features_path = args.features
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features not found: {features_path}. Run build_rerank_features.py first.")

    x, y, rows = load_features(features_path)
    session_keys = np.array([f"{row['session_id']}:{row['turn_number']}" for row in rows])

    unique_sessions = sorted(set(row["session_id"] for row in rows))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(unique_sessions)
    split_at = int(len(unique_sessions) * (1.0 - args.val_fraction))
    train_sessions = set(unique_sessions[:split_at])

    train_mask = np.array([row["session_id"] in train_sessions for row in rows])
    val_mask = ~train_mask

    train_data = lgb.Dataset(x[train_mask], label=y[train_mask], feature_name=FEATURE_COLUMNS)
    val_data = lgb.Dataset(x[val_mask], label=y[val_mask], feature_name=FEATURE_COLUMNS, reference=train_data)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=args.num_boost_round,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    model.save_model(args.output)

    val_pred = model.predict(x[val_mask])
    pos_rate = y[val_mask].mean()
    print(f"Saved model to {args.output}")
    print(f"Val positives: {pos_rate:.4f}, rows: {val_mask.sum()}")
    print(f"Val pred mean: {val_pred.mean():.4f}, max: {val_pred.max():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LightGBM reranker.")
    parser.add_argument(
        "--features",
        type=str,
        default="./cache/reranker/train_features.jsonl",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./cache/reranker/lgbm_model.txt",
    )
    parser.add_argument("--num-boost-round", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
