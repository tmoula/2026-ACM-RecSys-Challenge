#!/usr/bin/env python3
"""Run Blind A v2 inference locally (Cursor / terminal). No Colab."""

from __future__ import annotations

import json
import os
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import torch
from huggingface_hub import hf_hub_download, login
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "music-crs-baselines"
CONFIG_TID = os.environ.get("CONFIG_TID", "llama1b_bm25_rerank_blindset_A")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))


def load_hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    env_file = ROOT / ".env"
    if not token and env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("HF_TOKEN="):
                token = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
    if not token or token == "hf_your_token_here":
        raise SystemExit(
            "Missing HF_TOKEN. Create RecSys Competition/.env with:\n"
            "  HF_TOKEN=hf_...\n"
            "(copy from .env.example)"
        )
    return token


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    reranker = BASE / "mcrs/reranker_modules/embedding_reranker.py"
    if not reranker.exists():
        raise SystemExit(f"v2 code missing: {reranker}")

    os.chdir(BASE)
    sys.path.insert(0, str(BASE))

    token = load_hf_token()
    login(token=token)
    os.environ["HF_TOKEN"] = token
    hf_hub_download("meta-llama/Llama-3.2-1B-Instruct", "config.json", token=token)
    print("Llama access OK")

    device = pick_device()
    print(f"Device: {device} (batch_size={BATCH_SIZE})")
    if device == "cpu":
        print("WARNING: CPU is very slow. Prefer Colab GPU if this takes too long.")

    cfg_path = BASE / "config" / f"{CONFIG_TID}.yaml"
    cfg = OmegaConf.load(cfg_path)
    cfg.device = device
    cfg.attn_implementation = "eager"
    OmegaConf.save(cfg, cfg_path)

    from run_inference_blindset import main as run_inference

    print(f"Starting inference: {CONFIG_TID}")
    run_inference(
        SimpleNamespace(
            tid=CONFIG_TID,
            eval_dataset="blindset_A",
            batch_size=BATCH_SIZE,
            save_path="./exp/inference",
        )
    )

    out = BASE / "exp/inference/blindset_A" / f"{CONFIG_TID}.json"
    data = json.loads(out.read_text())
    print(f"Predictions: {len(data)}")

    zip_path = ROOT / f"{CONFIG_TID}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(out, arcname="prediction.json")
    print(f"Submit to Codabench: {zip_path}")


if __name__ == "__main__":
    main()
