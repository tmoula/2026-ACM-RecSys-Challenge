# Colab runbook — RecSys Blind Set A

Upload fresh zips from `/Users/tahamoula/Desktop/RecSys Competition/`:

| Zip | Purpose |
|-----|---------|
| **`music-crs-baselines-v8.zip`** | Latest **code** bundle (`recall_at_k.py`, all configs). Filename is legacy — **run v7 configs**, not v8 champion. |
| **`music-crs-evaluator-v8.zip`** | Devset eval + catalog diversity verifier |

Regenerate on Mac: `./pack_colab_zips.sh`

**Current best (Blind A):** **v7 champion** — composite **0.378**, nDCG@20 **0.298**, judge **3.05**, catalog div **0.031**

**v8 blind submit regressed — do not use** `llama1b_champion_v8_blindset_A`. Stick to v7 for inference, LambdaMART restore, and recall@K diagnostics.

---

## Catalog diversity — read this before chasing 1.0

Official formula:

```
catalog_diversity = (# unique tracks recommended) / 47071
```


| Eval set    | Turns × 20 tracks      | Theoretical max diversity |
| ----------- | ---------------------- | ------------------------- |
| **Blind A** | 80 × 20 = 1600 slots   | **1600/47071 ≈ 0.034**    |
| **Devset**  | 8000 × 20 = 160k slots | up to **1.0**             |


Your FT submission **0.032** on Blind A is **~94% of the blind ceiling** — not broken.
Leaderboard entries showing **~1.0** are almost certainly on **devset-scale** eval (or a different display), not Blind A.

Verify any prediction file:

```python
!python /content/music-crs-evaluator/verify_catalog_diversity.py \
  --predictions exp/inference/blindset_A/llama1b_bert_ft_blindset_A.json
```

**Priority:** nDCG + judge (your FT win). MMR config below is optional for a small blind diversity bump.

---

## Cell 1 — Setup (upload zips)

**On Mac:** run `./pack_colab_zips.sh` then upload both zips from  
`/Users/tahamoula/Desktop/RecSys Competition/`:

- `music-crs-baselines-v8.zip` — **code only**; you will run **v7** configs below
- `music-crs-evaluator-v8.zip`

```python
from google.colab import files
import os, zipfile

uploaded = files.upload()
for zip_name in uploaded:
    zip_path = os.path.abspath(zip_name)
    print(f"Extracting {zip_name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall("/content")
    if zip_name.startswith("music-crs-evaluator"):
        !cat /content/music-crs-evaluator/VERSION.txt
    if zip_name.startswith("music-crs-baselines"):
        !cat /content/music-crs-baselines/VERSION.txt

%cd /content/music-crs-baselines
!pip install -q -e .
!pip install -q lightgbm openai
```

## Cell 2 — Hugging Face login (required for Llama)

```python
from google.colab import userdata
from huggingface_hub import login
import os

token = userdata.get("HF_TOKEN")  # or login(token="hf_...")
login(token=token)
os.environ["HF_TOKEN"] = token
```

Accept the license at [https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) first.

## Cell 3 — Mount Drive + restore checkpoints (skip if still in session)

```python
from google.colab import drive
import os, shutil

drive.mount("/content/drive")
%cd /content/music-crs-baselines

for name in ["bge-talkplay-v5-bs32", "bge-talkplay-v4"]:
    drive_path = f"/content/drive/MyDrive/recsys-checkpoints/{name}"
    local_path = f"./checkpoints/{name}"
    if os.path.isdir(drive_path) and not os.path.isdir(local_path):
        shutil.copytree(drive_path, local_path)
        print(f"Restored {name} from Drive")
    elif os.path.isdir(local_path):
        print(f"{name} already local")
    else:
        print(f"WARNING: {name} not found — train v5 below or copy to Drive")
```

---

## Train v5 — batch 32 + validation loss (recommended next)

Uses official **dev split** (`split=test`, 1K sessions) for val — session-level, no leakage.

```python
%cd /content/music-crs-baselines

!python train_dense_retriever.py \
  --model-name BAAI/bge-small-en-v1.5 \
  --output-dir ./checkpoints/bge-talkplay-v5-bs32 \
  --epochs 2 \
  --batch-size 32 \
  --max-length 512 \
  --train-split train \
  --val-split test
```

After training, check `checkpoints/bge-talkplay-v5-bs32/training_metrics.json`:

- Compare **epoch-2 val_avg_loss** vs v4 (no val logged, but train was ~0.608)
- If val still dropping → rerun with `--epochs 3`

Backup:

```python
!cp -r ./checkpoints/bge-talkplay-v5-bs32 \
  "/content/drive/MyDrive/recsys-checkpoints/"
```

---

## Dense recall@K diagnostic (pre-rerank — run on Colab)

Measures whether the **fine-tuned dense retriever** puts the ground-truth track in the top-K pool **before** LambdaMART. Use this to decide if you need a larger pool, better FT, or reranker tuning.

**Requires:** Cell 1–3 (baselines zip + HF login + v5 checkpoint on Drive).

This test is **version-agnostic** — same dense encoder for v7 and v8. It does not run LambdaMART or GPT.

Compare pool sizes and hybrid RRF. **Check script version first** (must print `2026-06-22-pool`):

```python
!grep RECALL_AT_K_VERSION /content/music-crs-baselines/recall_at_k.py || echo "STALE — re-upload recall_at_k.py or fresh zip"
```

**Fast fix (upload one file from Mac):**

```python
from google.colab import files
%cd /content/music-crs-baselines
uploaded = files.upload()  # pick recall_at_k.py from music-crs-baselines/ on your Mac
!grep RECALL_AT_K_VERSION recall_at_k.py
```

**Or re-upload** `music-crs-baselines-v8.zip` (run `./pack_colab_zips.sh` on Mac), extract, then `!cat VERSION.txt` should show `baselines-colab-2026-06-22-pool`.

```python
%cd /content/music-crs-baselines

# Baseline dense pool 200 (your original run: recall@200 ≈ 0.60)
!python recall_at_k.py --tid llama1b_lambdamart_devset --batch-size 16

# Wider dense pool 1000 — no extra yaml needed
!python recall_at_k.py --tid llama1b_lambdamart_devset --retrieve-topk 1000 --batch-size 16

# BM25 + dense RRF pool 1000 — no extra yaml needed
!python recall_at_k.py --tid llama1b_lambdamart_devset --retrieve-topk 1000 --hybrid --batch-size 16

# Or use dedicated yamls after re-uploading fresh zip:
# !python recall_at_k.py --tid llama1b_lambdamart_pool1000_devset --batch-size 16
# !python recall_at_k.py --tid llama1b_lambdamart_hybrid_pool1000_devset --batch-size 16
```

### Hybrid LambdaMART retrain (do before hybrid blind submit)

```python
%cd /content/music-crs-baselines

!python build_lambdamart_features.py \
  --tid llama1b_lambdamart_hybrid_pool1000_devset \
  --output ./cache/reranker/lambdamart_hybrid_features.jsonl

!python train_lambdamart.py \
  --features ./cache/reranker/lambdamart_hybrid_features.jsonl \
  --output ./cache/reranker/lambdamart_hybrid_model.txt \
  --num-boost-round 300 --log-period 10

!python evaluate_rerank_ndcg_devset.py \
  --baseline-tid llama1b_v5_lambdamart_devset \
  --candidate-tid llama1b_hybrid_lambdamart_devset \
  --batch-size 4
```

Fallback v7 blind: `llama1b_champion_lambdamart_gpt_blindset_A` + `lambdamart_model.txt` (unchanged).

Blind submit after rerank devset nDCG improves:

| Config | Pool | Retrieval | LambdaMART model |
|--------|------|-----------|------------------|
| `llama1b_champion_lambdamart_gpt_blindset_A` | 200 | dense (**fallback ~0.38**) | `lambdamart_model.txt` |
| `llama1b_champion_hybrid_pool1000_blindset_A` | 1000 | BM25+dense RRF | `lambdamart_hybrid_model.txt` |

```python
!python run_inference_blindset.py --tid llama1b_champion_hybrid_pool1000_blindset_A --batch_size 1 --keep-cache
```

---

## Devset eval (free — do before next blind submit)

```python
import gc, torch, shutil, importlib
from types import SimpleNamespace
from omegaconf import OmegaConf

%cd /content/music-crs-baselines

CONFIG_TID = "llama1b_bert_ft_devset"
cfg = OmegaConf.load(f"config/{CONFIG_TID}.yaml")
cfg.retrieval_kwargs.model_name = "./checkpoints/bge-talkplay-v5-bs32"  # or v4
OmegaConf.save(cfg, f"config/{CONFIG_TID}.yaml")

shutil.rmtree("cache/dense", ignore_errors=True)
gc.collect(); torch.cuda.empty_cache()

import run_inference_devset
importlib.reload(run_inference_devset)
run_inference_devset.main(SimpleNamespace(
    tid=CONFIG_TID, batch_size=8, save_path="./exp/inference", keep_cache=True,
))
```

Copy predictions to evaluator repo and score (or clone evaluator into Colab):

```python
# Compare FT vs grounded BM25 on devset nDCG@20
```

---

## Blind inference configs


| `CONFIG_TID`                     | What it tests                                   |
| -------------------------------- | ----------------------------------------------- |
| `llama1b_bert_ft_blindset_A`     | **Current best** — FT v4 dense + grounded Llama |
| `llama1b_hybrid_ft_blindset_A`   | BM25 ∪ FT-BGE RRF fusion                        |
| `llama1b_bert_ft_mmr_blindset_A` | FT v4 + MMR diversity rerank (pool 100, λ=0.7)  |


### Inference cell template

```python
import gc, torch, shutil, importlib
from types import SimpleNamespace
from omegaconf import OmegaConf

%cd /content/music-crs-baselines

CONFIG_TID = "llama1b_bert_ft_blindset_A"  # change per table
CKPT = "./checkpoints/bge-talkplay-v5-bs32"  # or v4 / Drive path

cfg = OmegaConf.load(f"config/{CONFIG_TID}.yaml")
cfg.retrieval_kwargs.model_name = CKPT
cfg.device = "cuda"
cfg.attn_implementation = "eager"
OmegaConf.save(cfg, f"config/{CONFIG_TID}.yaml")

shutil.rmtree("cache/dense", ignore_errors=True)
gc.collect(); torch.cuda.empty_cache()

import run_inference_blindset
importlib.reload(run_inference_blindset)
run_inference_blindset.main(SimpleNamespace(
    tid=CONFIG_TID,
    eval_dataset="blindset_A",
    batch_size=8,
    save_path="./exp/inference",
    keep_cache=True,
))
```

### Download submission zip

```python
import json, zipfile
from pathlib import Path
from google.colab import files

out = Path(f"exp/inference/blindset_A/{CONFIG_TID}.json")
zip_path = Path(f"{CONFIG_TID}.zip")
with zipfile.ZipFile(zip_path, "w") as zf:
    zf.write(out, arcname="prediction.json")
files.download(str(zip_path))
```

---

## v7 Champion pipeline (LambdaMART + GPT-4o-mini) — USE THIS

Fullstack/hybrid failed on blind. v7 is the **production** stack (~0.378 composite).

| Metric                 | Solution                                            |
| ---------------------- | --------------------------------------------------- |
| **nDCG@20**            | v5 dense pool 200 → **LambdaMART** rerank → top 20  |
| **LLM judge + LexDiv** | **GPT-4o-mini** + curator prompt + dynamic few-shot |

Set OpenAI key (Cell 2 already has HF):

```python
import os
os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")  # Colab secret
```

### Restore v7 LambdaMART from Drive (skip retrain if backup exists)

```python
import shutil, os
BACKUP = "/content/drive/MyDrive/recsys-checkpoints/lambdamart-v7"
os.makedirs("cache/reranker", exist_ok=True)
os.makedirs("cache/generation", exist_ok=True)
for name in ["lambdamart_model.txt", "devset_ground_truth.json", "lambdamart_groups.json"]:
    shutil.copy2(f"{BACKUP}/{name}", f"cache/reranker/{name}")
shutil.copy2(f"{BACKUP}/fewshot_pool.json", "cache/generation/fewshot_pool.json")
!ls -lh cache/reranker/lambdamart_model.txt
```

The saved v7 model has **7 features**; the reranker auto-detects that (ignore the 2 extra v8 feature columns in code).

### Train LambdaMART from scratch (only if no Drive backup)

```python
%cd /content/music-crs-baselines
!python build_fewshot_pool.py
!python build_lambdamart_features.py --tid llama1b_lambdamart_devset --max-candidates 200
!python train_lambdamart.py --num-boost-round 300
```

Backup after train:

```python
BACKUP = "/content/drive/MyDrive/recsys-checkpoints/lambdamart-v7"
os.makedirs(BACKUP, exist_ok=True)
for name in ["lambdamart_model.txt", "devset_ground_truth.json", "lambdamart_groups.json"]:
    shutil.copy2(f"cache/reranker/{name}", f"{BACKUP}/{name}")
shutil.copy2("cache/generation/fewshot_pool.json", f"{BACKUP}/fewshot_pool.json")
```

### Devset sanity check (free, Llama — no OpenAI cost)

```python
!python run_inference_devset.py --tid llama1b_v5_lambdamart_blindset_A --batch_size 4 --keep-cache
```

### Blind submit (v7 champion)

```python
!python run_inference_blindset.py \
  --tid llama1b_champion_lambdamart_gpt_blindset_A \
  --batch_size 1 --keep-cache
```

Configs:

- `llama1b_v5_lambdamart_blindset_A` — nDCG-only test (Llama responses)
- `llama1b_champion_lambdamart_gpt_blindset_A` — **submit this** (LambdaMART + GPT-4o-mini)

---

## v8 Champion pipeline — DO NOT USE (blind regression)

v8 added global diversity masking, 2 extra LambdaMART features, and aggressive GPT sampling. **Blind scores dropped** vs v7. Kept for reference only.

<details>
<summary>v8 cells (collapsed)</summary>

| Metric | v8 change |
|--------|-----------|
| **Catalog diversity** | Global frequency masking (`max_global_allowed: 3`, penalty 2.0) |
| **nDCG@20** | 9-feature LambdaMART + 500 boost rounds |
| **LLM judge** | Anti-hallucination prompt |
| **Lexical diversity** | Higher temperature / penalties, 3 few-shot |

```python
!python build_lambdamart_features.py --tid llama1b_lambdamart_devset --max-candidates 200
!python train_lambdamart.py --num-boost-round 500
!python run_inference_blindset.py --tid llama1b_champion_v8_blindset_A --batch_size 1 --keep-cache
```

Config: `llama1b_champion_v8_blindset_A`

</details>

---

## Suggested submission order (remaining budget)

1. **`llama1b_champion_lambdamart_gpt_blindset_A`** — v7 champion (~0.378)
2. **`llama1b_bert_ft_blindset_A`** — v5 dense fallback (~0.33)
3. Do **not** resubmit v8

Do **not** sacrifice nDCG for catalog diversity on Blind A — the metric is structurally capped low.

---

## Score history


| Config         | nDCG@20   | Judge    | Composite |
| -------------- | --------- | -------- | --------- |
| BM25 grounded  | 0.177     | 1.55     | 0.200     |
| **FT v4 bs16** | **0.228** | **2.55** | **0.302** |


---

## Local Mac (optional)

Checkpoints and CUDA are on Colab — prefer the cells above. On Mac you need the FT checkpoint copied from Drive and `.venv/bin/python3` (not conda `python`).

```bash
cd music-crs-baselines
.venv/bin/python3 recall_at_k.py --tid llama1b_lambdamart_devset --checkpoint ./checkpoints/bge-talkplay-v5-bs32
.venv/bin/python3 run_inference_devset.py --tid llama1b_bert_ft_devset --batch_size 8 --keep-cache
```

