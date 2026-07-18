# Legacy local runners

These scripts preserve the early v2 local-inference workflow for historical
reference. They are not used by the final v7 pipeline.

- `run_blind_v2_local.py` runs the BM25 + embedding-reranker Blind A experiment.
- `run_local.sh` loads `HF_TOKEN` from the repository-root `.env` and invokes
  the Python runner with the local virtual environment.

For current training and inference, use the commands documented in
[`music-crs-baselines/README.md`](../../music-crs-baselines/README.md).
