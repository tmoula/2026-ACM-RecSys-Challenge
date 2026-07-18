#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="music-crs-baselines/.venv/bin/python3"
if [[ ! -x "$PY" ]]; then
  echo "Missing venv at music-crs-baselines/.venv"
  exit 1
fi
if [[ ! -f .env ]]; then
  echo "Create .env first: cp .env.example .env  then add your HF_TOKEN"
  exit 1
fi
export HF_TOKEN="$(grep '^HF_TOKEN=' .env | cut -d= -f2- | tr -d '\"')"
exec "$PY" scripts/legacy/run_blind_v2_local.py