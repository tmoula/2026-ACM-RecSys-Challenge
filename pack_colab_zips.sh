#!/usr/bin/env bash
# Build Colab upload zips from the RecSys Competition workspace.
# Zips include the top-level folder so extractall("/content") -> /content/music-crs-baselines/
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

BASELINES_ZIP="music-crs-baselines-v8.zip"
EVALUATOR_ZIP="music-crs-evaluator-v8.zip"
BASELINES_ALIAS="UPLOAD-TO-COLAB-baselines.zip"
EVALUATOR_ALIAS="UPLOAD-TO-COLAB-evaluator.zip"

echo "Building ${BASELINES_ZIP} ..."
rm -f "${BASELINES_ZIP}"
zip -r "${BASELINES_ZIP}" music-crs-baselines \
  -x "music-crs-baselines/.venv/*" \
  -x "music-crs-baselines/.venv/**/*" \
  -x "music-crs-baselines/cache/*" \
  -x "music-crs-baselines/cache/**/*" \
  -x "music-crs-baselines/checkpoints/*" \
  -x "music-crs-baselines/checkpoints/**/*" \
  -x "music-crs-baselines/exp/inference/*" \
  -x "music-crs-baselines/exp/inference/**/*" \
  -x "music-crs-baselines/exp/training/*" \
  -x "music-crs-baselines/exp/training/**/*" \
  -x "*__pycache__*" \
  -x "*.pyc" \
  -x "music-crs-baselines/.git/*"

echo "Building ${EVALUATOR_ZIP} ..."
rm -f "${EVALUATOR_ZIP}"
zip -r "${EVALUATOR_ZIP}" music-crs-evaluator \
  -x "music-crs-evaluator/.git/*" \
  -x "music-crs-evaluator/.git/**/*" \
  -x "music-crs-evaluator/exp/inference/*" \
  -x "music-crs-evaluator/exp/inference/**/*" \
  -x "music-crs-evaluator/exp/scores/*" \
  -x "music-crs-evaluator/exp/scores/**/*" \
  -x "*__pycache__*" \
  -x "*.pyc"

echo "Done."
cp "${BASELINES_ZIP}" "${BASELINES_ALIAS}"
cp "${EVALUATOR_ZIP}" "${EVALUATOR_ALIAS}"
ls -lh "${BASELINES_ZIP}" "${EVALUATOR_ZIP}" "${BASELINES_ALIAS}" "${EVALUATOR_ALIAS}"
echo "Baselines zip paths (first 5):"
unzip -l "${BASELINES_ZIP}" | head -8
echo "Baselines version:"
unzip -p "${BASELINES_ZIP}" music-crs-baselines/VERSION.txt
echo "Evaluator version:"
unzip -p "${EVALUATOR_ZIP}" music-crs-evaluator/VERSION.txt
