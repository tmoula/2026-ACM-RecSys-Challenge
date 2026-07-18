#!/usr/bin/env bash
# Build Colab upload zips from the repository root.
# Zips include the project folders so extractall("/content") creates the
# expected /content/music-crs-baselines and /content/music-crs-evaluator paths.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUTPUT_DIR="$ROOT/artifacts/colab"
mkdir -p "$OUTPUT_DIR"

BASELINES_ZIP="$OUTPUT_DIR/music-crs-baselines.zip"
EVALUATOR_ZIP="$OUTPUT_DIR/music-crs-evaluator.zip"
BASELINES_ALIAS="$OUTPUT_DIR/UPLOAD-TO-COLAB-baselines.zip"
EVALUATOR_ALIAS="$OUTPUT_DIR/UPLOAD-TO-COLAB-evaluator.zip"

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
  -x "music-crs-baselines/*.egg-info/*" \
  -x "music-crs-baselines/*.egg-info/**/*" \
  -x "music-crs-baselines/.DS_Store" \
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
  -x "music-crs-evaluator/.DS_Store" \
  -x "*__pycache__*" \
  -x "*.pyc"

echo "Done."
cp "${BASELINES_ZIP}" "${BASELINES_ALIAS}"
cp "${EVALUATOR_ZIP}" "${EVALUATOR_ALIAS}"
ls -lh "${BASELINES_ZIP}" "${EVALUATOR_ZIP}" "${BASELINES_ALIAS}" "${EVALUATOR_ALIAS}"
echo "Testing archive integrity:"
unzip -tq "${BASELINES_ZIP}"
unzip -tq "${EVALUATOR_ZIP}"
echo "Baselines version:"
unzip -p "${BASELINES_ZIP}" music-crs-baselines/VERSION.txt
echo "Evaluator version:"
unzip -p "${EVALUATOR_ZIP}" music-crs-evaluator/VERSION.txt
