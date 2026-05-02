#!/usr/bin/env bash
set -e

# Evaluate Task 3 snapshots for 20 episodes with seeds 0..19.
cd "$(dirname "$0")"
if [ -z "${PYTHON_BIN:-}" ]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python.exe >/dev/null 2>&1; then
    PYTHON_BIN="python.exe"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "Python not found. Please install Python or set PYTHON_BIN."
    exit 1
  fi
fi

MODEL_PATHS=(
  "../LAB5_M11407439_task3_600000.pt"
  "../LAB5_M11407439_task3_1000000.pt"
  "../LAB5_M11407439_task3_1500000.pt"
  "../LAB5_M11407439_task3_2000000.pt"
  "../LAB5_M11407439_task3_2500000.pt"
  "../LAB5_M11407439_task3_best.pt"
)

if [ "${#MODEL_PATHS[@]}" -eq 0 ]; then
  echo "Please set MODEL_PATHS in eval_task3.sh."
  exit 1
fi

for model_path in "${MODEL_PATHS[@]}"; do
  if [ -z "$model_path" ]; then
    echo "Please set MODEL_PATHS in eval_task3.sh."
    exit 1
  fi
  echo "Evaluating ${model_path}"
  "$PYTHON_BIN" test_model_task3.py \
    --model_path "$model_path" \
    --episodes 20 \
    --seed 0
done
