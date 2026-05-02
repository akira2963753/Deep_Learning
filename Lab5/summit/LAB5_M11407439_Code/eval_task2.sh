#!/usr/bin/env bash
set -e

# Evaluate Task 2 snapshot for 20 episodes with seeds 0..19.
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

MODEL_PATH="../LAB5_M11407439_task2.pt"

if [ -z "$MODEL_PATH" ]; then
  echo "Please set MODEL_PATH in eval_task2.sh."
  exit 1
fi

"$PYTHON_BIN" test_model_task2.py \
  --model-path "$MODEL_PATH" \
  --episodes 20 \
  --seed 0 \
  --output-dir ./eval_videos_task2
