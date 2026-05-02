#!/usr/bin/env bash
set -e

# Train Task 3 on ALE/Pong-v5 with the command used for the reported run.
cd "$(dirname "$0")"
export WANDB_MODE=disabled
export WANDB_DISABLED=true
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

"$PYTHON_BIN" dqn_task3.py \
  --env ALE/Pong-v5 \
  --num-envs 4 \
  --use-ddqn \
  --use-per \
  --use-multistep \
  --n-step 3 \
  --save-dir ./results_task3 \
  --episodes 1500 \
  --memory-size 200000 \
  --replay-start-size 20000 \
  --target-update-frequency 2000 \
  --epsilon-decay 0.99996 \
  --epsilon-min 0.01 \
  --batch-size 32 \
  --lr 0.00025 \
  --discount-factor 0.99
