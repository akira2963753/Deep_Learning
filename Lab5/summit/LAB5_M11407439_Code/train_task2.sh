#!/usr/bin/env bash
set -e

# Train Task 2: vanilla DQN on ALE/Pong-v5.
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

"$PYTHON_BIN" dqn_task2.py \
  --env ALE/Pong-v5 \
  --num-envs 4 \
  --save-dir ./results_task2 \
  --episodes 10000 \
  --memory-size 500000 \
  --replay-start-size 10000 \
  --epsilon-decay 0.9999995 \
  --epsilon-min 0.05 \
  --target-update-frequency 1000 \
  --batch-size 32 \
  --lr 0.0001 \
  --discount-factor 0.99 \
  --max-episode-steps 10000
