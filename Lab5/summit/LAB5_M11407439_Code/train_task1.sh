#!/usr/bin/env bash
set -e

# Train Task 1: vanilla DQN on CartPole-v1.
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

"$PYTHON_BIN" dqn_task1.py \
  --save-dir ./results_task1 \
  --replay-start-size 5000 \
  --episodes 3000 \
  --epsilon-decay 0.9995
