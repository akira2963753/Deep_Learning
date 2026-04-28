"""
dqn_auto.py — Automated train-evaluate-retry loop for Task 3 Pong DQN.

Strategy:
  Per attempt: train with a unique seed, watch for the
  `[Milestone] Saved model_600000.pt` line, then SUSPEND the subprocess
  and evaluate that snapshot over 20 seeds. If avg >= target, RESUME the
  same subprocess so training continues to 2.5M — preserving checkpoint
  continuity for the remaining milestones (1M / 1.5M / 2M / 2.5M).
  Otherwise, kill the subprocess, wipe the attempt's save_dir, and retry
  with the next seed.

Why per-attempt seed variation:
  In single-env mode the only remaining source of run-to-run noise is
  cudnn.benchmark, which is too weak to reliably explore different policy
  basins across attempts. Using `--seed (base + attempt)` gives true
  statistical independence between attempts.

Critical implementation notes:
  * subprocess output is forced unbuffered via `python -u` + PYTHONUNBUFFERED
    so the milestone-save line is observed immediately.
  * psutil.suspend() pauses the training process during eval to avoid
    GPU contention. resume() continues it without losing any state.
"""

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import psutil
except ImportError:
    print("[FATAL] psutil is required. Install it with:  pip install psutil")
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Process-tree control (Windows-safe)
# --------------------------------------------------------------------------- #
def kill_process_tree(pid, timeout=10):
    """Terminate the process and all of its descendants.

    Uses psutil because plain proc.terminate() doesn't always clean up
    grandchildren on Windows (e.g. wandb's sync subprocess).
    """
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    try:
        children = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        children = []

    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    try:
        parent.terminate()
    except psutil.NoSuchProcess:
        pass

    _, alive = psutil.wait_procs(children + [parent], timeout=timeout)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass


def suspend_process_tree(pid):
    """Freeze a process and its descendants — releases GPU compute but
    keeps GPU memory allocated. Use during eval to avoid contention."""
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    try:
        for child in parent.children(recursive=True):
            try:
                child.suspend()
            except psutil.NoSuchProcess:
                pass
        parent.suspend()
    except psutil.NoSuchProcess:
        pass


def resume_process_tree(pid):
    """Resume a previously-suspended process tree."""
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    try:
        parent.resume()
        for child in parent.children(recursive=True):
            try:
                child.resume()
            except psutil.NoSuchProcess:
                pass
    except psutil.NoSuchProcess:
        pass


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
class TeeLogger:
    """Print to stdout AND append to a log file."""
    def __init__(self, log_path):
        self.log_path = log_path
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8"):
            pass

    def __call__(self, msg):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# --------------------------------------------------------------------------- #
# Subprocess helpers
# --------------------------------------------------------------------------- #
def make_unbuffered_env():
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return env


def spawn_dqn(train_args_list, save_dir, run_name, episodes, seed):
    """Start dqn.py as an unbuffered subprocess. Returns Popen handle."""
    cmd = [
        sys.executable, "-u", "dqn.py",
        "--save-dir", save_dir,
        "--wandb-run-name", run_name,
        "--episodes", str(episodes),
        "--seed", str(seed),
    ] + train_args_list

    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
        env=make_unbuffered_env(),
    )


def stream_until(proc, target_substr, log, timeout_seconds=None):
    """Stream subprocess stdout to console+log; return True when a line
    containing `target_substr` appears, False on process exit / timeout.
    """
    start = time.time()
    while True:
        if timeout_seconds is not None and (time.time() - start) > timeout_seconds:
            log(f"[stream] Timeout after {timeout_seconds}s waiting for '{target_substr}'")
            return False

        line = proc.stdout.readline()
        if line == "":
            ret = proc.poll()
            log(f"[stream] Subprocess exited with code {ret} before milestone")
            return False

        sys.stdout.write(line)
        sys.stdout.flush()

        if target_substr in line:
            return True


def stream_to_completion(proc, log):
    """Stream subprocess stdout until it exits naturally."""
    while True:
        line = proc.stdout.readline()
        if line == "":
            proc.wait()
            return proc.returncode
        sys.stdout.write(line)
        sys.stdout.flush()


def run_eval(model_path, eval_episodes, log):
    """Call test_model_task3.py and parse the 20-seed average."""
    cmd = [
        sys.executable, "-u", "test_model_task3.py",
        "--model_path", model_path,
        "--episodes", str(eval_episodes),
        "--seed", "0",
    ]
    log(f"[eval] Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=make_unbuffered_env(),
            timeout=60 * 30,
        )
    except subprocess.TimeoutExpired:
        log("[eval] Timed out (>30 min)")
        return None

    sys.stdout.write(result.stdout)
    sys.stdout.flush()

    m = re.search(r"Average reward:\s*([\-0-9.]+)", result.stdout)
    if not m:
        log("[eval] Could not parse 'Average reward' from stdout")
        return None
    return float(m.group(1))


def cleanup_attempt(proc, save_dir, log, *, was_suspended=False):
    """Resume (if needed) → kill subprocess → wait → wipe save_dir."""
    if was_suspended:
        resume_process_tree(proc.pid)
    kill_process_tree(proc.pid)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        log("[cleanup] Subprocess didn't exit within 15s after kill")
    shutil.rmtree(save_dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Auto train-eval-retry loop. On success at 600K, the SAME "
                    "training run continues to 2.5M for checkpoint continuity."
    )
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument("--target-score", type=float, default=19.0)
    parser.add_argument("--base-save-dir", type=str, default="./auto_runs")
    parser.add_argument("--episodes-per-attempt", type=int, default=2500,
                        help="Episodes cap (single-env: ~2500 needed to reach 2.5M)")
    parser.add_argument("--eval-episodes", type=int, default=20,
                        help="20-seed test (seeds 0-19) per spec")
    parser.add_argument("--milestone-timeout-hours", type=float, default=4.0,
                        help="Hard timeout per attempt while waiting for the 600K milestone line")
    parser.add_argument("--seed-base", type=int, default=42,
                        help="First attempt uses this seed; attempt N uses seed_base + (N-1)")
    parser.add_argument(
        "--train-args",
        type=str,
        default=(
            "--env ALE/Pong-v5 "
            "--use-ddqn --use-per --use-multistep --n-step 3 "
            "--memory-size 200000 --replay-start-size 20000 "
            "--epsilon-decay 0.99996 --epsilon-min 0.01 "
            "--target-update-frequency 8000 "
            "--batch-size 32 --lr 0.00025 "
            "--discount-factor 0.99 --max-episode-steps 10000"
        ),
        help="Extra args passed to dqn.py (excluding --save-dir, --wandb-run-name, --episodes, --seed)",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_save_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    log_path = base_dir / "auto_log.txt"
    log = TeeLogger(str(log_path))

    train_args_list = shlex.split(args.train_args)
    log("=" * 70)
    log(f"dqn_auto.py started. base_dir={base_dir}, target={args.target_score}, "
        f"max_attempts={args.max_attempts}, seed_base={args.seed_base}")
    log(f"Train args: {args.train_args}")
    log("=" * 70)

    timeout_seconds = args.milestone_timeout_hours * 3600
    best_avg_seen = float("-inf")
    best_seed_seen = None

    for attempt in range(1, args.max_attempts + 1):
        seed = args.seed_base + (attempt - 1)
        run_name = f"auto-attempt-{attempt:03d}-seed{seed}"
        save_dir = base_dir / f"attempt_{attempt:03d}_seed{seed}"
        save_dir.mkdir(parents=True, exist_ok=True)

        log(f"\n--- Attempt {attempt}/{args.max_attempts}  seed={seed} ---")
        log(f"save_dir       = {save_dir}")
        log(f"wandb-run-name = {run_name}")

        proc = spawn_dqn(train_args_list, str(save_dir), run_name,
                         args.episodes_per_attempt, seed)

        try:
            milestone_hit = stream_until(
                proc,
                target_substr="model_600000.pt",
                log=log,
                timeout_seconds=timeout_seconds,
            )
        except KeyboardInterrupt:
            log("[main] KeyboardInterrupt — killing subprocess and exiting")
            kill_process_tree(proc.pid)
            sys.exit(130)

        if not milestone_hit:
            log(f"[attempt {attempt}] Did not reach 600K milestone — cleaning up")
            cleanup_attempt(proc, save_dir, log)
            continue

        model_600k = save_dir / "model_600000.pt"
        if not model_600k.exists():
            log(f"[attempt {attempt}] Milestone line seen but {model_600k} missing — cleaning up")
            cleanup_attempt(proc, save_dir, log)
            continue

        # Suspend training so eval doesn't compete for GPU
        log(f"[attempt {attempt}] 600K reached — suspending subprocess for eval")
        suspend_process_tree(proc.pid)

        avg = run_eval(str(model_600k), args.eval_episodes, log)

        if avg is None:
            log(f"[attempt {attempt}] Eval failed — cleaning up")
            cleanup_attempt(proc, save_dir, log, was_suspended=True)
            continue

        log(f"[attempt {attempt}] 600K 20-seed average = {avg:.2f} (seed={seed})")
        if avg > best_avg_seen:
            best_avg_seen = avg
            best_seed_seen = seed

        if avg < args.target_score:
            log(f"[attempt {attempt}] Below target ({avg:.2f} < {args.target_score}) — "
                f"trying next seed")
            cleanup_attempt(proc, save_dir, log, was_suspended=True)
            continue

        # SUCCESS — resume the same run so milestones 1M/1.5M/2M/2.5M
        # are produced from the same continuous trajectory as the 600K snapshot.
        log(f"[attempt {attempt}] *** SUCCESS (avg {avg:.2f} >= {args.target_score}) ***")
        log(f"[attempt {attempt}] Resuming training to 2.5M (continuous run, no checkpoint break)")
        resume_process_tree(proc.pid)

        try:
            ret = stream_to_completion(proc, log)
            log(f"[attempt {attempt}] Training finished with exit code {ret}")
        except KeyboardInterrupt:
            log("[main] KeyboardInterrupt during continuation — killing subprocess")
            kill_process_tree(proc.pid)
            sys.exit(130)

        log("\n" + "=" * 70)
        log(f"DONE. Successful attempt: {attempt}, seed={seed}, 600K avg={avg:.2f}")
        log(f"Submission milestones in: {save_dir}")
        for ms in (600_000, 1_000_000, 1_500_000, 2_000_000, 2_500_000):
            p = save_dir / f"model_{ms}.pt"
            log(f"  {p.name}: {'OK' if p.exists() else 'MISSING'}")
        log("=" * 70)
        return

    # Exhausted all attempts
    log("\n" + "=" * 70)
    log(f"Reached max_attempts={args.max_attempts} without hitting target.")
    log(f"Best 600K avg seen: {best_avg_seen:.2f} (seed={best_seed_seen})")
    log("Consider raising --max-attempts, lowering --target-score, "
        "or rerunning the best seed and accepting that snapshot.")
    log("=" * 70)


if __name__ == "__main__":
    main()
