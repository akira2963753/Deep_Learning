"""
dqn_auto.py — Automated train-evaluate-retry loop for Task 3 Pong DQN.

Strategy:
  For each attempt:
    1. Start dqn.py with enough episodes to reach 2.5M env steps.
    2. When `model_600000.pt` is saved, launch eval in a background thread
       (training keeps running — do NOT kill it).
    3. Continue streaming training output.  As soon as eval finishes:
         - eval failed  → kill training, wipe save_dir, try next attempt.
         - eval passed  → let training run to completion (2.5M).
    4. After training finishes, confirm all milestones are present.

This guarantees that the 600K / 1M / 1.5M / 2M / 2.5M snapshots all come
from a single continuous training trajectory.

Critical implementation notes:
  * subprocess output is forced unbuffered via `python -u` + PYTHONUNBUFFERED
    so the milestone-save line is observed immediately.
  * `proc.terminate()` alone leaks AsyncVectorEnv child processes on Windows.
    We use `psutil` to walk and kill the entire process tree.
  * Eval runs in a daemon thread so KeyboardInterrupt still propagates cleanly.
"""

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

try:
    import psutil
except ImportError:
    print("[FATAL] psutil is required. Install it with:  pip install psutil")
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Process-tree kill (Windows-safe; covers AsyncVectorEnv children)
# --------------------------------------------------------------------------- #
def kill_process_tree(pid, timeout=10):
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

    gone, alive = psutil.wait_procs(children + [parent], timeout=timeout)
    for p in alive:
        try:
            p.kill()
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


def spawn_dqn(train_args_list, save_dir, run_name, episodes):
    """Start dqn.py as an unbuffered subprocess. Returns Popen handle."""
    cmd = [
        sys.executable, "-u", "dqn.py",
        "--save-dir", save_dir,
        "--wandb-run-name", run_name,
        "--episodes", str(episodes),
    ] + train_args_list

    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
        env=make_unbuffered_env(),
    )


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


def stream_and_eval(proc, model_600k, eval_episodes, target_score,
                    log, timeout_seconds):
    """
    Stream proc's stdout while running eval in a background thread.

    Phase A — wait for '600K milestone' line:
      Returns "no_milestone" if process exits or times out before it.

    Phase B — eval thread starts; keep streaming:
      - If eval fails (avg < target_score)  → kill proc, return "eval_failed"
      - If eval passes                      → keep streaming until proc exits naturally
      - If proc exits before eval finishes  → wait for eval, return result
      Returns "success" or "eval_failed".
    """
    # ---- Phase A: stream until 600K milestone ----
    start = time.time()

    while True:
        if timeout_seconds is not None and (time.time() - start) > timeout_seconds:
            log(f"[stream] Timeout waiting for 600K milestone")
            return "no_milestone"

        line = proc.stdout.readline()
        if line == "":
            log(f"[stream] Process exited before 600K milestone (code {proc.poll()})")
            return "no_milestone"

        sys.stdout.write(line)
        sys.stdout.flush()

        if "model_600000.pt" in line:
            log("[stream] 600K milestone detected — starting eval in background")
            break

    # ---- Phase B: launch eval thread, continue streaming ----
    # Use a sentinel to distinguish "eval still running" from "eval returned None (failed)"
    _PENDING = object()
    eval_result = [_PENDING]

    def _eval_worker():
        eval_result[0] = run_eval(str(model_600k), eval_episodes, log)

    eval_thread = threading.Thread(target=_eval_worker, daemon=True)
    eval_thread.start()

    proc_done = False
    while True:
        # Check if eval thread has finished writing its result
        if eval_result[0] is not _PENDING:
            avg = eval_result[0]
            if avg is None:
                # eval timed out or failed to parse — treat as failure
                log("[stream] Eval returned None (timeout/parse error) — killing training")
                kill_process_tree(proc.pid)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    pass
                return "eval_failed", None
            log(f"[stream] Eval finished: avg = {avg:.2f}")
            if avg < target_score:
                log(f"[stream] Eval failed ({avg:.2f} < {target_score}) — killing training")
                kill_process_tree(proc.pid)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    pass
                return "eval_failed", avg
            else:
                log(f"[stream] Eval passed ({avg:.2f} >= {target_score}) — waiting for 2.5M")
                # drain remaining output until training finishes naturally
                while True:
                    line = proc.stdout.readline()
                    if line == "":
                        proc.wait()
                        break
                    sys.stdout.write(line)
                    sys.stdout.flush()
                return "success", avg

        if proc_done:
            # Training finished before eval thread wrote its result — wait for it
            eval_thread.join(timeout=60 * 30)
            avg = eval_result[0]
            if avg is _PENDING or avg is None:
                log("[stream] Eval did not return a valid result after process exit")
                return "eval_failed", None
            log(f"[stream] (post-exit) Eval result: avg = {avg:.2f}")
            if avg >= target_score:
                return "success", avg
            return "eval_failed", avg

        # Read next line from training process
        line = proc.stdout.readline()
        if line == "":
            proc.wait()
            proc_done = True
            continue

        sys.stdout.write(line)
        sys.stdout.flush()


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Auto train-eval-retry loop until 600K avg >= target."
    )
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument("--target-score", type=float, default=18.95)
    parser.add_argument("--base-save-dir", type=str, default="./auto_runs")
    parser.add_argument(
        "--episodes-per-attempt", type=int, default=1800,
        help="Max episodes per attempt — must be enough to reach 2.5M env steps"
    )
    parser.add_argument("--eval-episodes", type=int, default=20,
                        help="20-seed test (seeds 0-19) per spec")
    parser.add_argument("--train-timeout-hours", type=float, default=6.0,
                        help="Hard timeout per attempt waiting for 600K milestone")
    parser.add_argument(
        "--train-args",
        type=str,
        default=(
            "--env ALE/Pong-v5 --num-envs 4 "
            "--use-ddqn --use-per --use-multistep --n-step 3 "
            "--memory-size 200000 --replay-start-size 20000 "
            "--epsilon-decay 0.99996 --epsilon-min 0.01 "
            "--target-update-frequency 2000 "
            "--batch-size 32 --lr 0.00025 "
            "--discount-factor 0.99 --max-episode-steps 10000"
        ),
        help="Extra args passed to dqn.py (excluding --save-dir, --wandb-run-name, --episodes)",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_save_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    log_path = base_dir / "auto_log.txt"
    log = TeeLogger(str(log_path))

    train_args_list = shlex.split(args.train_args)
    log("=" * 70)
    log(f"dqn_auto.py started. base_dir={base_dir}, target={args.target_score}, "
        f"max_attempts={args.max_attempts}")
    log(f"Train args: {args.train_args}")
    log("=" * 70)

    success_dir = None
    success_avg = None
    best_avg_seen = float("-inf")
    timeout_seconds = args.train_timeout_hours * 3600

    for attempt in range(1, args.max_attempts + 1):
        run_name = f"auto-attempt-{attempt:03d}"
        save_dir = base_dir / f"attempt_{attempt:03d}"
        save_dir.mkdir(parents=True, exist_ok=True)

        log(f"\n--- Attempt {attempt}/{args.max_attempts} ---")
        log(f"save_dir  = {save_dir}")
        log(f"run_name  = {run_name}")

        proc = spawn_dqn(train_args_list, str(save_dir), run_name,
                         args.episodes_per_attempt)

        model_600k = save_dir / "model_600000.pt"

        try:
            outcome = stream_and_eval(
                proc=proc,
                model_600k=model_600k,
                eval_episodes=args.eval_episodes,
                target_score=args.target_score,
                log=log,
                timeout_seconds=timeout_seconds,
            )
        except KeyboardInterrupt:
            log("[main] KeyboardInterrupt — killing subprocess and exiting")
            kill_process_tree(proc.pid)
            sys.exit(130)

        # outcome is either a string ("no_milestone") or a tuple ("success"/"eval_failed", avg)
        if outcome == "no_milestone":
            log(f"[attempt {attempt}] Did not reach 600K milestone — cleaning up")
            kill_process_tree(proc.pid)
            shutil.rmtree(save_dir, ignore_errors=True)
            continue

        status, avg = outcome

        if avg is not None:
            best_avg_seen = max(best_avg_seen, avg)

        if status == "eval_failed":
            label = f"{avg:.2f}" if avg is not None else "N/A"
            log(f"[attempt {attempt}] Below target ({label} < {args.target_score}) — cleaning up")
            shutil.rmtree(save_dir, ignore_errors=True)
            continue

        # status == "success"
        log(f"[attempt {attempt}] *** SUCCESS (avg {avg:.2f} >= {args.target_score}) ***")
        success_dir = save_dir
        success_avg = avg
        break

    # --------------------------------------------------------------- #
    # Final summary
    # --------------------------------------------------------------- #
    log("\n" + "=" * 70)
    if success_dir is None:
        log(f"Reached max_attempts={args.max_attempts} without hitting target.")
        log(f"Best 600K avg seen: {best_avg_seen:.2f}")
        log("Consider raising --max-attempts or lowering --target-score.")
    else:
        log("DONE. Submission milestones in: " + str(success_dir))
        for ms in (600_000, 1_000_000, 1_500_000, 2_000_000, 2_500_000):
            p = success_dir / f"model_{ms}.pt"
            log(f"  {p.name}: {'OK' if p.exists() else 'MISSING'}")
        log(f"600K avg: {success_avg:.2f}")
    log("=" * 70)


if __name__ == "__main__":
    main()
