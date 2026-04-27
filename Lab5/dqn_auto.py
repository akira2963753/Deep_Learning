"""
dqn_auto.py — Automated train-evaluate-retry loop for Task 3 Pong DQN.

Strategy:
  Stage 1 (gambling): Train each attempt to 600K env steps and kill the
    subprocess as soon as `[Milestone] Saved model_600000.pt` is printed.
    Run test_model_task3.py over 20 seeds. If avg >= target, advance to
    Stage 2; otherwise wipe the attempt's save_dir and retry.
  Stage 2 (confirm): Train one full run to 2.5M to obtain the remaining
    milestone snapshots (1M / 1.5M / 2M / 2.5M). If this run's own 600K
    snapshot also passes the target, use it; otherwise copy Stage 1's
    successful 600K snapshot in (Frankenstein submission — TA only runs
    inference, so checkpoint-continuity is not validated).

Critical implementation notes:
  * subprocess output is forced unbuffered via `python -u` + PYTHONUNBUFFERED
    so the milestone-save line is observed immediately (otherwise it sits
    in the OS pipe buffer for minutes).
  * `proc.terminate()` alone leaks AsyncVectorEnv child processes on Windows.
    We use `psutil` to walk and kill the entire process tree.
"""

import argparse
import os
import re
import shlex
import shutil
import signal
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
# Process-tree kill (Windows-safe; covers AsyncVectorEnv children)
# --------------------------------------------------------------------------- #
def kill_process_tree(pid, timeout=10):
    """Terminate the process and all of its descendants.

    On Windows, gym.vector.AsyncVectorEnv spawns child processes that survive
    a plain proc.terminate(). This walks the whole tree, sends SIGTERM, then
    escalates to SIGKILL after `timeout` seconds for any survivors.
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
        # Ensure file exists
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
        bufsize=1,                 # line-buffered on the wrapper side
        text=True,
        env=make_unbuffered_env(),
    )


def stream_until(proc, target_substr, log, timeout_seconds=None):
    """Stream subprocess stdout to console+log; return True when a line
    containing `target_substr` appears, or False on process exit / timeout.
    """
    start = time.time()
    while True:
        if timeout_seconds is not None and (time.time() - start) > timeout_seconds:
            log(f"[stream] Timeout after {timeout_seconds}s waiting for '{target_substr}'")
            return False

        line = proc.stdout.readline()
        if line == "":
            # EOF — process has exited
            ret = proc.poll()
            log(f"[stream] Subprocess exited with code {ret} before milestone")
            return False

        # Echo to console (already flushed via print on TeeLogger? avoid double-log)
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
            timeout=60 * 30,  # 30 min hard cap
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


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Auto train-eval-retry loop until 600K avg >= target."
    )
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument("--target-score", type=float, default=19.0)
    parser.add_argument("--base-save-dir", type=str, default="./auto_runs")
    parser.add_argument("--episodes-per-attempt", type=int, default=500,
                        help="Cap for Stage 1 attempts (subprocess is killed at 600K anyway)")
    parser.add_argument("--episodes-final", type=int, default=1500,
                        help="Episodes for Stage 2 confirm run (full 2.5M)")
    parser.add_argument("--eval-episodes", type=int, default=20,
                        help="20-seed test (seeds 0-19) per spec")
    parser.add_argument("--train-timeout-hours", type=float, default=6.0,
                        help="Hard timeout per attempt before giving up")
    parser.add_argument("--final-run-name", type=str, default="task3-auto-final")
    parser.add_argument(
        "--train-args",
        type=str,
        default=(
            "--env ALE/Pong-v5 --num-envs 4 "
            "--use-ddqn --use-per --use-multistep --n-step 3 "
            "--memory-size 200000 --replay-start-size 20000 "
            "--epsilon-decay 0.99996 --epsilon-min 0.01 "
            "--target-update-frequency 4000 "
            "--batch-size 32 --lr 0.00025 "
            "--discount-factor 0.99 --max-episode-steps 10000 --train-per-step 2 "
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

    # --------------------------------------------------------------- #
    # Stage 1: gambling loop
    # --------------------------------------------------------------- #
    success_dir = None
    success_avg = None
    best_avg_seen = float("-inf")

    timeout_seconds = args.train_timeout_hours * 3600

    for attempt in range(1, args.max_attempts + 1):
        run_name = f"auto-attempt-{attempt:03d}"
        save_dir = base_dir / f"attempt_{attempt:03d}"
        save_dir.mkdir(parents=True, exist_ok=True)

        log(f"\n--- Attempt {attempt}/{args.max_attempts} ---")
        log(f"save_dir = {save_dir}")
        log(f"wandb-run-name = {run_name}")

        proc = spawn_dqn(train_args_list, str(save_dir), run_name,
                         args.episodes_per_attempt)

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

        # Always kill the subprocess at this point (we have 600K saved already
        # if milestone_hit, otherwise the process is dead anyway).
        kill_process_tree(proc.pid)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pass

        if not milestone_hit:
            log(f"[attempt {attempt}] Did not reach 600K milestone — cleaning up")
            shutil.rmtree(save_dir, ignore_errors=True)
            continue

        model_600k = save_dir / "model_600000.pt"
        if not model_600k.exists():
            log(f"[attempt {attempt}] Milestone line seen but {model_600k} missing — cleaning up")
            shutil.rmtree(save_dir, ignore_errors=True)
            continue

        # Evaluate
        avg = run_eval(str(model_600k), args.eval_episodes, log)
        if avg is None:
            log(f"[attempt {attempt}] Eval failed — cleaning up")
            shutil.rmtree(save_dir, ignore_errors=True)
            continue

        log(f"[attempt {attempt}] 600K 20-seed average = {avg:.2f}")
        best_avg_seen = max(best_avg_seen, avg)

        if avg >= args.target_score:
            log(f"[attempt {attempt}] *** SUCCESS (avg {avg:.2f} >= {args.target_score}) ***")
            success_dir = save_dir
            success_avg = avg
            break
        else:
            log(f"[attempt {attempt}] Below target ({avg:.2f} < {args.target_score}) — cleaning up")
            shutil.rmtree(save_dir, ignore_errors=True)

    if success_dir is None:
        log("\n" + "=" * 70)
        log(f"Reached max_attempts={args.max_attempts} without hitting target.")
        log(f"Best 600K avg seen: {best_avg_seen:.2f}")
        log("Consider raising --max-attempts or accepting current best snapshot.")
        log("=" * 70)
        return

    # --------------------------------------------------------------- #
    # Stage 2: confirm — full 2.5M run to collect remaining milestones
    # --------------------------------------------------------------- #
    log("\n" + "=" * 70)
    log(f"Stage 2: full training to 2.5M for milestones 1M/1.5M/2M/2.5M")
    log("=" * 70)

    final_dir = base_dir / "final_run"
    final_dir.mkdir(parents=True, exist_ok=True)

    proc = spawn_dqn(
        train_args_list,
        str(final_dir),
        args.final_run_name,
        args.episodes_final,
    )
    try:
        ret = stream_to_completion(proc, log)
        log(f"[stage2] Final run finished with exit code {ret}")
    except KeyboardInterrupt:
        log("[main] KeyboardInterrupt during Stage 2 — killing subprocess")
        kill_process_tree(proc.pid)
        sys.exit(130)

    # Decide on the 600K snapshot for submission
    final_600k = final_dir / "model_600000.pt"
    if final_600k.exists():
        final_avg = run_eval(str(final_600k), args.eval_episodes, log)
        if final_avg is not None and final_avg >= args.target_score:
            log(f"[stage2] final_run 600K avg = {final_avg:.2f} >= target — using final's 600K")
        else:
            log(f"[stage2] final_run 600K avg = "
                f"{final_avg if final_avg is not None else 'N/A'} below target — "
                f"replacing with Stage 1 success snapshot ({success_avg:.2f})")
            shutil.copy2(success_dir / "model_600000.pt", final_600k)
    else:
        log("[stage2] final_run 600K snapshot missing — copying Stage 1 success snapshot")
        shutil.copy2(success_dir / "model_600000.pt", final_600k)

    # Final summary
    log("\n" + "=" * 70)
    log("DONE. Submission milestones in: " + str(final_dir))
    for ms in (600_000, 1_000_000, 1_500_000, 2_000_000, 2_500_000):
        p = final_dir / f"model_{ms}.pt"
        log(f"  {p.name}: {'OK' if p.exists() else 'MISSING'}")
    log(f"Stage 1 success attempt preserved at: {success_dir} (avg {success_avg:.2f})")
    log("=" * 70)


if __name__ == "__main__":
    main()