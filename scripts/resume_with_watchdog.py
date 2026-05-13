"""
Wrapper that runs run_eval.py --resume and auto-restarts when it stalls.

Stall = responses.jsonl mtime unchanged for STALL_SECONDS.
Exits when all expected (model, tier) records exist for every dataset_index.
"""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/Users/razvan/research/evals/tooluni")
RUN_DIR = ROOT / "runs" / "00020_gpt55_internal_only_47clean"
PY = str(ROOT / ".venv" / "bin" / "python")

DATASET = "genetic_benchmark_v1/47-submissions-clean.json"
RUN_NAME = "00020_gpt55_internal_only_47clean"
MODEL = "claude-opus-4-7"
TIERS = ["internal_only", "web_tools", "tooluniverse"]
EXPECTED_INDICES = set(range(1, 48))

STALL_SECONDS = 7 * 60          # 7 minutes of no file growth = stall
POLL_SECONDS = 30
MAX_RESTARTS = 20


def parse_pretty_jsonl(path):
    import json
    if not path.exists():
        return []
    text = path.read_text()
    out = []
    decoder = json.JSONDecoder()
    i, n = 0, len(text)
    while i < n:
        while i < n and text[i] in " \n\r\t,":
            i += 1
        if i >= n:
            break
        obj, end = decoder.raw_decode(text, i)
        out.append(obj)
        i = end
    return out


def missing_jobs():
    resp = parse_pretty_jsonl(RUN_DIR / "responses.jsonl")
    done = {(r["dataset_index"], r["model_name"], r["tier"]) for r in resp}
    missing = []
    for idx in EXPECTED_INDICES:
        for tier in TIERS:
            if (idx, MODEL, tier) not in done:
                missing.append((idx, tier))
    return missing


def kill_tree(pid):
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    time.sleep(3)
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except ProcessLookupError:
        return
    # Also kill any leftover tooluniverse uv processes
    subprocess.run(["pkill", "-KILL", "-f", "tooluniverse"], check=False)


def launch():
    cmd = [
        PY, "scripts/run_eval.py",
        "--dataset", DATASET,
        "--run-name", RUN_NAME,
        "--models", MODEL,
        "--tiers", *TIERS,
        "--workers", "1",
        "--resume",
    ]
    log_path = RUN_DIR / "run.log"
    log_fh = open(log_path, "a", encoding="utf-8")
    log_fh.write(f"\n=== watchdog launch at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    log_fh.flush()
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return proc, log_fh


def main():
    restart_count = 0
    while True:
        miss = missing_jobs()
        if not miss:
            print(f"[watchdog] all jobs done — exiting")
            return 0
        print(f"[watchdog] {len(miss)} jobs missing — launching run_eval (attempt {restart_count + 1})")
        proc, log_fh = launch()
        resp_path = RUN_DIR / "responses.jsonl"
        last_mtime = resp_path.stat().st_mtime if resp_path.exists() else 0
        last_change = time.time()
        try:
            while True:
                ret = proc.poll()
                if ret is not None:
                    print(f"[watchdog] run_eval exited with code {ret}")
                    break
                time.sleep(POLL_SECONDS)
                cur_mtime = resp_path.stat().st_mtime if resp_path.exists() else 0
                if cur_mtime > last_mtime:
                    last_mtime = cur_mtime
                    last_change = time.time()
                    miss = missing_jobs()
                    print(f"[watchdog] progress: {len(miss)} jobs remaining")
                    if not miss:
                        print("[watchdog] all jobs done — terminating subprocess")
                        kill_tree(proc.pid)
                        return 0
                stall_for = time.time() - last_change
                if stall_for > STALL_SECONDS:
                    print(f"[watchdog] STALL: no responses.jsonl growth for {stall_for:.0f}s — killing")
                    kill_tree(proc.pid)
                    break
        finally:
            log_fh.close()
        restart_count += 1
        if restart_count >= MAX_RESTARTS:
            print(f"[watchdog] hit MAX_RESTARTS={MAX_RESTARTS} — giving up")
            return 1
        time.sleep(5)


if __name__ == "__main__":
    sys.exit(main())
