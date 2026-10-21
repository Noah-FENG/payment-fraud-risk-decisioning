"""Supervise `fraud develop` with retries around the nondeterministic native GIL crash.

The sklearn/numpy/scipy stack on this Windows host dies randomly with
"PyEval_SaveThread: the function must be called with the GIL held" (a native-level
fatal, not a Python exception). A full pass succeeded before, so this supervisor
re-runs develop until the run manifest reaches status "developed" or attempts run out.
Each attempt gets its own log: artifacts/develop_attempt_<n>.log.

Usage: uv run python scripts/run_develop_supervised.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts" / "main"
MANIFEST = RUN / "run_manifest.json"
MAX_ATTEMPTS = 6

# Single-thread every native math library: the GIL fatal correlates with their
# worker threads. XGBoost uses its own pool and stays parallel.
# FRAUD_MONITOR=0 disables the psutil sampler thread (same crash family).
ENV = {**os.environ,
       "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
       "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
       "FRAUD_MONITOR": "0"}

FRAUD = ROOT / ".venv" / "Scripts" / "fraud.exe"


def developed() -> bool:
    try:
        return json.loads(MANIFEST.read_text(encoding="utf-8")).get("status") == "developed"
    except (OSError, json.JSONDecodeError):
        return False


def main() -> None:
    if developed():
        print("already developed; nothing to do")
        return
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if RUN.exists():
            shutil.rmtree(RUN)
        log = (ROOT / "artifacts" / f"develop_attempt_{attempt}.log").open("w", encoding="utf-8")
        print(f"[attempt {attempt}] started {time.strftime('%H:%M:%S')}", flush=True)
        try:
            proc = subprocess.Popen([str(FRAUD), "--config", "config/default.json", "develop"],
                                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=ENV)
            code = proc.wait()
        except BaseException as exc:  # keep supervising even if wait() itself fails
            print(f"[attempt {attempt}] supervisor error: {exc!r}", flush=True)
            code = "supervisor-error"
        log.close()
        if developed():
            print(f"[attempt {attempt}] SUCCESS (exit {code})", flush=True)
            return
        print(f"[attempt {attempt}] failed (exit {code}); retrying", flush=True)
        time.sleep(10)
    print("all attempts exhausted; develop still not complete", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
