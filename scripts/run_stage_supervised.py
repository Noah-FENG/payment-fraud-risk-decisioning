"""Supervise a long fraud pipeline stage with retries around the native GIL crash.

For `evaluate`: a crashed attempt leaves evaluation_started.json behind, which would
block any retry — the marker is removed before each new attempt. The test-exposure
ledger is append-only and records every attempt honestly; a crashed attempt never
reaches the ledger write, so no phantom exposure is recorded.

Usage: uv run python scripts/run_stage_supervised.py <develop|evaluate|report|explain>
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
MAX_ATTEMPTS = 6

ENV = {**os.environ,
       "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
       "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
       "FRAUD_MONITOR": "0"}

FRAUD = ROOT / ".venv" / "Scripts" / "fraud.exe"

DONE_CHECKS = {
    "develop": lambda: json.loads((RUN / "run_manifest.json").read_text(encoding="utf-8")).get("status") == "developed",
    "evaluate": lambda: (RUN / "evaluation.json").exists(),
    "report": lambda: (RUN / "report_manifest.json").exists(),
    "explain": lambda: (RUN / "explanations" / "explanation.json").exists() or (RUN / "explanation.json").exists(),
}


def cleanup_for_retry(stage: str) -> None:
    if stage == "develop" and RUN.exists():
        shutil.rmtree(RUN)
    if stage == "evaluate":
        marker = RUN / "evaluation_started.json"
        if marker.exists():
            marker.unlink()


def main() -> None:
    stage = sys.argv[1]
    done = DONE_CHECKS[stage]
    def finished() -> bool:
        try:
            return done()
        except (OSError, json.JSONDecodeError):
            return False
    if finished():
        print(f"{stage} already complete; nothing to do", flush=True)
        return
    for attempt in range(1, MAX_ATTEMPTS + 1):
        cleanup_for_retry(stage)
        log = (ROOT / "artifacts" / f"{stage}_attempt_{attempt}.log").open("w", encoding="utf-8")
        print(f"[{stage} attempt {attempt}] started {time.strftime('%H:%M:%S')}", flush=True)
        try:
            proc = subprocess.Popen([str(FRAUD), "--config", "config/default.json", stage],
                                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=ENV)
            code = proc.wait()
        except BaseException as exc:
            print(f"[{stage} attempt {attempt}] supervisor error: {exc!r}", flush=True)
            code = "supervisor-error"
        log.close()
        if finished():
            print(f"[{stage} attempt {attempt}] SUCCESS (exit {code})", flush=True)
            return
        print(f"[{stage} attempt {attempt}] failed (exit {code}); retrying", flush=True)
        time.sleep(10)
    print(f"{stage}: all attempts exhausted", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
