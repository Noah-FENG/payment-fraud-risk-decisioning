"""Launch a detached finite batch process and return immediately.

Usage: uv run python scripts/run_develop_detached.py [script]
Defaults to the retrying supervisor; the plain script argument is kept for
single-shot debugging. The process terminates by itself; nothing here is a server.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "artifacts" / "develop_supervisor.log"
PIDFILE = ROOT / "artifacts" / "develop.pid"

target = sys.argv[1] if len(sys.argv) > 1 else "scripts/run_develop_supervised.py"
extra = sys.argv[2:]
log = LOG.open("w", encoding="utf-8")
kwargs = {"cwd": ROOT, "stdout": log, "stderr": subprocess.STDOUT}
if sys.platform == "win32":
    kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
else:
    kwargs["start_new_session"] = True
proc = subprocess.Popen([sys.executable, target, *extra], **kwargs)
PIDFILE.write_text(str(proc.pid), encoding="utf-8")
print(f"started pid={proc.pid} target={target}, log={LOG}")
