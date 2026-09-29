"""Wait for `ppxl16/rsna-knee-slope` to finish, pull its log, and census the per-stage timings.

This is the payoff step for `scripts/build_slope_kernel.py`: the slope kernel runs the SUBMISSION
pipeline over 200 TRAINING studies, and what we want out of it is the `[phase]` records -- one line per
stage with `elapsed_seconds` -- which give the per-study slope for the six stages that have no second
study count anywhere on disk (docs/EFFICIENCY-SLOPE.md §5).

Design notes, each of which is a bug this repo has already paid for:
  * `kaggle/__init__.py` runs `api.authenticate()` AT IMPORT, and the working credential path
    (`~/.kaggle/access_token`) makes a network call, so the import itself can raise SSLError. The import
    therefore lives INSIDE the retry, not outside it.
  * The downloaded log is a kaggle CLI capture, so it can begin with a version warning before the JSON
    payload. `phase_census.py` finds the payload rather than assuming the file starts with it.
  * Polling is deliberately slow (90 s). The run is expected at 10-35 min; anything tighter burns API
    calls for no information.

Usage:
    python scripts/watch_slope.py --minutes 70
"""
from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import subprocess
import sys
import time

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = pathlib.Path(__file__).resolve().parent.parent
REF = "ppxl16/rsna-knee-slope"
DEST = ROOT / "downloads" / "slope"


def _norm(st) -> str:
    """kaggle 2.1.2's `kernels_status` returns an ENUM, not a dict.

    Two bugs, both found by running it: the first version tested `isinstance(st, dict)`, fell through to
    `str(st)`, and compared the whole repr against "RUNNING"/"QUEUED" -- so it declared the run terminal
    after one poll and exited 1 while the kernel was still going. The second version found no `.status`
    attribute either, because the returned object IS the status. It is `KernelWorkerStatus.RUNNING`, so
    read `.name`; keep the other shapes as fallbacks rather than assuming this stays true.
    """
    if isinstance(st, dict):
        return str(st.get("status", "?")).upper()
    nm = getattr(st, "name", None)                      # enum member: KernelWorkerStatus.RUNNING
    if isinstance(nm, str):
        return nm.upper()
    attr = getattr(st, "status", None)
    if attr is not None and not isinstance(attr, type):
        return str(attr).rsplit(".", 1)[-1].upper()
    m = re.search(r'"status"\s*:\s*"([A-Za-z_]+)"', str(st))
    if m:
        return m.group(1).upper()
    return str(st).rsplit(".", 1)[-1].strip().upper()


def status_once() -> str:
    """Return the kernel status, or 'ERR' after retrying the whole thing including the import."""
    for attempt in range(4):
        try:
            from kaggle import api
            return _norm(api.kernels_status(REF))
        except Exception as e:
            print("[watch_slope] status attempt %d failed: %s: %s"
                  % (attempt + 1, type(e).__name__, str(e)[:90]), flush=True)
            time.sleep(20 * (attempt + 1))
    return "ERR"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, default=70)
    ap.add_argument("--poll", type=int, default=90)
    a = ap.parse_args()

    deadline = time.time() + a.minutes * 60
    last = None
    while time.time() < deadline:
        st = status_once()
        if st != last:
            print("[watch_slope] %s  (%.0f min left)" % (st, (deadline - time.time()) / 60), flush=True)
            last = st
        if st not in ("RUNNING", "QUEUED", "ERR"):
            break
        time.sleep(a.poll)

    print("[watch_slope] final status: %s" % last, flush=True)
    if last in ("RUNNING", "QUEUED"):
        print("[watch_slope] still running at the timeout; re-run to collect", flush=True)
        return 0

    DEST.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            from kaggle import api
            api.kernels_output(REF, path=str(DEST), force=True, quiet=True)
            break
        except Exception as e:
            print("[watch_slope] output attempt %d failed: %s: %s"
                  % (attempt + 1, type(e).__name__, str(e)[:90]), flush=True)
            time.sleep(20 * (attempt + 1))

    logs = sorted(DEST.glob("*.log"))
    if not logs:
        print("[watch_slope] no .log downloaded into %s; contents: %s"
              % (DEST, [p.name for p in DEST.iterdir()][:20]), flush=True)
        return 1
    src = max(logs, key=lambda p: p.stat().st_size)
    dst = ROOT / "state" / "logs_slope.json"
    shutil.copyfile(src, dst)
    print("[watch_slope] log -> %s (%d bytes)" % (dst, dst.stat().st_size), flush=True)

    print("\n===== per-stage durations =====", flush=True)
    print(subprocess.run([sys.executable, str(ROOT / "scripts" / "phase_census.py"),
                          "--roots", "state", "--fits"],
                         capture_output=True, text=True, encoding="utf-8",
                         errors="replace").stdout[-4000:], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
