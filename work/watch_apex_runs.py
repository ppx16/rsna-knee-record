"""Watch the two pushed RSNA kernels until they reach a terminal state.

FACTS A173/A174: a transport failure must be retried in-call and must NEVER be
reported as a kernel verdict. Every poll distinguishes:
    RUNNING / QUEUED      -> keep waiting
    COMPLETE / ERROR / CANCEL* -> terminal, real verdict
    TRANSPORT             -> retried; if it persists the run is still "unknown", not failed

Usage: python work/watch_apex_runs.py [max_minutes]
"""
from __future__ import annotations

import re
import subprocess
import sys
import time

KERNELS = [
    ("ARC duck18-ve (self-contained pth)", "ppxl16/arc-duck18-ve"),
]

TERMINAL_OK = ("COMPLETE",)
TERMINAL_BAD = ("ERROR", "CANCEL", "FAILED")


def status_of(slug: str) -> str | None:
    """Return the status string, or None if every attempt was a transport failure."""
    for attempt in range(4):
        try:
            r = subprocess.run(["kaggle", "kernels", "status", slug],
                               capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=300)
        except Exception:  # noqa: BLE001
            time.sleep(10 * (attempt + 1))
            continue
        out = ((r.stdout or "") + (r.stderr or "")).strip()
        out = "\n".join(l for l in out.splitlines() if "outdated" not in l).strip()
        if not out:
            time.sleep(10 * (attempt + 1))
            continue
        if any(t in out for t in ("Max retries", "SSLError", "UNEXPECTED_EOF",
                                  "Connection", "timed out", "Timeout")):
            time.sleep(10 * (attempt + 1))
            continue
        m = re.search(r"KernelWorkerStatus\.([A-Z_]+)", out)
        if m:
            return m.group(1)
        return out[:200]
    return None


def main() -> int:
    max_min = float(sys.argv[1]) if len(sys.argv) > 1 else 90.0
    t0 = time.time()
    last: dict[str, str] = {}
    while True:
        elapsed = (time.time() - t0) / 60.0
        print(f"\n--- poll at {time.strftime('%H:%M:%SZ', time.gmtime())} "
              f"({elapsed:.1f} min elapsed) ---", flush=True)
        all_done = True
        for label, slug in KERNELS:
            st = status_of(slug)
            if st is None:
                print(f"  {label}: TRANSPORT FAILURE -- retried; status UNKNOWN "
                      f"(this is NOT a verdict, A173/A174)", flush=True)
                all_done = False
                continue
            mark = ""
            if st != last.get(slug):
                mark = "   <== CHANGED"
                last[slug] = st
            print(f"  {label}: {st}{mark}", flush=True)
            if not any(st.startswith(p) for p in TERMINAL_OK + TERMINAL_BAD):
                all_done = False

        if all_done:
            print("\n=== both kernels reached a terminal state ===", flush=True)
            for label, slug in KERNELS:
                print(f"  {label}: {last.get(slug)}", flush=True)
            return 0
        if elapsed > max_min:
            print(f"\n=== giving up polling after {elapsed:.1f} min (still running) ===",
                  flush=True)
            return 2
        time.sleep(120)


if __name__ == "__main__":
    raise SystemExit(main())
