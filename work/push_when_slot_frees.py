"""Retry a Kaggle kernel push until the GPU session slot frees up.

Why this exists: on 2026-10-11 the high-tier ARC kernel (`kernel-arc-duck18-va`, the one with the
rendering blocker FIXED) could not be pushed because the batch GPU session cap is 2 and BOTH slots
were held by our own kernels:

    Kernel push error: Maximum batch GPU session count of 2 reached.

The CLI has no `cancel` verb, so the only way forward is to wait for a slot. Doing that by hand
means the high-tier line starts whenever someone next looks -- which is exactly the "I stopped at
14:00" failure mode. So this loop owns it instead.

It is deliberately dumb and safe:
  * it calls the SAME guarded pusher (`work/push_kernel.py`), so the allow-list, the GBK check and
    the "successfully pushed" verdict all still apply;
  * a quota/session refusal is NOT an error -- it is the expected waiting state;
  * any OTHER failure (bad metadata, auth, transport) stops the loop so it cannot spin on a real bug;
  * it gives up after a bounded window and says so.

Usage: python work/push_when_slot_frees.py <kernel-dir> [hours] [interval_min]
"""
from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Substrings that mean "not now, try again" rather than "this will never work".
WAIT_FOR = (
    "Maximum batch GPU session count",
    "Maximum weekly GPU quota",
    "Maximum batch TPU session count",
    "Too many sessions",
    # ⚠️ Added 2026-10-11 after this waiter STOPPED on one of these and treated it as a real bug.
    # A bodyless / non-JSON response is a TRANSPORT failure, and FACTS A173/A174 are explicit that a
    # transport failure must never be read as a verdict. The Kaggle CLI surfaces it as
    # "Expecting value: line 1 column 1 (char 0)" with rc=1.
    "Expecting value",
    "SSLError",
    "UNEXPECTED_EOF",
    "Max retries",
    "Connection aborted",
    "ConnectionReset",
    "Read timed out",
    "timed out",
)


def try_push(kernel_dir: str) -> tuple[bool, str]:
    r = subprocess.run([sys.executable, str(ROOT / "work" / "push_kernel.py"), kernel_dir],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=1800, cwd=str(ROOT))
    return r.returncode == 0, (r.stdout or "") + (r.stderr or "")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    kernel_dir = argv[1]
    hours = float(argv[2]) if len(argv) > 2 else 10.0
    every = float(argv[3]) if len(argv) > 3 else 10.0
    t0 = time.time()
    print(f"waiting for a GPU slot to push {kernel_dir!r} "
          f"(window {hours:.1f} h, retry every {every:.0f} min)")
    attempt = 0
    while True:
        attempt += 1
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%SZ")
        ok, out = try_push(kernel_dir)
        if ok:
            tail = [l for l in out.splitlines() if "PUSH OK" in l or "MISMATCH" in l]
            print(f"[{stamp}] attempt {attempt}: PUSHED -- {' | '.join(tail)}", flush=True)
            return 0
        if any(w in out for w in WAIT_FOR):
            why = next(w for w in WAIT_FOR if w in out)
            print(f"[{stamp}] attempt {attempt}: still blocked ({why}) -- sleeping {every:.0f} min",
                  flush=True)
        else:
            print(f"[{stamp}] attempt {attempt}: NON-RETRYABLE failure -- stopping", flush=True)
            print("\n".join(out.splitlines()[-25:]))
            return 1
        if (time.time() - t0) / 3600.0 >= hours:
            print(f"\ngiving up after {hours:.1f} h -- the slot never freed", flush=True)
            return 2
        time.sleep(every * 60)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
