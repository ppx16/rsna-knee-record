"""Submit a pre-registered ARM, with a fresh BEFORE snapshot and the read rule written down FIRST.

WHY THIS IS GENERALISED AND NOT A COPY. The reader leg got its own one-off `submit_rdr.py`. That pattern
does not survive contact with a second arm: the next arm means another copy, and a copy is where the
pre-registered rule quietly stops matching the thing it was registered for. One script, one place where the
BEFORE snapshot and the read rule are recorded.

⚠️ IT IS A DRY RUN UNLESS YOU PASS --fire. A submission costs one of five daily slots and is irreversible in
the sense that matters: the public score is revealed, which is exactly the information a pre-registration
exists to keep out of the decision. Default-safe is the only defensible default.

⚠️ IT REFUSES TO SUBMIT A KERNEL THAT IS NOT COMPLETE. `competition_submit_code` re-runs the notebook
server-side, so submitting an unfinished or errored kernel spends a slot to learn nothing.

Usage:
    python work/submit_arm.py --kernel ppxl16/rsna-knee-0943cfmsft2-rdrtta --arm 0943cfmsft2-rdrtta
    python work/submit_arm.py --kernel ... --arm ... --fire
"""
from __future__ import annotations




# --- GBK CONSOLE GUARD (see work/patch_gbk_guard.py) -------------------------------------------------
# This console is cp936, which cannot encode the U+26A0 warning sign used below. An unguarded
# `print("... \u26a0 ...")` raises UnicodeEncodeError at that line and kills the script MID-REPORT, while
# everything printed earlier is already on screen -- so the run looks complete. Self-contained on purpose:
# it imports `sys` itself and depends on nothing else in this tree.
import sys as _sys

try:
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001  -- non-reconfigurable stream: leave it alone, prints still degrade
    pass
# ----------------------------------------------------------------------------------------------------
import argparse
import datetime as dt
import json
import pathlib
import sys
import time

from kaggle.api.kaggle_api_extended import KaggleApi

COMP = "rsna-knee-abnormality-detection"
STATE = pathlib.Path("state")


def snapshot_now() -> dict | None:
    """Re-run the board snapshot so BEFORE is never a remembered number."""
    import subprocess

    subprocess.run([sys.executable, "work/lb_snapshot.py"], capture_output=True, text=True)
    snaps = sorted(STATE.glob("lb_snapshot_*.json"))
    if not snaps:
        return None
    return json.loads(snaps[-1].read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True, help="owner/slug of the pushed kernel")
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--arm", required=True, help="short arm name, used for the state file")
    ap.add_argument("--message", default="", help="submission description (pick the pair by THIS text later)")
    ap.add_argument("--file-name", default="submission.csv", help="the name the KERNEL wrote")
    ap.add_argument("--fire", action="store_true", help="actually submit; without it this is a dry run")
    a = ap.parse_args()

    api = KaggleApi()
    _last = None
    for _t in range(6):
        try:
            api.authenticate()
            break
        except Exception as e:  # noqa: BLE001
            _last = e
            print("auth attempt %d failed: %s" % (_t + 1, type(e).__name__), file=sys.stderr)
            time.sleep(8)
    else:
        print("COULD NOT AUTHENTICATE (%r) -- TRANSPORT, not a result." % (_last,), file=sys.stderr)
        return 2

    st = ""
    try:
        st = str(getattr(api.kernels_status(a.kernel), "status", ""))
    except Exception as e:  # noqa: BLE001
        print("could not read kernel status (%s) -- refusing to guess" % type(e).__name__, file=sys.stderr)
        return 2
    print("kernel %s status: %s" % (a.kernel, st))
    if not st.endswith("COMPLETE"):
        print("REFUSING: the kernel is not COMPLETE. Submit only a verified arm.", file=sys.stderr)
        return 3

    before = snapshot_now()
    print("=== BEFORE ===")
    print(json.dumps(before, indent=2) if before else "(no snapshot available -- comparison will be weak)")

    rec = {
        "at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "arm": a.arm,
        "kernel": a.kernel,
        "version": a.version,
        "before_snapshot": before,
        "message": a.message,
        "note": "server-side re-run; the 9 h cap is the risk",
    }

    if not a.fire:
        print()
        print("DRY RUN -- nothing submitted. The record that WOULD be written:")
        print(json.dumps({k: v for k, v in rec.items() if k != "before_snapshot"}, indent=2))
        print()
        print("⚠️ Before firing, confirm the pre-registered read rule for THIS arm is already written down.")
        print("   Re-run with --fire to spend one of the five daily slots.")
        return 0

    print("\nsubmitting %s v%d ..." % (a.kernel, a.version))
    resp = api.competition_submit_code(file_name=a.file_name, message=a.message, competition=COMP,
                                       kernel=a.kernel, kernel_version=a.version, quiet=True)
    print("response:", resp)
    ref = getattr(resp, "ref", None) or (resp.get("ref") if isinstance(resp, dict) else None)
    rec["ref"] = ref
    print("SUBMISSION REF:", ref)
    out = STATE / ("submit_%s.json" % (ref or "unknown"))
    out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print("recorded ->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
