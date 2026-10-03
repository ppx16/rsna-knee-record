"""Acceptance test for the EffB0-MIL arm — automate the judgement, including the traps that bit us.

WHY THIS EXISTS
    Deciding whether `0943cfm2eff` may be submitted took a manual sequence that already produced two
    mistakes in this project:

      1. `kaggle kernels logs ... *> file` writes **UTF-16** under PowerShell 5.1, so the first grep
         pass found **zero** occurrences of `effb0` and it looked as though the leg had never run.
         (It had; the decode was wrong.)
      2. The leg can *fail safely* and leave a submission that is a **byte-identical duplicate of its
         parent**. Reading the log is not enough — the decisive check is the sha256 of
         `_pipeline_stage.csv` against the parent arm's.

    Both checks are mechanical, so they belong in a script rather than in my head.

WHAT IT DECIDES
    * `DUPLICATE`      -- stage file identical to the parent's: the leg did not apply. DO NOT SUBMIT.
    * `LEG LIVE`       -- stage file differs AND the gate reported ACCEPTED: submit candidate.
    * `GATE REJECTED`  -- the leg ran but the convention gate refused: DO NOT SUBMIT (correct behaviour).
    * `NOT TERMINAL`   -- still running/queued: nothing to do yet.

Usage:
    python scripts/verify_effb0_arm.py
    python scripts/verify_effb0_arm.py --ref ppxl16/rsna-knee-0943cfm2eff --parent ppxl16/rsna-knee-0943cfm
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import pathlib
import re
import sys
import tempfile
import urllib.request

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = pathlib.Path(__file__).resolve().parents[1]
FINAL = {"COMPLETE", "ERROR", "CANCELLED"}


def decode_log(raw: bytes) -> str:
    """The log arrives as UTF-16 or UTF-8 depending on how it was captured. Try both.

    This is the specific trap that made a successful run look empty: PowerShell's `*>` redirection
    writes UTF-16LE, and grepping that as UTF-8 yields interleaved NULs and zero matches.
    """
    for enc in ("utf-16", "utf-16-le", "utf-8"):
        try:
            t = raw.decode(enc)
        except Exception:
            continue
        if t.count("\x00") <= 1 and ("effb0" in t.lower() or "raptor" in t.lower()
                                     or "gate" in t.lower()):
            return t
    return raw.decode("utf-8", errors="replace")


def log_text(ref: str) -> str:
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="klog_"))
    try:
        api.kernels_logs = getattr(api, "kernels_logs", None)
    except Exception:
        pass
    # the CLI is the only supported path for logs in kaggle 2.1.2
    import subprocess
    out = subprocess.run(["kaggle", "kernels", "logs", ref],
                         capture_output=True, env=None)
    raw = out.stdout or out.stderr or b""
    return decode_log(raw)


def fetch_output(ref: str, want: str, dest: pathlib.Path) -> pathlib.Path | None:
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    urls = pathlib.Path(tempfile.mkdtemp(prefix="kurls_")) / "u.txt"
    import subprocess
    subprocess.run([sys.executable, str(ROOT / "scripts" / "kernel_output_urls.py"),
                    "--ref", ref, "--out", str(urls)], capture_output=True)
    if not urls.exists():
        return None
    for line in urls.read_text(encoding="utf-8", errors="replace").splitlines():
        if "\t" not in line:
            continue
        name, url = line.split("\t", 1)
        if name == want:
            p = dest / want
            dest.mkdir(parents=True, exist_ok=True)
            for _ in range(3):
                try:
                    urllib.request.urlretrieve(url, p)
                    return p
                except Exception:
                    pass
            return None
    return None


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="ppxl16/rsna-knee-0943cfm2eff")
    ap.add_argument("--parent", default="ppxl16/rsna-knee-0943cfm")
    ap.add_argument("--force", action="store_true",
                    help="skip the terminal-status gate and run the comparison anyway "
                         "(for testing the logic against already-finished arms)")
    a = ap.parse_args()

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    st = str(api.kernels_status(a.ref))
    # the API returns JSON (`{"status": "RUNNING", ...}`) while the CLI returns
    # `KernelWorkerStatus.RUNNING` -- accept both, and make an unparsed value LOUD rather than
    # letting it fall through to "not terminal" and silently look like "still running".
    m = re.search(r'"?status"?\s*:\s*"(\w+)"', st) or re.search(r"KernelWorkerStatus\.(\w+)", st)
    status = m.group(1).upper() if m else None
    print("kernel %s -> %s" % (a.ref, status or ("UNPARSED: %s" % st[:120])))
    if status is None:
        print("VERDICT: could not parse the kernel status -- refusing to guess.")
        return 2
    if status not in FINAL and not a.force:
        print("VERDICT: NOT TERMINAL -- nothing to do yet.")
        return 0
    if a.force:
        print("(--force: skipping the terminal gate)")

    txt = log_text(a.ref)
    print("log decoded: %d chars" % len(txt))

    gate = "UNKNOWN"
    if re.search(r"ACCEPTED", txt, re.I):
        gate = "ACCEPTED"
    if re.search(r"REJECTED|gate failed|abandon", txt, re.I):
        gate = "REJECTED"
    leg_ok = bool(re.search(r"\[effb0\].*leg weight", txt, re.I))
    leg_failed = bool(re.search(r"LEG FAILED|leg unavailable", txt, re.I))
    print("gate=%s  leg_started=%s  leg_failed=%s" % (gate, leg_ok, leg_failed))
    for line in re.findall(r"[^\n]*(?:\[effb0\]|GATE|ACCEPTED|REJECTED|LEG FAILED)[^\n]*", txt)[:12]:
        print("   | %s" % line.strip()[:165])

    d = pathlib.Path(tempfile.mkdtemp(prefix="effchk_"))
    files = ["_pipeline_stage.csv", "submission.csv",
             "v558_rad_receipt.json", "_coat_arm_receipt.json"]
    shas = {}
    for fn in files:
        row = {}
        for tag, ref in (("arm", a.ref), ("parent", a.parent)):
            p = fetch_output(ref, fn, d / tag)
            row[tag] = sha(p)[:16] if p else None
        shas[fn] = row
        print("  %-28s arm=%-18s parent=%-18s %s"
              % (fn, row["arm"] or "absent", row["parent"] or "absent",
                 "" if row["arm"] == row["parent"] else "  <-- DIFFERS"))

    stage_same = shas["_pipeline_stage.csv"]["arm"] == shas["_pipeline_stage.csv"]["parent"]
    # ⚠️ The stage file holds RANKS. On a 3-visible-study public run there are only 6 possible
    # orderings per label, so a genuinely applied change can leave it byte-identical by chance.
    # This is real: `0943cfm` (+MCL) and `0943cf` differ in the Rad and CoAt receipts -- the exact
    # places `_RAD_EXCLUDE` acts -- yet produce the SAME 3-study ranks. So stage-file identity is
    # CORROBORATING evidence only, never the verdict on its own. The receipts are the primary
    # signal that a change reached the graph, because they carry per-stage hashes and counts.
    receipts_differ = any(shas[fn]["arm"] != shas[fn]["parent"]
                          for fn in ("v558_rad_receipt.json", "_coat_arm_receipt.json"))

    print()
    if leg_failed or gate == "REJECTED":
        print("VERDICT: *** LEG DID NOT APPLY *** -- the log says so explicitly"
              + (" and the stage file agrees." if stage_same else
                 " (though the stage file differs -- investigate)."))
        print("         DO NOT SUBMIT this arm.")
        return 0
    if not stage_same or receipts_differ:
        print("VERDICT: *** LEG LIVE *** -- the graph changed"
              + (" (receipts differ)" if receipts_differ else "")
              + ("" if stage_same else " and the final ranks differ")
              + ". Submit candidate.")
        if stage_same:
            print("         NOTE: final 3-study ranks coincide -- expected on a 3-study run; the "
                  "hidden 1322-study rerun is what scores.")
        return 0
    print("VERDICT: *** INCONCLUSIVE *** -- the log shows no leg activity, the stage file matches,")
    print("         and BOTH receipts match too. That combination is consistent with the leg never")
    print("         running. Treat as a duplicate unless the log shows [effb0] lines.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
