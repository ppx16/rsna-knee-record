"""Rank the nine `0943*` arms by MEASURED runtime, and break the runtime into stages.

WHY THIS EXISTS
    `eff_rank.py` established that our efficiency board row carries
    `DateSubmitted = Sat Sep 26 05:24:23 2026`, which is submission `56570144` (`0943f`) -- while our
    LATEST submission at that moment was `56599159`. Many submissions since Sep 26 did not move that
    date. The only reading consistent with that is that **the efficiency board tracks a SELECTED
    submission**, i.e. one of the two ticks.

    That makes the final tick choice a JOINT decision over both tracks, and the efficiency metric is
    dominated by runtime:
        Efficiency = AUC / (Benchmark - max AUC) + RuntimeSeconds / 32400      (lower is better)
    with `Benchmark = 0.5` measured and the denominator NEGATIVE, so more AUC is better and less
    runtime is better, and **1 hour = 0.0510 AUC-equivalent**. Meanwhile 308 teams share our 0.943 and
    the best of them is eff#14 against our eff#3388 -- the entire gap to them is runtime, at identical
    accuracy.

    So "which of the nine arms should we tick" is not only an accuracy question. This script extracts
    each arm's real stage timeline from its own downloaded log and puts them side by side.

WHAT IT READS
    state/logs_<arm>.json -- Kaggle kernel logs as a list of {stream_name, time, data} records. The
    interesting content is the notebook's own `[  123.4s] label` progress lines.

⚠️ Read the output with two caveats, both of which the numbers will try to hide:
  1. These runs cover **3 visible studies**. Anything that LOADS checkpoints is fixed overhead here and
     will be amortised on the hidden 1322-study rerun; anything that LOOPS over studies is the part that
     actually scales. The `scale` column is a guess at which is which, and it is the single most
     important unmeasured quantity in the project (see docs/EFFICIENCY-PLAN.md Step 1).
  2. A tie in total runtime is NOT a tie in the scaling term. `0943c`/`0943cf` add a fifth Raptor view
     (A143 measured the Raptor stage 28.9 s -> 33.2 s, +15%), so their per-study cost is genuinely
     higher even if the 3-study total looks similar.

Usage:
    python scripts/arm_timing.py
    python scripts/arm_timing.py --arm 0943cf --verbose
"""
from __future__ import annotations

import argparse
import io
import json
import pathlib
import re
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = pathlib.Path(__file__).resolve().parent.parent
STATE = ROOT / "state"

STAMP = re.compile(r"^\[\s*([0-9]+(?:\.[0-9]+)?)s\]\s*(.*)$")

# (label printed in the table, regex, does it loop over studies?)
MARKERS = [
    ("data_ok", r"test g1: COMPLETE", False),
    ("gate", r"fail-closed gate PASS", False),
    ("dino", r"DINOv2.*(done|complete)", True),
    ("rad", r"rad_legacy_second: COMPLETE", True),
    ("raptor", r"raptor.*(COMPLETE|done)", True),
    ("coat", r"coat.*(COMPLETE|done)", True),
    ("submit", r"submission\.csv", False),
]


def parse(path: pathlib.Path):
    # ⚠️ Tolerant on purpose, and the tolerance is not hypothetical: 5 of the 10 logs in state/ are NOT
    # parseable JSON, because they were captured from a CLI call whose *stderr was merged into stdout*,
    # so they begin with
    #     Warning: Looks like you're using an outdated `kaggle` version (installed: 2.1.2)...
    # which makes `json.load` fail at line 1 column 1 and silently drops those arms from the comparison.
    # Find the payload instead of assuming the file starts with it.
    text = io.open(path, encoding="utf-8-sig", errors="replace").read()
    idx = text.find('[{"stream_name"')
    if idx < 0:
        idx = text.find("[{")
    if idx < 0:
        return None, "NO JSON PAYLOAD FOUND"
    try:
        recs = json.loads(text[idx:])
    except Exception as e:
        return None, "UNREADABLE (%s: %s)" % (type(e).__name__, str(e)[:70])
    if not isinstance(recs, list) or not recs:
        return 0.0, []
    total = max((r.get("time") or 0.0) for r in recs if isinstance(r, dict))
    stamps = []
    for r in recs:
        if not isinstance(r, dict):
            continue
        body = r.get("data") or ""
        for ln in body.splitlines():
            m = STAMP.match(ln.strip())
            if m:
                stamps.append((float(m.group(1)), m.group(2).strip()))
    return total, stamps


def configs(path: pathlib.Path) -> str:
    """The notebook's own `[config]` lines -- the only in-log record of preset / routing / weights.

    This matters because the entire reason the recommended tick pair is (`0943cf`, `0943p`) is a
    ROUTING HEDGE: `0943cf` carries the published per-label `coatnet_w` while `0943p` flattens it, and
    the published routing is the one knob whose provenance the frontier notebook itself calls
    public-fitted (A122). Reading the config back is how we confirm which arm is which side of that
    hedge, rather than trusting the arm names.
    """
    text = io.open(path, encoding="utf-8-sig", errors="replace").read()
    idx = text.find('[{"stream_name"')
    if idx < 0:
        idx = text.find("[{")
    if idx < 0:
        return "NO PAYLOAD"
    try:
        recs = json.loads(text[idx:])
    except Exception:
        return "BAD JSON"
    body = "".join(r.get("data", "") for r in recs if isinstance(r, dict))
    got = [ln.strip() for ln in body.splitlines() if ln.strip().startswith("[config]")]
    if not got:
        # Fall back to the coatnet_w print, which some arms log separately from [config].
        got = [ln.strip() for ln in body.splitlines() if "coatnet_w" in ln and len(ln) < 200][:2]
    return " || ".join(g[:160] for g in got) or "(none)"


def markers(stamps):
    out = {}
    for name, rx, _scales in MARKERS:
        hits = [t for t, lab in stamps if re.search(rx, lab, re.I)]
        if hits:
            out[name] = max(hits)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="", help="show one arm's full stamp list")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--configs", action="store_true",
                    help="print each arm's [config] line -- which side of the routing hedge it is on")
    a = ap.parse_args()

    logs = sorted(STATE.glob("logs_0943*.json"))
    logs = [p for p in logs if p.name != "logs_0943_probe.json"]
    if not logs:
        print("no state/logs_0943*.json found")
        return 1

    if a.arm:
        want = a.arm.replace("logs_", "").replace(".json", "")
        for p in logs:
            if p.stem == "logs_" + want:
                total, stamps = parse(p)
                print("%s  total=%.1f s  (%d stamps)" % (p.name, total, len(stamps)))
                for t, lab in stamps:
                    print("  %8.1f  %s" % (t, lab[:120]))
                return 0
        print("no log for arm %r" % a.arm)
        return 1

    if a.configs:
        for p in logs:
            print("%-10s %s" % (p.stem.replace("logs_", ""), configs(p)))
        return 0

    print("arm      total_s   " + "".join("%9s" % n for n, _r, _s in MARKERS))
    print("-" * (19 + 9 * len(MARKERS)))
    rows = []
    for p in logs:
        total, stamps = parse(p)
        arm = p.stem.replace("logs_", "")
        if total is None:
            print("%-8s %8s   %s" % (arm, "-", stamps))
            continue
        mk = markers(stamps)
        rows.append((total, arm, mk))
        print("%-8s %8.1f   " % (arm, total)
              + "".join("%9s" % ("%.0f" % mk[n] if n in mk else "-") for n, _r, _s in MARKERS))
    print("-" * (19 + 9 * len(MARKERS)))
    if rows:
        fastest = min(rows)[1]
        print("\nfastest arm by 3-study wall clock: %s (%.1f s)" % (fastest, min(rows)[0]))
        print("spread: %.1f s (%.1f%% of the slowest)"
              % (max(r[0] for r in rows) - min(r[0] for r in rows),
                 100.0 * (max(r[0] for r in rows) - min(r[0] for r in rows))
                 / max(r[0] for r in rows)))
    print("\n⚠️ 3 visible studies. Checkpoints/fingerprints are FIXED here and amortise on the hidden")
    print("   1322-study rerun; only the per-study stages scale, and they are not separated by this")
    print("   table. The per-study slope is still UNMEASURED -- docs/EFFICIENCY-PLAN.md Step 1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
