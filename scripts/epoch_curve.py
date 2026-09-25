# -*- coding: utf-8 -*-
"""Extract the per-epoch holdout curve from a training log, per fold.

Why this exists: a run saves one member per fold, and the member records only its BEST
holdout and how many epochs it got. When a run is cut short by its per-fold time budget
it banks members trained for fewer epochs, and those score lower for that reason alone.
Comparing a 5-epoch member against a 20-epoch baseline therefore measures the budget,
not the idea. The 224x6 baseline's log carries the whole curve, so the honest comparison
is against the baseline's holdout AT THE SAME EPOCH - which is what this prints.

Usage:
    python scripts/epoch_curve.py downloads/weights-out-224/rsna-knee-train-v1.log
    python scripts/epoch_curve.py <log> --at 5,10,15,20
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

FOLD = re.compile(r"fold (\d+): train (\d+) / holdout (\d+) studies")
EPOCH = re.compile(r"epoch (\d+)/(\d+)\s+loss ([\d.]+)\s+holdout ([\d.]+)"
                   r"(?:\s+annot\(n=(\d+)\) ([\d.]+))?")
MEMBER = re.compile(r"saved member ([0-9a-f]+) -> (\S+) \(holdout ([\d.]+)\)")


def lines(path: Path):
    """Yield the stdout payload of a kaggle JSON-lines log.

    Read as utf-8-sig, not utf-8: the log may have been captured with
    `kaggle kernels logs ... | Out-File -Encoding UTF8`, and PowerShell 5.1 writes a BOM
    that plain utf-8 would keep as a leading \\ufeff - which stops the first line from
    starting with "{" and silently drops it.
    """
    for raw in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        raw = raw.strip().lstrip(",")
        if not raw.startswith("{"):
            continue
        try:
            rec = json.loads(raw)
        except ValueError:
            continue
        if rec.get("stream_name") == "stdout" and "data" in rec:
            yield rec["data"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--at", default="1,3,5,10,15,20",
                    help="epochs to tabulate across folds")
    args = ap.parse_args()

    path = Path(args.log)
    if not path.is_file():
        raise SystemExit(f"no such log: {path}")

    curves: dict[int, dict[int, float]] = {}
    fold = None
    members = []
    n_epochs_target = None
    for d in lines(path):
        m = FOLD.search(d)
        if m:
            fold = int(m.group(1))
            curves.setdefault(fold, {})
            continue
        m = EPOCH.search(d)
        if m and fold is not None:
            n_epochs_target = int(m.group(2))
            curves[fold][int(m.group(1))] = float(m.group(4))
            continue
        m = MEMBER.search(d)
        if m:
            members.append((m.group(1), float(m.group(3))))

    if not curves:
        raise SystemExit("no epoch lines found - is this a training log?")

    folds = sorted(curves)
    max_ep = max(max(c) for c in curves.values())

    print(f"\n===== {path.name} =====")
    print(f"  folds {len(folds)}, epochs up to {max_ep}"
          + (f" (target {n_epochs_target})" if n_epochs_target else ""))

    print(f"\n  holdout by epoch")
    print("    ep  " + "".join(f"{('f' + str(f)):>9}" for f in folds) + f"{'mean':>9}")
    for ep in range(1, max_ep + 1):
        vals = [curves[f][ep] for f in folds if ep in curves[f]]
        if not vals:
            continue
        row = f"    {ep:>3} " + "".join(
            (f"{curves[f][ep]:>9.4f}" if ep in curves[f] else f"{'-':>9}") for f in folds)
        row += f"{sum(vals) / len(vals):>9.4f}"
        if ep == max_ep:
            row += "  <- final epoch"
        print(row)

    want = [int(x) for x in args.at.split(",") if x.strip().isdigit()]
    print(f"\n  epoch-matched means (use these against a run cut short by its budget)")
    print(f"    {'epoch':>6}{'mean holdout':>14}{'folds present':>15}")
    for ep in want:
        vals = [curves[f][ep] for f in folds if ep in curves[f]]
        if vals:
            print(f"    {ep:>6}{sum(vals) / len(vals):>14.4f}{len(vals):>15}")

    if members:
        print(f"\n  saved members")
        print(f"    {'id':<12}{'holdout':>9}")
        for i, (mid, h) in enumerate(members):
            print(f"    {mid:<12}{h:>9.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
