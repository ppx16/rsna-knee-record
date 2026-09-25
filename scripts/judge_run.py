# -*- coding: utf-8 -*-
"""Judge a finished training run in one command: the read-only half of HANDOFF 2.0.

At terminal time the sequence in HANDOFF 2.0 is five steps (confirm terminal, find the
package, read the real epoch counts, take the epoch-matched baseline, do the paired
per-fold comparison). Run by hand under time pressure that is five chances to skip the
step that matters - and the step that matters most is easy to skip, because the naive
comparison (this run's holdout against the banked 20-epoch mean) is wrong whenever the
per-fold time budget cut the run short.

This does all five and prints a verdict and the next step. It is READ-ONLY: it never
pushes, submits, or edits metadata.

    python scripts/judge_run.py                 # the v4 defaults
    python scripts/judge_run.py --log <path>    # use an already-downloaded log

Reuses scripts/epoch_curve.py and scripts/manifest_report.py rather than re-implementing
their parsing, so there is one definition of each format.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import epoch_curve  # noqa: E402
import manifest_report  # noqa: E402

KERNEL = "ppxl16/rsna-knee-train-v1"
BASELINE_DIR = "downloads/weights-out-224/weights"
BASELINE_LOG = "downloads/weights-out-224/rsna-knee-train-v1.log"


def status_of(ref: str) -> str:
    p = subprocess.run(["kaggle", "kernels", "status", ref],
                       capture_output=True, text=True,
                       env={**os.environ, "PYTHONUTF8": "1"})
    m = re.search(r"KernelWorkerStatus\.([A-Z]+)", p.stdout or "")
    return m.group(1) if m else "UNKNOWN"


def find_local_manifest(cands):
    for c in cands:
        p = Path(c)
        if p.is_dir():
            try:
                mp, man, members, missing = manifest_report.load(p)
                return mp, man, members, missing
            except SystemExit:
                continue
    return None


def fetch_log(dest: Path) -> Path | None:
    """kaggle kernels logs -> UTF-8 without BOM (the CLI's output is JSON lines)."""
    p = subprocess.run(["kaggle", "kernels", "logs", KERNEL],
                       capture_output=True, text=True)
    text = p.stdout or ""
    if "stream_name" not in text:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    return dest


def epochs_from_log(path: Path):
    """{fold: {epoch: holdout}} plus the members the log reported saving."""
    curves: dict[int, dict[int, float]] = {}
    saved = []
    fold = None
    for d in epoch_curve.lines(path):
        m = epoch_curve.FOLD.search(d)
        if m:
            fold = int(m.group(1))
            curves.setdefault(fold, {})
            continue
        m = epoch_curve.EPOCH.search(d)
        if m and fold is not None:
            curves[fold][int(m.group(1))] = float(m.group(4))
            continue
        m = epoch_curve.MEMBER.search(d)
        if m:
            saved.append((m.group(1), float(m.group(3))))
    return curves, saved


def curve_means(curves):
    if not curves:
        return {}
    hi = max(max(c) for c in curves.values())
    out = {}
    for ep in range(1, hi + 1):
        vals = [curves[f][ep] for f in curves if ep in curves[f]]
        if vals:
            out[ep] = sum(vals) / len(vals)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", default=KERNEL)
    ap.add_argument("--log", default=None, help="already-downloaded training log")
    ap.add_argument("--baseline-dir", default=BASELINE_DIR)
    ap.add_argument("--baseline-log", default=BASELINE_LOG)
    ap.add_argument("--candidates", default=None,
                    help="comma-separated dirs to look for the new package in")
    ap.add_argument("--force", action="store_true",
                    help="judge even if the kernel is not terminal. Exists so the judging "
                         "path can be TESTED against a known-good log before the real run "
                         "finishes - an untested judge is worse than none")
    args = ap.parse_args()

    print("=== judge_run ===")
    st = status_of(args.kernel) if not args.force else "FORCED"
    print(f"  kernel : {args.kernel}")
    print(f"  status : {st}")
    if not args.force:
        if st not in ("COMPLETE", "ERROR", "CANCELLED"):
            print("\n  not terminal yet - nothing to judge. "
                  "(The watchers download on their own.)")
            return 0
        if st != "COMPLETE":
            print(f"\n  run ended {st}: no members were produced. "
                  f"Read the log for the cause.")
            return 0

    cands = (args.candidates.split(",") if args.candidates else
             ["outputs/gpu_weights/v4_224x12/weights",
              "outputs/gpu_weights/v4_224x12"])
    pkg = find_local_manifest(cands)
    if pkg:
        mp, man, members, missing = pkg
        print(f"  package: {mp}  ({len(members)} members"
              + (f", {len(missing)} MISSING)" if missing else ")"))
    else:
        print("  package: not downloaded locally yet (fine - the log is enough to judge)")

    log = Path(args.log) if args.log else None
    if log is None:
        for c in ("downloads/v4-train.log",):
            if Path(c).is_file():
                log = Path(c)
                break
    if log is None:
        log = fetch_log(Path("downloads/v4-train.log"))
    if log is None or not log.is_file():
        print("\n  could not obtain the training log - cannot judge the epoch counts.")
        return 1
    print(f"  log    : {log}")

    curves, saved = epochs_from_log(log)
    means = curve_means(curves)
    if not means:
        print("\n  the log has no per-epoch lines. If the run died early, read it directly.")
        return 1

    eps = {f: max(c) for f, c in curves.items() if c}
    per_fold = ", ".join(f"f{f}={e}" for f, e in sorted(eps.items()))
    print(f"\n  epochs actually run : {per_fold}")
    print(f"  best epoch overall  : {max(means)}")
    if saved:
        print(f"  members saved       : {len(saved)}")
        for mid, h in saved:
            print(f"      {mid}  holdout {h:.4f}")

    # Epoch-matched baseline: the honest comparison when the budget cut the run short.
    bl = Path(args.baseline_log)
    if bl.is_file():
        bcurves, _ = epochs_from_log(bl)
        bmeans = curve_means(bcurves)
        target = max(means)
        row = []
        for ep in (3, 5, 10, 15, 20):
            if ep in bmeans:
                row.append(f"ep{ep}={bmeans[ep]:.4f}")
        print(f"\n  baseline 224x6 small, epoch-matched table: {'  '.join(row)}")
        if target in bmeans:
            print(f"  -> at {target} epochs the baseline scores {bmeans[target]:.4f}"
                  f"   <-- compare against THIS")
        if target != 20 and 20 in bmeans:
            print(f"  -> at 20 epochs (its own final)          {bmeans[20]:.4f}")
        print("  comparing against the 20-epoch mean would measure the time budget, "
              "not the idea")
    else:
        print(f"\n  baseline log {bl} not found - cannot epoch-match")

    if pkg:
        mp, man, members, missing = pkg
        print(f"\n  --- paired comparison vs {args.baseline_dir} ---")
        subprocess.run([sys.executable, str(HERE / "manifest_report.py"),
                        args.baseline_dir, str(mp.parent)])

    print("\n=== verdict / next step ===")
    print("  Decide with HANDOFF 2.5:")
    print("   * clearly better than the epoch-matched baseline -> large pays; re-run the")
    print("     SAME config with EPOCHS set to what actually fits (~5) so OneCycleLR")
    print("     anneals inside the same 8h budget.")
    print("   * at or below it -> switch to VARIANT=small at 224x12, which reaches ~17")
    print("     epochs in the same budget AND isolates the slice change from the backbone.")
    print("  Then: prep_inference_kernel.py --kernel-source " + args.kernel +
          " --variant <v> --apply")
    return 0


if __name__ == "__main__":
    sys.exit(main())
