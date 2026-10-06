# -*- coding: utf-8 -*-
"""Turn a TPU run's per-worker RSS trace into per-segment growth rates.

The training loop writes one line per checkpoint to /kaggle/working/worker_<i>_mem.txt:

    <tag> rss_kb=<int> avail_kb=<int>

where <tag> is a heartbeat name, usually "fold0 r336 ep3 si17 fwd g=2 shape=(...)".
Two generations of markers exist and this script reads both:

    v17 and earlier : fwd, bwd, opt
    v18 and later   : a_rows, b_aug, c_ymw, fwd, bwd, opt

The question the v19 run exists to answer is *which segment leaks*, so the useful
output is not the RSS curve but the difference between consecutive markers, averaged
over an early window and a late window. A leak that is constant per step shows the same
number in both; the v17 trace grew from ~47 MB/step to ~178 MB/step, which is why the
acceleration matters more than the average.

Usage:
    python scripts/analyze_tpu_mem.py outputs/tpu_weights/seed2026_v17
    python scripts/analyze_tpu_mem.py outputs/tpu_weights/seed2026_v19 --window 19
    python scripts/analyze_tpu_mem.py <path-to-worker_0_mem.txt>

`_mem` records RSS, which is HOST memory. TPU HBM is 16 GB per chip and is not what
this file measures.
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
import re
import sys
from pathlib import Path

LINE = re.compile(r"^(?P<tag>.*?)\s+rss_kb=(?P<rss>-?\d+)\s+avail_kb=(?P<avail>-?\d+)\s*$")
STEP = re.compile(r"ep(?P<ep>\d+)\s+si(?P<si>\d+)\s+(?P<marker>[a-z_]+)")
MB = 1024.0  # KB per MB


def parse(path: Path):
    """[(marker, ep, si, rss_kb, avail_kb)] in file order; preamble gets ep=si=None."""
    out = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LINE.match(raw.strip())
        if not m:
            continue
        tag = m.group("tag")
        rss, avail = int(m.group("rss")), int(m.group("avail"))
        s = STEP.search(tag)
        if s:
            out.append((s.group("marker"), int(s.group("ep")), int(s.group("si")),
                        rss, avail))
        else:
            # preamble: fork_after / warmup_pre / warmup_done / smoke_pre / smoke_done
            head = tag.split()[0] if tag.split() else "?"
            out.append((head, None, None, rss, avail))
    return out


def segments(samples):
    """Attribute each consecutive RSS delta to a 'prev -> cur' segment label.

    Returns [(label, ep, si, delta_kb, ordinal)] for every step-to-step transition,
    including the transition from the previous step's last marker into this step's
    first one (that is the data-preparation segment when the first marker is fwd).

    `ordinal` counts steps across the whole run. Windows must be taken on it rather
    than on `si`, because `si` restarts at 0 every epoch and a two-epoch trace would
    otherwise report two overlapping windows as if they were the first and last third.
    """
    segs = []
    order: dict[tuple[int, int], int] = {}
    for i in range(1, len(samples)):
        prev, cur = samples[i - 1], samples[i]
        pm, _, _, prss, _ = prev
        cm, ep, si, crss, _ = cur
        if ep is None:
            continue
        order.setdefault((ep, si), len(order))
        segs.append((f"{pm} -> {cm}", ep, si, crss - prss, order[(ep, si)]))
    return segs


PREAMBLE = {"fork_after", "warmup_pre", "warmup_done", "smoke_pre", "smoke_done"}


def label_bucket(label: str) -> str:
    """Fold the raw pair label into the handful of names worth tabulating."""
    prev = label.split(" -> ")[0]
    if prev in PREAMBLE:
        # The transition out of the smoke predict into the first forward pass carries
        # the last of the one-time setup, not a per-step cost. Bucketing it as "prep"
        # would put ~115 MB that happens once into the average of the first window.
        return "startup (preamble -> first fwd)"
    table = {
        "opt -> fwd": "prep (opt->fwd, whole data path)",
        "opt -> a_rows": "prep a: slice + H2D transfer",
        "a_rows -> b_aug": "prep b: augment",
        "b_aug -> c_ymw": "prep c: m/y/w transfer",
        "c_ymw -> fwd": "prep d: model call",
        "fwd -> bwd": "forward + backward",
        "bwd -> opt": "optimizer",
    }
    return table.get(label, label)


def window_stats(segs, lo, hi, params):
    """Mean MB/step per segment label over step ordinals [lo, hi] (inclusive)."""
    acc: dict[str, list[int]] = {}
    for label, ep, si, d, o in segs:
        if lo <= o <= hi:
            acc.setdefault(label_bucket(label), []).append(d)
    rows = []
    for label in params:
        vals = acc.get(label)
        if not vals:
            continue
        rows.append((label, len(vals), sum(vals) / len(vals) / MB))
    return rows


def analyse(path: Path, window: int, verbose: bool):
    samples = parse(path)
    if not samples:
        print(f"  {path.name}: no rss lines")
        return None
    segs = segments(samples)
    steps = [(ep, si) for _, ep, si, _, _ in segs]
    if not steps:
        print(f"  {path.name}: no per-step markers")
        return None
    n = max(o for _, _, _, _, o in segs) + 1

    first_rss = samples[0][3]
    last_rss = samples[-1][3]
    avail_first = samples[0][4]
    avail_last = samples[-1][4]

    # The preamble (fork, warmup, smoke predict) allocates once and is NOT a leak, so
    # the leak rate has to be measured from the first training step rather than from the
    # top of the file. On v17 measuring from fork_after inflated the rate by ~15%
    # (118.1 vs 102.0 MB/step) simply because 2.9 GB of one-time setup got divided by
    # the step count. HANDOFF 4.3 quotes the training-phase number.
    tr0 = next(i for i, s in enumerate(samples) if s[1] is not None)
    train_first = samples[tr0][3]
    train_growth = last_rss - train_first

    print(f"\n===== {path.name} =====")
    print(f"  samples {len(samples)}   steps {n}   "
          f"epochs {sorted({ep for ep, _ in steps})}")
    print(f"  preamble    {first_rss / MB / 1024:8.2f} -> {train_first / MB / 1024:8.2f} GB"
          f"   one-time {(train_first - first_rss) / MB / 1024:+.2f} GB (not a leak)")
    print(f"  TRAINING    {train_first / MB / 1024:8.2f} -> {last_rss / MB / 1024:8.2f} GB"
          f"   growth {train_growth / MB / 1024:+.2f} GB"
          f"   {train_growth / n / MB:+.1f} MB/step  <-- the leak rate")
    print(f"  MemAvail {avail_first / MB / 1024:8.2f} GB -> {avail_last / MB / 1024:8.2f} GB"
          f"   change {(avail_last - avail_first) / MB / 1024:+.2f} GB")

    params = ["startup (preamble -> first fwd)",
              "prep (opt->fwd, whole data path)", "prep a: slice + H2D transfer",
              "prep b: augment", "prep c: m/y/w transfer", "prep d: model call",
              "forward + backward", "optimizer"]

    w = min(window, max(1, n // 3)) if window else max(1, n // 3)
    early = window_stats(segs, 0, w - 1, params)
    late = window_stats(segs, n - w, n - 1, params)

    print(f"\n  MB/step by segment          first {w} steps      last {w} steps")
    labels = [r[0] for r in early] + [r[0] for r in late]
    seen = []
    for lb in labels:
        if lb not in seen:
            seen.append(lb)
    early_d = {r[0]: r[2] for r in early}
    late_d = {r[0]: r[2] for r in late}
    for lb in seen:
        e = f"{early_d[lb]:+8.1f}" if lb in early_d else "       -"
        l = f"{late_d[lb]:+8.1f}" if lb in late_d else "       -"
        print(f"    {lb:<34} {e}         {l}")

    if verbose:
        print("\n  per-step transition trace (first 12 and last 6):")
        # On a trace that died early the two slices overlap and would print twice.
        shown = segs if len(segs) <= 18 else segs[:12] + segs[-6:]
        for label, ep, si, d, _ in shown:
            print(f"    ep{ep} si{si:<4} {label:<18} {d / MB:+8.1f} MB")

    return {"steps": n, "growth_gb": train_growth / MB / 1024,
            "mb_per_step": train_growth / n / MB,
            "preamble_gb": (train_first - first_rss) / MB / 1024,
            "early": early_d, "late": late_d}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="a seed directory, or one worker_*_mem.txt")
    ap.add_argument("--window", type=int, default=0,
                    help="steps in the early/late window (default: a third of the run)")
    ap.add_argument("--worker", default=None,
                    help="only this worker index (default: worker_0, or every file "
                         "when you pass --all-workers)")
    ap.add_argument("--all-workers", action="store_true")
    args = ap.parse_args()

    p = Path(args.path)
    if p.is_dir():
        files = sorted(p.glob("worker_*_mem.txt"))
        if not files:
            print(f"no worker_*_mem.txt under {p}", file=sys.stderr)
            return 1
        if not args.all_workers:
            pick = args.worker if args.worker is not None else "0"
            files = [f for f in files if f.name == f"worker_{pick}_mem.txt"] or files[:1]
    else:
        files = [p]

    results = {}
    for f in files:
        r = analyse(f, args.window, verbose=len(files) == 1)
        if r:
            results[f.name] = r

    if len(results) > 1:
        print("\n===== summary across workers =====")
        print(f"  {'worker':<22} {'steps':>6} {'growth GB':>10} {'MB/step':>9} "
              f"{'early prep':>11} {'late prep':>10} {'late opt':>9}")
        for name, r in results.items():
            ep_ = r["early"].get("prep (opt->fwd, whole data path)")
            lp_ = r["late"].get("prep (opt->fwd, whole data path)")
            lo_ = r["late"].get("optimizer")
            f = lambda v: f"{v:+8.1f}" if v is not None else "       -"
            print(f"  {name:<22} {r['steps']:>6} {r['growth_gb']:>10.2f} "
                  f"{r['mb_per_step']:>9.1f} {f(ep_):>11} {f(lp_):>10} {f(lo_):>9}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
