# -*- coding: utf-8 -*-
"""Report what is inside one or more weights packages, and compare them.

A training run writes weights/{manifest.json, m_*.pt}. Whether a new run is worth
keeping is decided by its members' `holdout` against the members already banked, so the
question this answers is always the same: which members do I have, what did each score,
and does the new package beat the old one on the same folds.

Usage:
    python scripts/manifest_report.py downloads/weights-out-224/weights
    python scripts/manifest_report.py A B            # compare two packages
    python scripts/manifest_report.py A B --csv      # machine-readable

`holdout` is the AUC the member scored on the fold it did NOT train on, so members are
only comparable when their `config` (img / slices / variant) matches; a package that
mixes configs is reported in separate groups rather than averaged together.
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
import json
import statistics
import sys
from pathlib import Path


def find_manifest(p: Path) -> Path:
    if p.is_file():
        return p
    hits = sorted(p.rglob("manifest.json"), key=lambda q: len(q.parts))
    if not hits:
        raise SystemExit(f"no manifest.json under {p}")
    return hits[0]


def load(p: Path):
    mp = find_manifest(p)
    man = json.loads(mp.read_text(encoding="utf-8"))
    members = man.get("members", [])
    missing = [m["file"] for m in members if not (mp.parent / m["file"]).is_file()]
    return mp, man, members, missing


def cfg_key(m: dict) -> str:
    c = m.get("config", {})
    return (f"{c.get('img')}px x{c.get('slices')} {c.get('variant', '?')} "
            f"{str(c.get('backbone', '?')).split('/')[-1]}")


def fmt(v, width=7, prec=4):
    if v is None:
        return " " * width
    try:
        return f"{float(v):.{prec}f}".rjust(width)
    except (TypeError, ValueError):
        return str(v).rjust(width)


def pkg_label(manifest_path: str) -> str:
    """A short readable name for a package: the run directory, not the literal 'weights'.

    Every training run saves into a directory called `weights/`, so on its own the
    parent name is identical across packages and useless in a comparison table. Paths are
    made absolute first, otherwise a relative argument like `weights` resolves its parent
    to `.` and the label comes out as a bare dot.
    """
    d = Path(manifest_path).resolve().parent
    if d.name in ("weights", "output"):
        d = d.parent
    try:
        rel = str(d.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(d)
    return d.name if rel in (".", "") else rel


def report(path: Path, csv: bool):
    mp, man, members, missing = load(path)
    if not members:
        print(f"{mp}: no members")
        return None
    groups: dict[str, list[dict]] = {}
    for m in members:
        groups.setdefault(cfg_key(m), []).append(m)

    if not csv:
        print(f"\n===== {mp} =====")
        print(f"  format {man.get('format')}   {len(members)} member(s)"
              + (f"   MISSING FILES: {len(missing)}" if missing else ""))
        for g, ms in sorted(groups.items()):
            hs = [m["holdout"] for m in ms if m.get("holdout") is not None]
            an = [m["annot"] for m in ms if m.get("annot") is not None]
            print(f"\n  config: {g}")
            print(f"    {'id':<12}{'fold':>5}{'seed':>6}{'ep':>4}{'holdout':>9}{'annot':>8}")
            for m in sorted(ms, key=lambda x: (x.get("fold", -1), x.get("seed", 0))):
                print(f"    {m['id']:<12}{m.get('fold', -1):>5}{m.get('seed', 0):>6}"
                      f"{m.get('epochs_done', 0):>4}{fmt(m.get('holdout')):>9}"
                      f"{fmt(m.get('annot')):>8}")
            if hs:
                print(f"    holdout  mean {statistics.fmean(hs):.4f}  "
                      f"min {min(hs):.4f}  max {max(hs):.4f}  n={len(hs)}")
            if an:
                print(f"    annot    mean {statistics.fmean(an):.4f}  n={len(an)}")
    if missing:
        print(f"  !! {len(missing)} member file(s) absent: {missing[:3]}")
    return {"path": str(mp), "members": members, "groups": groups}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="package dir, manifest.json, or run output")
    ap.add_argument("--csv", action="store_true", help="one row per member")
    args = ap.parse_args()

    results = [r for r in (report(Path(p), args.csv) for p in args.paths) if r]

    if args.csv:
        print("package,config,id,fold,seed,epochs_done,holdout,annot")
        for r in results:
            for m in r["members"]:
                print(f"{r['path']},{cfg_key(m)},{m['id']},{m.get('fold')},"
                      f"{m.get('seed')},{m.get('epochs_done')},"
                      f"{m.get('holdout')},{m.get('annot')}")
        return 0

    if len(results) > 1:
        print("\n===== comparison =====")
        for r in results:
            lbl = pkg_label(r["path"])
            for g, ms in sorted(r["groups"].items()):
                hs = [m["holdout"] for m in ms if m.get("holdout") is not None]
                if not hs:
                    continue
                print(f"  {lbl:<22} {g:<28} n={len(hs):<3} "
                      f"mean holdout {statistics.fmean(hs):.4f}")

        # Paired comparison. Fold membership is fixed by an md5 of the report text, so
        # it does not move with resolution or backbone: fold k holds out the SAME
        # studies in every run of this corpus. That makes the same-fold members a paired
        # sample, and the per-fold delta a far more sensitive test than two means -
        # a small real gain is visible fold by fold long before it moves the mean, and a
        # fold that swings the wrong way is visible instead of being averaged away.
        maps = []
        for r in results:
            if len(r["groups"]) != 1:
                maps.append((pkg_label(r["path"]), None))
                continue
            d = {m.get("fold"): m.get("holdout") for m in r["members"]
                 if m.get("holdout") is not None}
            maps.append((pkg_label(r["path"]), d))

        paired = [(l, d) for l, d in maps if d]
        if len(paired) == 2:
            (la, da), (lb, db) = paired
            shared = sorted(set(da) & set(db))
            if len(shared) >= 2:
                print(f"\n  paired per-fold holdout (same folds = same held-out studies)")
                print(f"    {'fold':>4}  {la:>12}  {lb:>12}  {'delta':>9}")
                deltas = []
                for f in shared:
                    dlt = db[f] - da[f]
                    deltas.append(dlt)
                    print(f"    {f:>4}  {da[f]:>12.4f}  {db[f]:>12.4f}  {dlt:>+9.4f}")
                ma, mb = statistics.fmean(da[f] for f in shared), \
                    statistics.fmean(db[f] for f in shared)
                print(f"    {'mean':>4}  {ma:>12.4f}  {mb:>12.4f}  {mb - ma:>+9.4f}")
                wins = sum(1 for d in deltas if d > 0)
                print(f"    folds improved: {wins}/{len(shared)}   "
                      f"worst {min(deltas):+.4f}   best {max(deltas):+.4f}")

        print("\n  A new package is worth banking when its mean holdout beats the banked "
              "one on the SAME config; different img/slices are different experiments, "
              "not an improvement.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
