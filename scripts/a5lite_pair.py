"""Score a5lite-family OOF arms on ONE label set, paired, over a shared id intersection.

WHY THIS EXISTS (pre-registered 2026-09-26, written BEFORE the augmented pair landed)
-------------------------------------------------------------------------------------
A153 measured, on the SAME script / fold / epochs / bs / no-aug, that the frontier A5
architecture beats our head by delta = +0.0575 (95% CI [+0.0411, +0.0763], P=1.00).  But that
number was taken at a DEGRADED operating point (no augmentation, bs=2, n_slice=6), and the
project's own rule (A109/A112) says a gap measured at a broken point need not survive at a
good one.  So the transfer check re-runs the pair symmetrically with augmentation ON both
arms, and this script reads the result.

DECISION RULE, fixed before the numbers existed (state/FACTS.md 210.3):
    * delta_aug = macro(a5 + aug) - macro(r + aug), same fold, same 881 studies.
    * delta_aug >= +0.02  -> the architecture advantage transfers -> the 48 GB faithful
      6x16x336 cache is justified (the gain is not an artefact of our degraded setup).
    * delta_aug collapses (CI straddles 0, or |delta| < 0.02) -> route A stops; the ~0.05
      would then live in the 16-slice density + the frontier's recipe, NOT in the checkpoint.
No other comparison is admissible as evidence for this decision: the cross-setup comparison
d3f0 (0.8530) vs a5f0 (0.7947) is CONFOUNDED and was already read backwards once (A153).

HONESTY LIMITS baked into the printout
    * The metric is the surrogate masked-macro on report labels, not the LB.  It is internally
      consistent across arms and is the only instrument that can see a 0.002 effect; it is NOT
      the competition metric and its absolute level is not a leaderboard prediction.
    * Arms are compared on the INTERSECTION of their id sets.  Two fold systems exist in this
      project (a5lite arms: 881 studies; cn0/d3f0: 870), so a cross-family number must be read
      on the intersection, never on each arm's own full set.
    * A single seed per arm cannot separate a real architecture effect from seed noise; that is
      exactly why the pair is run symmetrically and why the decision threshold is a LARGE
      effect (+0.02), not a CI that just clears zero.

Usage:
    python scripts/a5lite_pair.py \
        --arm a5fa=downloads/a5lite_a5/raptor_oof_a5fa_fold0.npz \
        --arm rfa=downloads/a5lite_r/raptor_oof_rfa_fold0.npz \
        --arm a5f0=downloads/a5lite_a5/raptor_oof_a5f0_fold0.npz \
        --arm rf0=downloads/a5lite_r/raptor_oof_rf0_fold0.npz \
        --ref rf0 --compare a5f0,a5fa,rfa
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
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from calibrator_probe import TARGETS, macro, rank_columns, read_labels  # noqa: E402
from oof_surrogate import paired_bootstrap_masked  # noqa: E402

LABELS_CSV = ROOT / "downloads" / "research" / "oof" / "report_labels_v2.csv"
THRESHOLD = 0.02          # 210.3: fixed before the augmented numbers existed


def load_arm(path):
    d = np.load(path, allow_pickle=True)
    ids = [str(u) for u in d["ids"]]
    P = np.asarray(d["pred"], dtype=np.float64)
    if P.shape != (len(ids), 12):
        raise SystemExit("!! %s: pred %r does not match %d ids" % (path, P.shape, len(ids)))
    return ids, P


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", default=[], required=True,
                    metavar="TAG=PATH", help="repeatable")
    ap.add_argument("--ref", default="", help="reference tag for the paired bootstrap")
    ap.add_argument("--compare", default="", help="comma list of tags to test against --ref")
    ap.add_argument("--labels", default=str(LABELS_CSV))
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--space", default="raw", choices=["raw", "rank"])
    args = ap.parse_args(argv)

    arms = {}
    for spec in args.arm:
        if "=" not in spec:
            raise SystemExit("!! --arm wants TAG=PATH, got %r" % spec)
        tag, path = spec.split("=", 1)
        arms[tag] = load_arm(ROOT / path if not pathlib.Path(path).is_absolute() else path)

    labs = {}
    L = pathlib.Path(args.labels)
    if not L.is_absolute():
        L = ROOT / L
    labels_path = L if L.is_file() else None
    if labels_path is None:
        raise SystemExit("!! no label file at %s" % L)
    lids, Y0, M0 = read_labels(str(labels_path))
    for i, u in enumerate(lids):
        labs[u] = i

    # ---- intersect: every arm AND the label table must have the id ----
    sets = [set(ids) & set(labs) for ids, _ in arms.values()]
    common = set.intersection(*sets) if sets else set()
    order = sorted(common)
    print("label table : %d ids (%s)" % (len(lids), labels_path.name))
    for tag, (ids, _) in arms.items():
        print("arm %-6s : %d ids, %d in label table" % (tag, len(ids), len(set(ids) & set(labs))))
    n = len(order)
    print("intersection: %d studies  <-- ALL numbers below are on these only" % n)
    if n == 0:
        raise SystemExit("!! empty intersection")
    if len(set(len(s) for s in sets)) > 1:
        print("  (arms had different id coverage; the intersection is the honest frame)")

    Y = np.zeros((n, 12)); M = np.zeros((n, 12), dtype=bool)
    for k, u in enumerate(order):
        j = labs[u]
        Y[k] = Y0[j]; M[k] = M0[j]

    P = {}
    for tag, (ids, pred) in arms.items():
        pos = {u: i for i, u in enumerate(ids)}
        p = np.stack([pred[pos[u]] for u in order])
        P[tag] = rank_columns(p) if args.space == "rank" else p

    print("\n-- masked macro-AUC on the %d shared studies (space=%s) --" % (n, args.space))
    score = {}
    for tag in arms:
        s, per = macro(Y, M, P[tag], TARGETS)
        score[tag] = s
        print("  %-6s  %.4f" % (tag, s))
    best = max(score, key=lambda t: score[t])
    print("  best: %s (%.4f)" % (best, score[best]))

    # ---- invariant: an arm against ITSELF must be exactly 0 with a degenerate CI ----
    d0, lo0, hi0, _ = paired_bootstrap_masked(Y, M, P[best], P[best], n_boot=25, seed=0)
    ok = (d0 == 0.0) and (lo0 == 0.0) and (hi0 == 0.0)
    print("\nself-pair invariant (%s vs itself): %.4f [%.4f, %.4f] %s"
          % (best, d0, lo0, hi0, "OK" if ok else "**FAILED**"))
    if not ok:
        raise SystemExit("!! self-pair must be 0 with a degenerate CI; the bootstrap is resampling "
                         "the wrong axis -- do NOT read the numbers below")

    if args.ref:
        if args.ref not in P:
            raise SystemExit("!! --ref %r not among arms %s" % (args.ref, sorted(P)))
        tags = [t.strip() for t in args.compare.split(",") if t.strip()] or \
               [t for t in arms if t != args.ref]
        print("\n-- paired bootstrap vs %s (delta = arm - %s) --" % (args.ref, args.ref))
        for t in tags:
            if t not in P:
                print("  %-6s  SKIP (no such arm)" % t); continue
            d, lo, hi, p = paired_bootstrap_masked(Y, M, P[t], P[args.ref],
                                                   n_boot=args.boot, seed=0)
            print("  %-6s  delta %+.4f  CI [%+.4f, %+.4f]  P(>0)=%.2f  (arm %.4f vs %s %.4f)"
                  % (t, d, lo, hi, p, score[t], args.ref, score[args.ref]))

        print("\n-- 210.3 threshold check (threshold %+.2f, fixed before the numbers) --" % THRESHOLD)
        print("   NOTE this line compares whatever arms you passed.  It is the 210.3 transfer test")
        print("   ONLY for the AUGMENTED pair (a5fa vs rfa); beside an unaugmented pair it means")
        print("   nothing about the cache.  And passing it does NOT by itself justify the 48 GB")
        print("   build: the MEMBER GATE must also show the arm adds given what the stack already")
        print("   contains -- our four A-lite arms all scored weight 0 there (A163).")
        for t in tags:
            if t not in P:
                continue
            d, lo, hi, p = paired_bootstrap_masked(Y, M, P[t], P[args.ref],
                                                   n_boot=args.boot, seed=0)
            if d >= THRESHOLD:
                v = "advantage survives at THIS operating point (necessary, not sufficient)"
            elif d <= -THRESHOLD:
                v = "REVERSED -> the reference arm is better here; re-read before acting"
            else:
                v = "COLLAPSED -> no advantage at this operating point"
            print("  %-6s %s" % (t, v))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
