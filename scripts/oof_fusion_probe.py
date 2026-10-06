"""Probe the OLD public-notebook OOF (downloads/oof/) as an OFFLINE FUSION instrument.

WHY
    HANDOFF §13.3 recorded that this OOF "does not track the LB", so the whole CV programme
    moved to retraining members.  But the arrays in downloads/oof/ are richer than that note
    suggests, and nobody has re-measured them since the routing result was itself questioned
    (§53: probe22's 0.941 is the MAX over 22 probes).

    What is actually in there (verified 2026-09-14):

        train_ids (4407)  train_y (4407,12)  train_fold (4407)  valid_mask (4407)
        gold_mask (4407)                         <- the 58 officially-labelled studies
        oof_arm_prob        (4407, 4, 12)        <- 4 ARMS, out-of-fold, per label
        oof_window_prob     (4407, 4, 10, 12)    <- same, before pooling over 10 windows
        oof_equal_prob      (4407, 12)
        test_member_prob    (20, 3, 12)          <- 20 members on the 3 local test rows

    `arm_labels = ['2026','20260808','31337','7717']` and `global_weight_full = [.3,.2,.3,.2]`.

    So this is a 4407-study OOF over the 4 arms -- the SAME 4 arms whose fusion weights we have
    been probing one 3-decimal LB reading at a time (`raptor_view_w`, the `arm-weights-alt`
    axis).  4407 studies is 5x the CV fold, costs no GPU, and is already on disk.

WHAT IT ANSWERS
    1. How much does the arm-weight axis move this OOF at all?  (If the spread over the whole
       simplex is ~0.001, the flat LB readings are CONFIRMED, not contradicted.)
    2. How big is MAX-SELECTION bias?  Draw N random weightings, take the max -- that is
       exactly what "22 probes, keep the best" did on the LB.  The max-minus-median gap is the
       size of the artefact we have to beat before believing any tuned gain.

Usage:
    python scripts/oof_fusion_probe.py
    python scripts/oof_fusion_probe.py --draws 400 --boot 400
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
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parent.parent
OOF = ROOT / "downloads" / "oof"
sys.path.insert(0, str(ROOT / "scripts"))
from cv_score import LABELS, paired_bootstrap, macro_soft_auc  # noqa: E402

# Switched by --soft.  The binarised metric is the competition analogue; the soft one is the
# sharper instrument (36% tighter paired CI, HANDOFF 101) and is worth re-reading the CLOSED
# conclusions with -- if a fusion effect was hidden by evaluation noise, it shows up here.
SOFT = False


def macro(Y, P, mask=None):
    if mask is not None:
        Y, P = Y[mask], P[mask]
    if SOFT:
        return macro_soft_auc(Y, P)[0]
    vals = []
    for j in range(Y.shape[1]):
        yy = (Y[:, j] >= 0.5).astype(int)
        if len(set(yy.tolist())) > 1:
            vals.append(roc_auc_score(yy, P[:, j]))
    return float(np.mean(vals)) if vals else float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--soft", action="store_true",
                    help="score against the SOFT truth (sharper instrument, HANDOFF 101). "
                         "Different scale from the binarised numbers -- compare verdicts and CIs, "
                         "never the levels.")
    a = ap.parse_args(argv)
    global SOFT
    SOFT = bool(a.soft)
    print("metric: %s" % ("SOFT truth (sharper)" if SOFT else "binarised at 0.5 (competition analogue)"))

    try:
        arms = np.load(OOF / "oof_arm_prob.npy").astype(np.float64)      # (N,4,12)
        Y = np.load(OOF / "train_y.npy").astype(np.float64)              # (N,12)
        valid = np.load(OOF / "valid_mask.npy")
        gold = np.load(OOF / "gold_mask.npy")
        names = np.load(OOF / "arm_labels.npy", allow_pickle=True)
    except FileNotFoundError as e:
        print("missing %s" % e)
        return 1

    N = Y.shape[0]
    print("OOF: %d studies x %d arms x %d labels" % (N, arms.shape[1], arms.shape[2]))
    print("  arms  : %s" % list(names))
    print("  valid : %d   gold: %d" % (int(valid.sum()), int(gold.sum())))

    def blend(w):
        return np.tensordot(arms, np.asarray(w, float), axes=([1], [0]))   # (N,12)

    equal = np.ones(arms.shape[1]) / arms.shape[1]
    P_eq = blend(equal)
    print("\n== reference: equal arm weights ==")
    print("  all-valid  macro-AUC %.4f" % macro(Y, P_eq, valid))
    print("  gold-58    macro-AUC %.4f" % macro(Y, P_eq, gold))

    # per-arm solo
    print("\n== solo arms (all-valid / gold-58) ==")
    for i, nm in enumerate(names):
        w = np.zeros(arms.shape[1])
        w[i] = 1.0
        P = blend(w)
        print("  %-10s %.4f / %.4f" % (nm, macro(Y, P, valid), macro(Y, P, gold)))

    # window pooling: the LB probes poolfluidmax / pooloamax / poolefforig all read 0.941
    # (flat).  This is the offline cross-check: do the pooling choices move a 4407-study
    # instrument at all?  If they do not, the flat LB readings are CONFIRMED rather than
    # unexplained.
    try:
        win = np.load(OOF / "oof_window_probabilities.npy").astype(np.float64)   # (N,4,10,12)
        pools = {"mean": win.mean(2), "max": win.max(2), "median": np.median(win, axis=2)}
        print("\n== window pooling over the 10 windows per study ==")
        P_mean = None
        for nm, wpool in pools.items():
            P = np.tensordot(wpool, equal, axes=([1], [0]))
            if nm == "mean":
                P_mean = P
            print("  %-8s all-valid %.4f  gold %.4f"
                  % (nm, macro(Y, P, valid), macro(Y, P, gold)))
        for nm, wpool in pools.items():
            if nm == "mean":
                continue
            P = np.tensordot(wpool, equal, axes=([1], [0]))
            _, _, ci, _ = paired_bootstrap(Y[valid], P[valid], P_mean[valid],
                                           n_boot=a.boot, seed=5)
            print("  %-8s vs mean: %+.4f  CI [%+.4f, %+.4f]  %s"
                  % (nm, macro(Y, P, valid) - macro(Y, P_mean, valid), ci[0], ci[1],
                     "REAL" if (ci[0] > 0 or ci[1] < 0) else "not resolved"))
    except FileNotFoundError:
        pass

    # ---- per-label pooling table vs the SHIPPED map (HANDOFF 74 / 87.8) ----
    # The shipped production map is not "plain mean": it sets
    #     max  -> Fracture, Contusion, Medial Meniscus, Lateral Meniscus, Baker's
    #     top2 -> ACL, MCL
    #     original_mean -> Synovitis
    #     (default plain mean) -> Medial OA, Lateral OA, PF OA, Effusion
    # So the OOF's global "max beats mean by +0.0018" is consistent with the map ALREADY
    # exploiting it.  The question this table answers is whether any label left at the
    # default has a large per-label max/top2 gain -- i.e. whether the three flat LB pooling
    # probes (poolfluidmax / pooloamax / poolefforig) were flat because there was nothing
    # there, or because the change was too small to show.
    SHIPPED = {"Fracture": "max", "Contusion": "max", "Medial Meniscus": "max",
               "Lateral Meniscus": "max", "Baker's": "max", "ACL": "top2", "MCL": "top2",
               "Synovitis": "original_mean"}
    try:
        win = np.load(OOF / "oof_window_probabilities.npy").astype(np.float64)
        print("\n== per-label pooling: what each choice is worth, label by label ==")
        print("  %-18s %-14s %8s %8s %8s   %-9s %s"
              % ("label", "shipped", "mean", "max", "top2", "shipped", "max-mean"))
        for j, nm in enumerate(LABELS):
            w = win[:, :, :, j]                          # (N, 4, 10)
            cand = {
                "mean": w.mean(2),
                "max": w.max(2),
                "top2": np.sort(w, axis=2)[:, :, -2:].mean(2),
            }
            sc = {}
            yj = (Y[valid, j] >= 0.5).astype(int)
            if len(set(yj.tolist())) < 2:
                continue
            for k, arr in cand.items():
                sc[k] = roc_auc_score(yj, np.tensordot(arr, equal, axes=([1], [0]))[valid])
            ship = SHIPPED.get(nm, "mean")
            gain = sc["max"] - sc["mean"]
            flag = "  <== LARGE, untested" if (ship == "mean" and gain > 0.002) else ""
            print("  %-18s %-14s %8.4f %8.4f %8.4f   %-9s %+.4f%s"
                  % (nm, ship, sc["mean"], sc["max"], sc["top2"], ship, gain, flag))
    except FileNotFoundError:
        pass

    # --- 1. how much does the axis move at all? ---
    rng = np.random.default_rng(0)
    draws = rng.dirichlet(np.ones(arms.shape[1]), size=a.draws)
    scores = np.array([macro(Y, blend(w), valid) for w in draws])
    lo, med, hi = np.percentile(scores, [2.5, 50, 97.5])
    print("\n== %d random weightings over the 4-simplex ==" % a.draws)
    print("  p2.5 %.4f | median %.4f | p97.5 %.4f | max %.4f | min %.4f"
          % (lo, med, hi, scores.max(), scores.min()))
    print("  spread p97.5-p2.5 = %.4f    MAX-SELECTION GAP (max-median) = %+.4f"
          % (hi - lo, scores.max() - med))
    print("  => tuning arm weights on a 3-decimal readout and keeping the best of %d draws"
          % a.draws)
    print("     would buy about %+.4f of PURE ARTEFACT on this instrument." % (scores.max() - med))

    # --- 2. the shape of the axis ---
    # NOTE ON NAMING: the LB experiments we ran (`arm-weights-alt` and friends) moved
    # `raptor_view_w`, whose 4 keys are view names (maxspan-v5 / native384dense-v10 /
    # maxspan-v5-reverse / native384-v8).  The 4 arms HERE are labelled with what look like
    # training seeds ('2026','20260808','31337','7717').  I have NOT verified that the two
    # fours are the same four, so the weight vectors below are named NEUTRALLY.  What they
    # measure is the SENSITIVITY OF A 4-WAY BLEND to weight choice on this instrument -- which
    # is the property that decides whether any such axis can be tuned at 3 decimals at all.
    print("\n== weight settings along the axis (names are positional, not claims) ==")
    cands = {
        "canonical-ish [.55,.10,.15,.20]": np.array([0.55, 0.10, 0.15, 0.20]),
        "shifted       [.60,.10,.05,.25]": np.array([0.60, 0.10, 0.05, 0.25]),
        "shifted       [.65,.05,.15,.15]": np.array([0.65, 0.05, 0.15, 0.15]),
        "shifted       [.50,.20,.15,.15]": np.array([0.50, 0.20, 0.15, 0.15]),
        "concentrated  [.70,.10,.10,.10]": np.array([0.70, 0.10, 0.10, 0.10]),
    }
    base = blend(cands["canonical-ish [.55,.10,.15,.20]"])
    b_base = macro(Y, base, valid)
    for nm, w in cands.items():
        P = blend(w)
        _, _, ci, _ = paired_bootstrap(Y[valid], P[valid], base[valid], n_boot=a.boot, seed=1)
        print("  %-32s all-valid %.4f  gold %.4f   vs row1: %+.4f CI [%+.4f, %+.4f] %s"
              % (nm, macro(Y, P, valid), macro(Y, P, gold),
                 macro(Y, P, valid) - b_base, ci[0], ci[1],
                 "REAL" if (ci[0] > 0 or ci[1] < 0) else "ns"))
    # equal weights are the honest reference: does tuning beat NOT tuning?
    _, _, ci, _ = paired_bootstrap(Y[valid], P_eq[valid], base[valid], n_boot=a.boot, seed=2)
    print("  %-32s all-valid %.4f  gold %.4f   vs row1: %+.4f CI [%+.4f, %+.4f] %s"
          % ("EQUAL (no tuning)", macro(Y, P_eq, valid), macro(Y, P_eq, gold),
             macro(Y, P_eq, valid) - b_base, ci[0], ci[1],
             "REAL" if (ci[0] > 0 or ci[1] < 0) else "ns"))

    # --- 3. per-label (routed) weights vs one global vector -- the §13.3 question, at n=4407 --
    def routed(arms_sub, W):
        """Per-label arm weights: W is (n_arms, 12) (transposed if stored the other way)."""
        W = np.asarray(W, float)
        if W.shape[0] != arms.shape[1]:
            W = W.T
        assert W.shape == (arms.shape[1], arms.shape[2]), "weight shape %s" % (W.shape,)
        W = W / np.maximum(W.sum(0, keepdims=True), 1e-9)
        P = np.zeros((arms_sub.shape[0], arms_sub.shape[2]))
        for k in range(arms_sub.shape[2]):
            P[:, k] = arms_sub[:, :, k] @ W[:, k]
        return P

    tgt = OOF / "target_weight_full.npy"
    if tgt.exists():
        P_r = routed(arms, np.load(tgt))
        _, _, ci, _ = paired_bootstrap(Y[valid], P_r[valid], P_eq[valid], n_boot=a.boot, seed=3)
        print("\n== per-label (routed) arm weights vs EQUAL -- the §13.3 question ==")
        print("  routed  all-valid %.4f  gold %.4f" % (macro(Y, P_r, valid), macro(Y, P_r, gold)))
        print("  equal   all-valid %.4f  gold %.4f" % (macro(Y, P_eq, valid), macro(Y, P_eq, gold)))
        print("  delta (routed - equal) = %+.4f  95%% CI [%+.4f, %+.4f]  %s"
              % (macro(Y, P_r, valid) - macro(Y, P_eq, valid), ci[0], ci[1],
                 "REAL" if (ci[0] > 0 or ci[1] < 0) else "NOT RESOLVED"))
        print("  ⚠️ `target_weight_full` was fitted ON this same OOF, so its number is")
        print("     optimistic by construction -- the per-fold version below is the honest one.")
    tgf = OOF / "target_weight_by_meta_fold.npy"
    if tgf.exists():
        Wf = np.load(tgf)                                    # (5,4,12) fitted per fold
        folds = np.load(OOF / "train_fold.npy")
        P_rf = np.full_like(P_eq, np.nan)
        for f in range(Wf.shape[0]):
            m = (folds == f)
            if m.sum() == 0:
                continue
            P_rf[m] = routed(arms[m], Wf[f])
        _, _, ci, _ = paired_bootstrap(Y[valid], P_rf[valid], P_eq[valid], n_boot=a.boot, seed=4)
        print("  per-FOLD routed (weights fitted out-of-fold) all-valid %.4f  gold %.4f"
              % (macro(Y, P_rf, valid), macro(Y, P_rf, gold)))
        print("  delta vs equal = %+.4f  95%% CI [%+.4f, %+.4f]  %s"
              % (macro(Y, P_rf, valid) - macro(Y, P_eq, valid), ci[0], ci[1],
                 "REAL" if (ci[0] > 0 or ci[1] < 0) else "NOT RESOLVED"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
