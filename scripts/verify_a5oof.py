"""Acceptance gate for the A5-OOF kernel's artifacts.

Different questions from both the 0943 submission gate and the E2 gate, because this kernel's
failure modes are different:

  * The A5 stage writes `/kaggle/working/speed_a5_raw.npz` with `raw_probabilities` of shape
    [fold, study, finding] plus `applicable`.  The kernel-level "did it run" questions are cheap;
    the one that matters is whether `raw[k]` is really the prediction of the fold-k checkpoint for
    fold-k studies -- because if the cohort global were not repointed, this whole thing silently
    scores the three public TEST studies instead (the FACTS A87 failure).
  * A5 is expected to be only PARTIALLY applicable: a study whose slots cannot be decoded keeps an
    all-zero mask and is flagged, by parent policy.  So "some studies inapplicable" is normal and
    must not be reported as failure -- while "ALL studies inapplicable" means the decode pipeline
    never worked.

Usage:
    python scripts/verify_a5oof.py --raw <dir>/speed_a5_raw.npz --fold 0
"""
import argparse
import csv
import sys

import numpy as np

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--fold", default="0")
    ap.add_argument("--folds-csv", default="downloads/research/oof/v52_oof.csv")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    d = np.load(args.raw, allow_pickle=True)
    print("keys: %s" % list(d.keys()))
    uids = [str(u) for u in d["study_uids"]]
    preds = np.asarray(d["raw_probabilities"], dtype=np.float64)
    applicable = np.asarray(d["applicable"])
    print("studies            : %d" % len(uids))
    print("raw shape          : %s   (expected [5 folds, study, 12])" % (preds.shape,))
    print("applicable         : %d / %d" % (int(applicable.sum()), len(applicable)))

    ok = True
    if preds.ndim != 3 or preds.shape[0] != 5 or preds.shape[2] != len(TARGETS):
        print("  !! raw_probabilities is not [5, study, 12]")
        ok = False
    if preds.shape[1] != len(uids):
        print("  !! raw_probabilities and study_uids disagree on the study count")
        ok = False

    fold = {}
    with open(args.folds_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fold[r["StudyInstanceUID"]] = r["fold"]
    want = {k for k, v in fold.items() if str(v) == str(args.fold)}
    inside = sum(1 for u in uids if u in want)
    print("in the requested fold: %d / %d  (want all)" % (inside, len(uids)))
    if inside != len(uids):
        print("  !! NOT the requested cohort -- the cohort global was probably not repointed")
        ok = False

    if int(applicable.sum()) == 0:
        print("  !! every study is inapplicable: the decode pipeline never produced pixels")
        ok = False
    elif int(applicable.sum()) < 0.5 * len(uids):
        print("  note: fewer than half the studies are applicable -- check the decode path")

    # the fold axis must actually carry information: a fold's own model should not be identical to
    # the others, and every fold slice should be non-degenerate on the applicable studies
    ap_idx = np.where(applicable)[0]
    if len(ap_idx):
        sub = preds[:, ap_idx, :]
        dead = []
        for k in range(preds.shape[0]):
            if float(np.nanstd(sub[k])) < 1e-12:
                dead.append(k)
        print("degenerate fold slices: %s  (want none)" % (dead or "none"))
        if dead:
            ok = False
        per_fold_std = [float(np.nanstd(sub[k])) for k in range(preds.shape[0])]
        print("per-fold std       : %s" % ["%.4f" % v for v in per_fold_std])
        fin = bool(np.isfinite(sub).all())
        print("finite on applicable: %s" % fin)
        if not fin:
            ok = False

    print()
    print("VERDICT: %s" % ("A5 OOF ARTIFACTS OK" if ok else "A5 OOF ARTIFACTS BAD - do not build on these"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
