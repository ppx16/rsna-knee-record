"""Acceptance gate for the E2-OOF kernel's diagnostics.

Why this is a separate gate from the 0943 one: the 0943 gate asks "20/20 fingerprints and a
submission.csv", which this kernel satisfies even when it is completely wrong.  The first two smoke
runs did exactly that -- all 20 fingerprints matched, the fail-closed gate passed, and every saved
member prediction was three all-zero rows for the public test studies, because `build_cache` takes
its cohort from a global that still held test.csv's 3 studies (FACTS A87).

So the questions here are different, and each one is a way that run failed:
  1. Are there exactly 20 member files?
  2. Do they all cover the SAME studies?
  3. Are those studies the ones we asked for -- i.e. in the requested fold, NOT the public test set?
  4. Are the values finite, and NOT degenerate (all-zero / constant)?
  5. Are members actually different from each other (a shared prefix must not collapse them)?

A gate that cannot observe 3, 4 and 5 would have called a fully broken run a success.

Usage:
    python scripts/verify_e2oof.py --dir downloads/research/e2oof_smoke --fold 0
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="a directory containing diagnostics/dino_*.npz")
    ap.add_argument("--fold", default="0")
    ap.add_argument("--folds-csv", default="downloads/research/oof/v52_oof.csv")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    diag = os.path.join(args.dir, "diagnostics")
    if not os.path.isdir(diag):
        diag = args.dir
    files = sorted(glob.glob(os.path.join(diag, "dino_*.npz")))
    print("member files: %d  (want 20)" % len(files))

    ok = True
    if len(files) != 20:
        print("  !! expected exactly 20 member files")
        ok = False

    fold = {}
    with open(args.folds_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fold[r["StudyInstanceUID"]] = r["fold"]
    want = {k for k, v in fold.items() if str(v) == str(args.fold)}

    uid_sets, mats, ids0 = [], [], None
    for p in files:
        d = np.load(p, allow_pickle=True)
        uids = [str(u) for u in d["study_uids"]]
        vals = np.asarray(d["values"], dtype=np.float64)
        if ids0 is None:
            ids0 = uids
        uid_sets.append(set(uids))
        mats.append(vals)

    if not mats:
        print("VERDICT: FAIL - no member predictions at all")
        return 1

    same = all(s == uid_sets[0] for s in uid_sets)
    print("all members cover the same studies: %s" % same)
    if not same:
        ok = False

    n = len(ids0)
    in_fold = sum(1 for u in ids0 if u in want)
    print("studies per member: %d" % n)
    print("in the requested fold : %d / %d  (want all)" % (in_fold, n))
    if in_fold != n:
        print("  !! these are NOT the requested cohort -- the cache cohort global was probably not")
        print("     repointed, or the public TEST studies leaked in (FACTS A87)")
        ok = False

    stack = np.stack(mats)                      # [member, study, label]
    finite = bool(np.isfinite(stack).all())
    print("all values finite     : %s" % finite)
    if not finite:
        ok = False

    flat = stack.reshape(len(mats), -1)
    spread = flat.std(axis=0)
    dead = int((stack.std(axis=1).reshape(len(mats), -1).max(axis=1) < 1e-9).sum())
    print("degenerate members    : %d  (want 0)" % dead)
    if dead:
        ok = False

    # members must not be copies of one another
    dup = 0
    for i in range(len(mats)):
        for j in range(i + 1, len(mats)):
            if np.allclose(mats[i], mats[j], atol=1e-9):
                dup += 1
    print("identical member pairs: %d  (want 0)" % dup)
    if dup:
        ok = False

    print("value range           : %.4f .. %.4f" % (stack.min(), stack.max()))
    print()
    print("VERDICT: %s" % ("E2 OOF ARTIFACTS OK" if ok else "E2 OOF ARTIFACTS BAD - do not build on these"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
