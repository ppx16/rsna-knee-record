"""Score every LABEL SOURCE we have against the 58 officially labelled studies.

WHY
    Label quality is one of the few things this project has measured to MATTER: our two label
    sets differ by +0.005 gold-58 when everything else is held fixed (HANDOFF 74: v4hybrid
    0.8470 vs rule 0.8418), and the label function itself scores 0.8991 on gold-58 against the
    0.8617 of the rule parser.  Meanwhile `downloads/labels/` holds SEVEN label files and only
    some of them have ever been scored -- the rest are sitting on disk unused.

    Scoring a label source against gold-58 costs seconds and no GPU: gold is exactly the
    train.csv rows where all twelve official columns are non-null (verified 2026-09-14), and
    every label file is keyed by StudyInstanceUID.  So this is the cheapest possible screen on
    the one axis we know moves the score.

    If any source beats v4hybrid's 0.8991 by a real margin, training on it becomes a candidate
    -- and unlike a new corpus that needs a week of work, swapping the label file is nearly
    free (a dataset version plus one argv path).

Usage:
    python scripts/label_source_audit.py
    python scripts/label_source_audit.py --dir downloads/labels --gold data/raw/train.csv
"""
from __future__ import annotations

import argparse
import glob
import os
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from cv_score import LABELS, paired_bootstrap  # noqa: E402

UID_KEYS = ("StudyInstanceUID", "study_id", "StudyId", "id", "UID", "case", "Case")


def find_uid(df: pd.DataFrame):
    for k in UID_KEYS:
        if k in df.columns:
            return k
    # first column that looks like a DICOM UID
    for c in df.columns:
        v = df[c].astype(str).head(20)
        if v.str.startswith("1.2.").all():
            return c
    return None


def label_cols(df: pd.DataFrame, uid_col: str):
    """The 12 finding columns, matched case-insensitively and tolerant of spacing."""
    norm = {c.lower().replace("_", " ").replace("-", " ").strip(): c for c in df.columns
            if c != uid_col}
    out, missing = [], []
    for want in LABELS:
        w = want.lower().replace("_", " ").replace("-", " ").strip()
        hit = norm.get(w)
        if hit is None:                       # try a looser containment match
            for k, c in norm.items():
                if w in k or k in w:
                    hit = c
                    break
        (out if hit else missing).append(hit if hit else want)
    return out, missing


def macro_auc(Y, P, idx):
    vals, per = [], []
    for j in range(12):
        yy = (Y[idx, j] >= 0.5).astype(int)
        if len(set(yy.tolist())) > 1:
            vals.append(roc_auc_score(yy, P[idx, j]))
            per.append(vals[-1])
        else:
            per.append(float("nan"))
    return (float(np.mean(vals)) if vals else float("nan")), per


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="downloads/labels")
    ap.add_argument("--gold", default="data/raw/train.csv")
    ap.add_argument("--boot", type=int, default=0, help="paired bootstrap vs --baseline if >0")
    ap.add_argument("--baseline", default=None)
    a = ap.parse_args(argv)

    gold = pd.read_csv(ROOT / a.gold)
    need = [c for c in LABELS if c in gold.columns]
    if len(need) != 12:
        print("gold file is missing label columns: %s" % [c for c in LABELS if c not in gold.columns])
        return 1
    gmask = gold[need].notna().all(axis=1)
    g = gold[gmask].set_index("StudyInstanceUID")
    print("gold studies: %d" % len(g))
    Y = g[need].to_numpy(dtype=np.float64)

    # NOTE: the label files live one directory DEEPER than expected -- downloads/labels/ holds
    # one subdirectory per public label dataset (rsna-knee-hybrid-report-labels,
    # rsna-knee-llm-report-labels, ...), so a non-recursive glob finds nothing.  That is what
    # the first run reported as "no csv under downloads/labels".
    base = pathlib.Path(ROOT) / a.dir
    files = sorted(str(p) for p in base.rglob("*.csv"))
    if not files:
        print("no csv under %s (searched recursively)" % base)
        return 1
    print("found %d csv file(s) under %s" % (len(files), base))

    rows = []
    preds = {}
    for f in files:
        name = os.path.basename(f)
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print("%-34s READ FAILED %s" % (name, e))
            continue
        uid = find_uid(df)
        if uid is None:
            print("%-34s no uid column (%s)" % (name, list(df.columns)[:6]))
            continue
        cols, missing = label_cols(df, uid)
        if len(missing) > 6:
            print("%-34s could not map %d label columns" % (name, len(missing)))
            continue
        df = df.set_index(df[uid].astype(str))
        idx, P = [], []
        for sid in g.index.astype(str):
            if sid in df.index:
                row = df.loc[sid]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                P.append([float(pd.to_numeric(row[c], errors="coerce")) if c in df.columns else np.nan
                          for c in cols])
                idx.append(True)
            else:
                idx.append(False)
        P = np.array(P, dtype=np.float64)
        cov = int(np.sum(idx))
        if cov < 20:
            print("%-34s only %d/58 gold studies present -> skipped" % (name, cov))
            continue
        m, per = macro_auc(Y, P, np.array(idx))
        preds[name] = (P, np.array(idx))
        # CONTAMINATION FLAG (HANDOFF 99): a public "merged" label file can carry the OFFICIAL
        # labels verbatim on the gold rows.  Such a file scores a perfect 1.0000 here, which
        # says nothing about its quality on the other 4349 rows -- and it must never be scored
        # against gold, nor trusted as a training target, without explaining that.
        sub = P[np.array(idx)]
        exact = float(np.isclose(sub, Y, atol=1e-9).mean())
        binary = float(np.isin(sub, (0.0, 1.0)).mean())
        flag = ""
        bad = False
        if exact > 0.999:
            flag = "  ⛔ CONTAMINATED: reproduces train.csv's official labels verbatim"
            bad = True
        elif binary > 0.999:
            flag = "  ⚠️ gold block is BINARY -- likely the official labels; score is not honest"
            bad = True
        rows.append((name, cov, m, per, bad))
        print("%-34s cov %2d/58   gold-58 macro-AUC %.4f   (exact %.3f, binary %.3f)%s"
              % (name, cov, m, exact, binary, flag))

    if rows:
        clean = [r for r in rows if not r[4]]
        dirty = [r for r in rows if r[4]]
        clean.sort(key=lambda r: (-(r[2] if r[2] == r[2] else -1)))
        print("\n== ranking (contaminated files excluded) ==")
        for name, cov, m, _, _ in clean:
            print("  %-34s %.4f  (cov %d)" % (name, m, cov))
        if dirty:
            print("\n== EXCLUDED as contaminated -- their gold score is not a measurement ==")
            for name, cov, m, _, _ in dirty:
                print("  %-34s %.4f  <- ignore this number" % (name, m))
        if clean:
            best = clean[0]
            print("\nbest clean: %s at %.4f  (v4hybrid's known value is 0.8991)"
                  % (best[0], best[2]))
            if best[2] > 0.8991 + 0.002:
                print("⭐ beats the known best by more than 0.002 -> CANDIDATE TRAINING TARGET.")
            else:
                print("   -> does not beat v4hybrid by a margin worth acting on.")

        # per-label table for the top three CLEAN sources
        top = clean[:3]
        print("\n== per-label macro-AUC, top 3 clean ==")
        print("  %-18s %s" % ("label", "  ".join("%-12s" % r[0][:12] for r in top)))
        for j, nm in enumerate(LABELS):
            line = "  %-18s" % nm
            for _, _, _, per, _ in top:
                line += "  %-12s" % ("%.4f" % per[j] if per[j] == per[j] else "n/a")
            print(line)

    if a.boot and a.baseline and a.baseline in preds:
        print("\n== paired bootstrap vs %s ==" % a.baseline)
        Pb, ib = preds[a.baseline]
        for name, cov, m, _, bad in rows:
            if name == a.baseline or bad:
                continue
            P, ii = preds[name]
            both = ib & ii
            if both.sum() < 20:
                continue
            _, _, ci, win = paired_bootstrap(Y[both], P[both], Pb[both],
                                             n_boot=a.boot, seed=3)
            print("  %-34s delta %+.4f  CI [%+.4f, %+.4f]  %s"
                  % (name, m - macro_auc(Y, Pb, ib)[0], ci[0], ci[1],
                     "REAL" if (ci[0] > 0 or ci[1] < 0) else "not resolved"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
