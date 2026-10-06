"""Score CV OOF files produced by the fold kernels.

The kernels write, into /kaggle/working (which we download into downloads/cv/<tag>/):
    raptor_oof_<tag>_fold<K>.npz            final / best-epoch OOF
    raptor_oof_<tag>_ep<e>_fold<K>.npz      OOF measured at epoch e
    raptor_oof_<tag>_epens_fold<K>.npz      mean over the top-K epochs
    raptor_oof_<tag>_swa_fold<K>.npz        SWA OOF

Every npz has pred (N,12), truth (N,12), ids (N,), fold, nfolds.
The truth is the *rule-parser soft label* (NOT the official label), so this metric is only
valid for comparing runs trained on the same label function.  See HANDOFF 09-14 snapshot.

Usage
-----
    python scripts/cv_score.py                       # list every tag + epoch curve
    python scripts/cv_score.py --dir downloads/cv
    python scripts/cv_score.py --pair cva0 cvg0      # paired bootstrap on shared ids
    python scripts/cv_score.py --tag cva0 --labels   # per-label table

The `--pair` mode is the one that answers "did this change help?": it aligns the two tags on
their common study ids and bootstraps *studies*, so the label noise that both tags share
cancels out.  That is the whole reason the CV is worth GPU time.
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
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

NAME = re.compile(r"raptor_oof_(?P<tag>.+?)_(?P<kind>ep\d+|epens|swa)?_?fold(?P<fold>\d+)\.npz$")

# Order verified 2026-09-14 against the kernel's own per-label printout in the cvg0 log:
# "ACL 0.652 / MCL 0.594 / Medial Meniscus 0.666 / Lateral Meniscus 0.773 / Medial OA 0.820 /
#  Lateral OA 0.843 / PF OA 0.723 / Effusion 0.909 / Synovitis 0.822 / Baker's 0.697 /
#  Contusion 0.650 / Fracture 0.653"  -- reproduced exactly by macro_auc on the npz.
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def parse_name(path: str):
    """Return (tag, kind, fold) for an OOF npz, or None.  gold files -> kind 'gold'."""
    stem = os.path.basename(path)
    if not stem.endswith(".npz"):
        return None
    m = re.match(r"raptor_gold_(?P<tag>.+?)_(?P<var>swa|epens|ep\d+)\.npz$", stem)
    if m:
        # The kernel also writes gold predictions for its SWA / epoch-ensemble variants.
        # Without this branch `raptor_gold_cva1_swa.npz` parses as a TAG called "cva1_swa", so a
        # `--pair cva1:final cva1:swa` gold block compared cva1's gold WITH ITSELF and printed a
        # meaningless +0.0000.  Found 2026-09-14 while reading the first landed fold.
        return m.group("tag"), "gold:" + m.group("var"), -1
    m = re.match(r"raptor_gold_(?P<tag>.+?)\.npz$", stem)
    if m:
        return m.group("tag"), "gold", -1
    m = re.match(r"raptor_oof_(?P<tag>.+?)_(?P<kind>ep\d+|epens|swa)_fold(?P<fold>\d+)\.npz$", stem)
    if m:
        return m.group("tag"), m.group("kind"), int(m.group("fold"))
    m = re.match(r"raptor_oof_(?P<tag>.+?)_fold(?P<fold>\d+)\.npz$", stem)
    if m:
        return m.group("tag"), "final", int(m.group("fold"))
    return None


def load(path: str):
    d = np.load(path)
    return d["pred"].astype(np.float64), d["truth"].astype(np.float64), np.asarray(d["ids"]).astype(str)


def soft_auc(y: np.ndarray, p: np.ndarray) -> float:
    """AUC against SOFT truth.

    The CV's `truth` is the report parser's soft label, and `macro_auc` binarises it at 0.5.
    That is the right analogue of the competition metric, but the binarisation throws away
    information and injects label noise into the EVALUATION -- a study at 0.49 and one at 0.51
    are near-identical yet counted as opposite classes.

    This is the natural generalisation: the probability that a random draw from the positive
    soft mass outranks a random draw from the negative soft mass,

        num = sum_i y_i * (mass of (1-y) with p strictly below p_i)
            + 0.5 * sum_i y_i * (mass of (1-y) tied with p_i)
        den = sum(y) * sum(1-y)

    For binary y it reduces to the ordinary AUC (ties at 0.5), so the two are comparable and
    the soft version should be the LESS NOISY instrument.  Whether it actually is gets
    measured, not assumed -- see scripts/test_soft_auc.py.
    """
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    neg = 1.0 - y
    den = y.sum() * neg.sum()
    if den <= 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ps, ys, ns = p[order], y[order], neg[order]
    below = np.concatenate([[0.0], np.cumsum(ns)[:-1]])          # strict, by sorted order
    # ties: for a run of equal p, subtract the part of `below` that belongs to the run
    num = 0.0
    i = 0
    n = len(ps)
    while i < n:
        j = i
        while j + 1 < n and ps[j + 1] == ps[i]:
            j += 1
        run_neg = ns[i:j + 1].sum()
        for k in range(i, j + 1):
            num += ys[k] * (below[i] + 0.5 * (run_neg - ns[k]))
        i = j + 1
    return float(num / den)


def macro_soft_auc(Y: np.ndarray, P: np.ndarray):
    """Same shape as macro_auc but against the soft truth; skips degenerate labels."""
    per = []
    for j in range(Y.shape[1]):
        y = Y[:, j]
        if y.max() <= 1e-9 or y.min() >= 1 - 1e-9:
            per.append(float("nan"))
        else:
            per.append(soft_auc(y, P[:, j]))
    ok = [v for v in per if not np.isnan(v)]
    return (float(np.mean(ok)) if ok else float("nan")), per


def macro_auc(Y: np.ndarray, P: np.ndarray, labels=None):
    """Macro AUC over labels that have both classes.  Returns (macro, per_label list)."""
    per = []
    for j in range(Y.shape[1]):
        yy = (Y[:, j] >= 0.5).astype(int)
        if len(set(yy.tolist())) > 1:
            per.append(roc_auc_score(yy, P[:, j]))
        else:
            per.append(float("nan"))
    keep = range(len(per)) if labels is None else [j for j in labels]
    ok = [per[j] for j in keep if not np.isnan(per[j])]
    return (float(np.mean(ok)) if ok else float("nan")), per


def collect(root: str):
    """tag -> kind -> list of (fold, path)."""
    table = defaultdict(lambda: defaultdict(list))
    for p in sorted(glob.glob(os.path.join(root, "**", "*.npz"), recursive=True)):
        parsed = parse_name(p)
        if not parsed:
            continue
        tag, kind, fold = parsed
        table[tag][kind].append((fold, p))
    return table


def group_of(tag: str) -> str:
    """cva0 / cva1 / cva2 -> 'cva'.  Folds are separate TAGS, so this is how an arm is named."""
    return tag.rstrip("0123456789") or tag


def resolve(name: str, table):
    """A name is either an exact tag, or a GROUP of tags (cva -> cva0..cva4)."""
    if name in table:
        return [name]
    hits = sorted(t for t in table if group_of(t) == name)
    return hits


def gather(table, tags, kind):
    """All (fold, path) entries of `kind` across several tags."""
    out = []
    for t in tags:
        out.extend(table[t].get(kind, []))
    return out


def pool_oof(paths, tag_label=""):
    """Concatenate OOF folds, dropping duplicate study ids (HANDOFF 81.3).

    Two tags can legitimately cover the same fold; counting a study twice would corrupt the
    table silently.  The duplicate count is returned and printed rather than swallowed.
    """
    Ys, Ps, seen, dups = [], [], set(), 0
    for fold, p in sorted(paths):
        P, Y, I = load(p)
        keep = np.array([v not in seen for v in I])
        dups += int((~keep).sum())
        for v in I[keep]:
            seen.add(v)
        if keep.any():
            Ys.append(Y[keep])
            Ps.append(P[keep])
    if not Ys:
        return None, None, None, 0
    return np.concatenate(Ys), np.concatenate(Ps), np.array(sorted(seen)), dups


def pooled_gold(paths):
    """Every fold predicts the SAME 58 gold studies.

    So they must NOT be concatenated (that would count each study F times and inflate n).
    They are aligned by id and AVERAGED -- which is the right thing anyway: five models
    trained on five different 3479-study subsets, averaged, is an ensemble, and averaging
    removes the model noise that makes a 58-study verdict so weak (HANDOFF 68.3).

    Returns (Y, P_mean, ids, per_fold_auc list).
    """
    first = load(sorted(paths)[0][1])
    ids = first[2]
    idx = {v: i for i, v in enumerate(ids)}
    Ps, per = [], []
    for fold, p in sorted(paths):
        P, Y, I = load(p)
        if I.shape == ids.shape and (I == ids).all():
            Pj, Yj = P, Y
        else:
            common = [v for v in ids if v in set(I.tolist())]
            if len(common) != len(ids):
                continue
            src = {v: i for i, v in enumerate(I)}
            sel = np.array([src[v] for v in ids])
            Pj, Yj = P[sel], Y[sel]
        Ps.append(Pj)
        per.append(macro_auc(Yj, Pj)[0])
    if not Ps:
        return first[1], first[0], ids, []
    return Yj, np.mean(Ps, axis=0), ids, per


def paired_bootstrap(Y, Pa, Pb, n_boot=400, seed=0, labels=None, soft=False):
    """Paired study-level bootstrap of macro-AUC(a) - macro-AUC(b).

    NOTE 2026-09-14: the first version of this function said `n = Y.shape[1]`, which is the
    number of LABELS (12), not the number of studies -- so every point estimate and every
    resample used only the first 12 studies.  It looked plausible (0.8343 instead of 0.7380)
    and was caught only by pairing a tag with ITSELF and noticing that the "paired" number
    disagreed with the listing number.  Invariant to test after any edit here:
        cv_score.py --dir D --pair TAG TAG   must print the SAME AUC as the listing, delta 0.
    """
    rng = np.random.default_rng(seed)
    n = Y.shape[0]                       # STUDIES, not labels
    Yb = (Y >= 0.5).astype(int)
    base_labels = [j for j in range(Y.shape[1]) if len(set(Yb[:, j].tolist())) > 1]
    if labels is not None:
        base_labels = [j for j in base_labels if j in labels]
    if not base_labels:
        return float("nan"), float("nan"), float("nan"), 0

    def macro(idx, P):
        if soft:
            # sharper instrument: measured 36% tighter paired CI than the binarised metric on a
            # known damage, with a better effect-to-CI ratio (scripts/test_soft_auc.py).
            # NOTE the two scales are NOT comparable to each other -- only within themselves.
            return macro_soft_auc(Y[idx], P[idx])[0]
        vals = []
        for j in base_labels:
            yy = Yb[idx, j]
            if len(set(yy.tolist())) > 1:
                vals.append(roc_auc_score(yy, P[idx, j]))
        return float(np.mean(vals)) if vals else float("nan")

    point_a = macro(np.arange(n), Pa)
    point_b = macro(np.arange(n), Pb)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a, b = macro(idx, Pa), macro(idx, Pb)
        if a == a and b == b:
            deltas.append(a - b)
    deltas = np.asarray(deltas)
    if deltas.size == 0:
        return point_a, point_b, float("nan"), 0
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return point_a, point_b, (lo, hi), int((deltas > 0).sum() * 100 // deltas.size)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join("downloads", "cv"))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--group", action="store_true",
                    help="group tags by arm (cva0..cva4 -> one 'cva' row pooling all folds). "
                         "Folds are SEPARATE TAGS, so this is the readout that matters once "
                         "more than one fold has landed.")
    ap.add_argument("--pair", nargs=2, default=None, metavar=("A", "B"))
    ap.add_argument("--kind", default="final",
                    help="final | epens | swa | ep<N> | gold  (default final). 'gold' lists the "
                         "gold-58 readout alone, which is all a --folds 0 member has.")
    ap.add_argument("--labels", action="store_true")
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--soft", action="store_true",
                    help="score against the SOFT truth instead of binarising at 0.5.  Measured "
                         "36%% tighter paired CI on a known damage (scripts/test_soft_auc.py) but "
                         "on a DIFFERENT scale -- never compare a soft number with a binarised one.")
    a = ap.parse_args(argv)

    if not os.path.isdir(a.dir):
        print("no such dir: %s" % a.dir)
        return 1
    table = collect(a.dir)
    if not table:
        print("no raptor_oof_*.npz under %s" % a.dir)
        return 1

    if a.pair:
        # A side may carry its own kind:  --pair cva0:final cva0:swa   (best epoch vs SWA)
        (A, ka), (B, kb) = [(s.split(":", 1) + [None])[:2] for s in a.pair]
        ka, kb = ka or a.kind, kb or a.kind
        ta, tb = resolve(A, table), resolve(B, table)
        if not ta or not tb:
            print("need both names present; have tags: %s" % ", ".join(sorted(table)))
            return 1

        def aligned(kind_a, kind_b):
            """(Ya, Pa, Yb, Pb, n) on the shared study ids, or None when unavailable."""
            pa, pb = gather(table, ta, kind_a), gather(table, tb, kind_b)
            if not pa or not pb:
                return None
            if "gold" in (kind_a, kind_b):
                # every run scores the same 58 studies -> average, never concatenate
                Ya, Pa, Ia, _ = pooled_gold(pa)
                Yb, Pb, Ib, _ = pooled_gold(pb)
            else:
                Ya, Pa, Ia, _ = pool_oof(pa)
                Yb, Pb, Ib, _ = pool_oof(pb)
            common = np.intersect1d(Ia, Ib)
            if common.size == 0:
                return None
            ia = {v: i for i, v in enumerate(Ia)}
            ib = {v: i for i, v in enumerate(Ib)}
            sa = np.array([ia[v] for v in common])
            sb = np.array([ib[v] for v in common])
            return Ya[sa], Pa[sa], Yb[sb], Pb[sb], common.size

        blocks = [("OOF", ka, kb)]
        # gold variants: raptor_gold_<tag>.npz is "gold", raptor_gold_<tag>_swa.npz is "gold:swa"
        ga = "gold" if ka in ("final", "gold") else "gold:" + ka
        gb = "gold" if kb in ("final", "gold") else "gold:" + kb
        if gather(table, ta, ga) and gather(table, tb, gb):
            blocks.append(("gold-58 (official-rubric truth)", ga, gb))
        for label, kind_a, kind_b in blocks:
            got = aligned(kind_a, kind_b)
            if got is None:
                print("%s: not available for both names" % label)
                continue
            Ya, Pa, Yb, Pb, n = got
            print("== %s -- %s:%s(%d tag%s) vs %s:%s(%d tag%s), paired on %d studies"
                  % (label, A, kind_a, len(ta), "s" if len(ta) > 1 else "",
                     B, kind_b, len(tb), "s" if len(tb) > 1 else "", n))
            pa, pb, ci, win = paired_bootstrap(Ya, Pa, Pb, n_boot=a.boot, soft=a.soft)
            print("  %-12s %.4f" % ("%s:%s" % (A, kind_a), pa))
            print("  %-12s %.4f" % ("%s:%s" % (B, kind_b), pb))
            print("  delta (A-B) = %+.4f   95%% CI [%+.4f, %+.4f]   A wins %.0f%% of bootstraps"
                  % (pa - pb, ci[0], ci[1], win))
            print("  verdict: %s" % ("REAL" if (ci[0] > 0 or ci[1] < 0)
                                     else "NOT RESOLVED (CI spans 0)"))
        return 0

    if a.group:
        names = sorted({group_of(t) for t in table})
    else:
        names = [a.tag] if a.tag else sorted(table)
    for name in names:
        tags = resolve(name, table)
        if not tags:
            continue
        _metric0 = macro_soft_auc if a.soft else macro_auc
        if a.kind == "gold":
            # `--folds 0` full-data members have NO OOF at all, so gold is the only readout they can
            # give.  Added 2026-09-24 after `--dir downloads/nxfull` printed NOTHING and exited 0,
            # which reads as "nothing landed" rather than "this member has a gold number".
            gold = gather(table, tags, "gold")
            if not gold:
                print("%-10s -- no gold files" % name)
                continue
            Yg, Pg, _, per_fold = pooled_gold(gold)
            print("%-10s kind=gold  n=%5d tags=%d  macro-AUC %.4f%s" % (
                name, len(Yg), len(tags), _metric0(Yg, Pg)[0],
                "   [per-model %s]" % " ".join("%.4f" % v for v in per_fold) if len(per_fold) > 1 else ""))
            continue
        paths = gather(table, tags, a.kind)
        if not paths:
            # NEVER skip silently.  Silence is indistinguishable from "nothing landed", and this
            # project has already been burned twice by a no-op that read as a clean result.
            if gather(table, tags, "gold"):
                print("%-10s -- no OOF for kind '%s' (gold files exist; try --kind gold)"
                      % (name, a.kind))
            else:
                print("%-10s -- no '%s' files at all" % (name, a.kind))
            continue
        Y, P, I, dups = pool_oof(paths)
        # Honour --soft in the LISTING too.  The first version passed it only to the pair block,
        # so `--dir D --group --soft` silently reprinted the binarised numbers -- exactly the kind
        # of quietly-ignored flag that gets a wrong number believed.
        _metric = macro_soft_auc if a.soft else macro_auc
        raw, per = _metric(Y, P)
        per_tag = []
        for t in tags:
            tp = table[t].get(a.kind)
            if not tp:
                continue
            Pt, Yt, _ = load(tp[0][1])
            per_tag.append("%s %.4f" % (t, _metric(Yt, Pt)[0]))
        print("%-10s kind=%-5s n=%5d tags=%d  macro-AUC %.4f%s"
              % (name, a.kind, len(Y), len(tags), raw,
                 "   [%s]" % " | ".join(per_tag) if len(tags) > 1 else ""))
        if len(tags) == 1 and per_tag:
            print("           (own fold: %s)" % per_tag[0])
        if dups:
            print("           ⚠️ dropped %d duplicate study ids while pooling" % dups)
        gold = gather(table, tags, "gold")
        if gold:
            # The gold-58 slice is the ONLY readout in the CV scored against the real
            # (official-rubric) labels rather than the rule parser's soft labels, so it is
            # the one that can in principle TRACK the public LB (HANDOFF 82.7 / 83.3).
            Yg, Pg, _, per_fold = pooled_gold(gold)
            print("           gold-58  %.4f  (n=%d%s; truth = official-rubric labels)%s"
                  % (_metric(Yg, Pg)[0], len(Yg),
                     ", %d models averaged" % len(per_fold) if len(per_fold) > 1 else "",
                     "  [per-model %s]" % " ".join("%.4f" % v for v in per_fold)
                     if len(per_fold) > 1 else ""))
        eps = sorted(((int(k[2:]), v) for t in tags for k, v in table[t].items()
                      if re.fullmatch(r"ep\d+", k)), key=lambda x: x[0])
        if eps:
            curve = []
            for e, paths_e in eps:
                Ye, Pe, _, _ = pool_oof(paths_e)
                curve.append("ep%d %.4f" % (e, _metric(Ye, Pe)[0]))
            print("           epoch curve: %s" % "  ".join(curve))
        for extra in ("epens", "swa"):
            ex = gather(table, tags, extra)
            if ex:
                Ye, Pe, _, _ = pool_oof(ex)
                print("           %-6s %.4f" % (extra, _metric(Ye, Pe)[0]))
        if a.labels:
            for j, v in enumerate(per):
                nm = LABELS[j] if j < len(LABELS) else "label%02d" % j
                print("             %-18s %s" % (nm, "  n/a" if np.isnan(v) else "%.4f" % v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
