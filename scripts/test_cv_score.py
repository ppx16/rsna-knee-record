"""Regression test for scripts/cv_score.py -- the instrument every CV decision rides on.

WHY THIS EXISTS
    On 2026-09-14 the first version of `paired_bootstrap` used `n = Y.shape[1]`, i.e. the
    number of LABELS (12) instead of the number of STUDIES.  Every point estimate and every
    resample therefore used only the first 12 studies.  It printed a plausible-looking number
    (0.8343 where the truth was 0.7380) and would have silently produced nonsense deltas for
    every experiment.  It was caught only by pairing a tag with ITSELF and noticing that the
    "paired" number disagreed with the listing number.

    That is HANDOFF §43.4 all over again: the verification was not itself verified.  So the
    checks live in a file now.

WHAT IT ASSERTS (all on synthetic copies of the real cvg0 npz, in .testtmp/)
    1. self-pair       : --pair T T  ==  the listing number, delta exactly 0
    2. big damage      : shuffling 25% of one arm's per-label predictions must read REAL-negative
    3. tiny noise      : 10% of studies nudged by ~1e-3 must read NOT RESOLVED
    4. gold path       : the same three checks on the gold-58 npz
    5. bootstrap spread: the CI must be non-degenerate when the arms differ (a bootstrap that
                         always returns [0,0] would pass checks 1 and 2 but be useless)

Usage:
    python scripts/test_cv_score.py
"""
from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
TMP = ROOT / ".testtmp" / "cvscore"
SRC_OOF = ROOT / "downloads" / "g0" / "raptor_oof_cvg0_fold0.npz"
SRC_GOLD = ROOT / "downloads" / "g0" / "raptor_gold_cvg0.npz"


def run(*args) -> str:
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "cv_score.py"), *args],
                       capture_output=True, text=True, cwd=str(ROOT))
    return (p.stdout or "") + (p.stderr or "")


def auc_of(text: str, tag: str) -> float:
    """Pull the AUC printed for `tag` out of a --pair block.

    The label is `TAG:KIND` since the kind-qualified pair syntax was added, so match on the
    tag prefix rather than on the whole first field.
    """
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and (parts[0] == tag or parts[0].startswith(tag + ":")):
            try:
                return float(parts[1])
            except ValueError:
                pass
    raise AssertionError("no AUC line for %s in:\n%s" % (tag, text))


def listing_auc(tag: str) -> float:
    text = run("--dir", str(TMP))
    for line in text.splitlines():
        parts = line.split()
        if parts and parts[0] == tag and "macro-AUC" in line:
            return float(line.split("macro-AUC")[1].split()[0])
    raise AssertionError("no listing line for %s in:\n%s" % (tag, text))


def make(tag: str, src: pathlib.Path, mode: str, rng) -> None:
    d = dict(np.load(src))
    P = d["pred"].astype(np.float64)
    n = P.shape[0]
    if mode == "shuffle":
        idx = rng.permutation(n)[: max(1, n // 4)]
        for j in range(P.shape[1]):
            P[idx, j] = P[rng.permutation(idx), j]
    elif mode == "noise":
        idx = rng.permutation(n)[: max(1, n // 10)]
        P[idx] += rng.normal(0, 1e-3, size=P[idx].shape)
    elif mode == "identity":
        pass
    else:
        raise ValueError(mode)
    d["pred"] = P.astype(np.float32)
    np.savez(TMP / ("raptor_%s_%s%s.npz" % ("gold" if "gold" in src.name else "oof", tag,
                                            "" if "gold" in src.name else "_fold0")), **d)


def main() -> int:
    if SRC_OOF.exists() is False or SRC_GOLD.exists() is False:
        print("SKIP: need downloads/g0/raptor_oof_cvg0_fold0.npz and raptor_gold_cvg0.npz")
        return 1
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    rng = np.random.default_rng(7)
    for tag, mode in (("t1", "identity"), ("t2", "shuffle"), ("t3", "noise"), ("t4", "identity")):
        make(tag, SRC_OOF, mode, rng)
        make(tag, SRC_GOLD, mode, rng)

    fails = []

    # 1. self-pair
    txt = run("--dir", str(TMP), "--pair", "t1", "t1")
    got, want = auc_of(txt, "t1"), listing_auc("t1")
    ok = abs(got - want) < 1e-9 and "+0.0000" in txt
    print("%-58s %s  (paired %.4f vs listing %.4f)" % ("1. self-pair == listing", "OK" if ok else "FAIL", got, want))
    if not ok:
        fails.append(1)

    # 2. big damage must be REAL and negative
    txt = run("--dir", str(TMP), "--pair", "t1", "t2", "--boot", "300")
    a, b = auc_of(txt, "t1"), auc_of(txt, "t2")
    ok = a > b and "REAL" in txt.split("== OOF")[1].split("== gold")[0]
    print("%-58s %s  (%.4f vs %.4f)" % ("2. 25%% shuffled reads REAL-negative", "OK" if ok else "FAIL", a, b))
    if not ok:
        fails.append(2)

    # 3. tiny noise must NOT resolve
    txt = run("--dir", str(TMP), "--pair", "t1", "t3", "--boot", "300")
    seg = txt.split("== OOF")[1]
    ok = "NOT RESOLVED" in seg
    print("%-58s %s" % ("3. 10%% x 1e-3 noise is NOT RESOLVED", "OK" if ok else "FAIL"))
    if not ok:
        fails.append(3)

    # 4. gold path present and scored on 58 studies
    ok = "gold-58" in txt and "paired on 58 studies" in txt
    print("%-58s %s" % ("4. gold-58 block present (n=58)", "OK" if ok else "FAIL"))
    if not ok:
        fails.append(4)

    # 5. non-degenerate CI + a real null (t1 vs t4 are byte-identical copies)
    ci = [l for l in txt.splitlines() if "95% CI" in l][0]
    null = run("--dir", str(TMP), "--pair", "t1", "t4", "--boot", "100")
    ok = ("[+0.0000, +0.0000]" not in ci) and ("+0.0000" in null)
    print("%-58s %s" % ("5. CI non-degenerate when arms differ", "OK" if ok else "FAIL"))
    if not ok:
        fails.append(5)

    # 6. group mode: t1..t4 are four COPIES of the same 870 studies, so pooling them must
    #    drop 3*870 duplicate ids and still report n=870 -- never n=3480 (HANDOFF 81.3).
    grp = run("--dir", str(TMP), "--group")
    row = [l for l in grp.splitlines() if l.startswith("t ")]
    ok = bool(row) and "n=  870" in row[0] and "dropped 2610 duplicate" in grp
    print("%-58s %s  (%s)" % ("6. group pooling dedups ids", "OK" if ok else "FAIL",
                              row[0].strip() if row else "no group row"))
    if not ok:
        fails.append(6)

    print("\nVERDICT: %s" % ("all 6 checks passed" if not fails else "FAILED %s" % fails))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
