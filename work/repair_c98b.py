# -*- coding: utf-8 -*-
"""REPAIR: undo the broken C98-b insertion, then apply it correctly.

What went wrong: the C98-b patch inserted executable statements INSIDE the `summary = { ... }`
dict literal, which is a SyntaxError. The previous script wrote before it compiled, so the
broken text is on disk. This restores the original line first (getting back to a file that
compiles), then does the job properly: compute the scale aggregates BEFORE the dict and only
put dict entries inside it.

Verified by compile() on every file. Reverting first is what makes this safe to run.
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

BROKEN = '''        # ---- C98-b: the HEADLINE verdict must be scale-aware too -------------------------
        # It used to be computed from `meds` (correlations) alone, so a corpus that is 43% too
        # dark still summarised as "FAITHFUL" -- which is precisely the string C86 cites as the
        # check that missed the defect. Fixing the per-row verdict without fixing this one left
        # the headline just as blind.
        _scales = [r["scale"] for r in results if r.get("scale") == r.get("scale")]
        median_scale = float(np.median(_scales)) if _scales else float("nan")
        _ok_corr_all = bool(meds) and np.median(meds) > 0.9
        _ok_scale_all = bool(_scales) and 0.85 <= median_scale <= 1.15
        "verdict": ("FAITHFUL" if (_ok_corr_all and _ok_scale_all)
                    else ("FAITHFUL-STRUCTURE-ONLY" if _ok_corr_all else "MISMATCH")),
        "median_scale": median_scale,
        "ok_scale": _ok_scale_all,'''

ORIGINAL = '''        "verdict": ("FAITHFUL" if meds and np.median(meds) > 0.9 else "MISMATCH"),'''

# the line that opens the summary dict; we compute the aggregates just before it
ANCHOR_OLD = '''        meds = [r["corr_med"] for r in results if "corr_med" in r]'''
ANCHOR_NEW = '''        meds = [r["corr_med"] for r in results if "corr_med" in r]
        # ---- C98-b: the HEADLINE verdict must be scale-aware too -------------------------
        # It used to come from `meds` (correlations) alone, so a corpus 43% too dark still
        # summarised as "FAITHFUL" -- the very string C86 cites as the check that missed the
        # defect. Fixing the per-row verdict without fixing the headline left it just as blind.
        # Computed HERE, before the dict literal: statements cannot live inside a dict.
        _scales = [r["scale"] for r in results if r.get("scale") == r.get("scale")]
        median_scale = float(np.median(_scales)) if _scales else float("nan")
        ok_corr_all = bool(meds) and np.median(meds) > 0.9
        ok_scale_all = bool(_scales) and 0.85 <= median_scale <= 1.15'''

FINAL_NEW = '''        "verdict": ("FAITHFUL" if (ok_corr_all and ok_scale_all)
                    else ("FAITHFUL-STRUCTURE-ONLY" if ok_corr_all else "MISMATCH")),
        "median_scale": median_scale,
        "ok_scale": ok_scale_all,'''


def main():
    targets = sorted(ROOT.glob("kernels/kernel-corpus64*/build_corpus.py"))
    fixed = 0
    for p in targets:
        src = p.read_text(encoding="utf-8")
        if BROKEN in src:
            src = src.replace(BROKEN, ORIGINAL, 1)
            print(f"  {p.parent.name}: reverted the broken insertion")
        if "FAITHFUL-STRUCTURE-ONLY" in src:
            print(f"  {p.parent.name}: already correct")
            continue
        for old, new, tag in ((ANCHOR_OLD, ANCHOR_NEW, "pre-dict computation"),
                              (ORIGINAL, FINAL_NEW, "verdict line")):
            n = src.count(old)
            if n != 1:
                raise SystemExit(f"FATAL {p.parent.name}/{tag}: found {n}, expected 1")
            src = src.replace(old, new, 1)
        compile(src, str(p), "exec")          # compile BEFORE writing, this time
        p.write_text(src, encoding="utf-8")
        fixed += 1
        print(f"  {p.parent.name}: patched + compile OK (written)")
    print(f"\n{fixed} patched, {len(targets)} checked")


if __name__ == "__main__":
    main()
