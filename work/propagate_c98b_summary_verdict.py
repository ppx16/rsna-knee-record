# -*- coding: utf-8 -*-
"""
C98-b: the SUMMARY verdict was still scale-blind.

The per-row verdict was fixed to require structure AND scale, but the headline number that
gets quoted -- "FAITHFUL, median correlation 0.9813" -- is computed further down from `meds`
alone:

    "verdict": ("FAITHFUL" if meds and np.median(meds) > 0.9 else "MISMATCH"),

So verify44's headline could STILL read FAITHFUL for a 43%-too-dark corpus. That string is
exactly what C86 quotes as the check that missed the defect. Same failure family as the fix
that landed in one copy only: the repair went into the per-row path and not the aggregate.

This applies the aggregate fix to the main builder and all 8 shard copies, compiling each.
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

OLD = '''        "verdict": ("FAITHFUL" if meds and np.median(meds) > 0.9 else "MISMATCH"),'''

NEW = '''        # ---- C98-b: the HEADLINE verdict must be scale-aware too -------------------------
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


def main():
    targets = sorted(ROOT.glob("kernels/kernel-corpus64*/build_corpus.py"))
    done = 0
    for p in targets:
        src = p.read_text(encoding="utf-8")
        if "FAITHFUL-STRUCTURE-ONLY" in src:
            print(f"  {p.parent.name}: already patched")
            continue
        n = src.count(OLD)
        if n != 1:
            raise SystemExit(f"FATAL {p.parent.name}: anchor found {n} times, expected 1")
        src = src.replace(OLD, NEW, 1)
        p.write_text(src, encoding="utf-8")
        compile(src, str(p), "exec")
        done += 1
        print(f"  {p.parent.name}: patched + compile OK")
    print(f"\n{done} of {len(targets)} patched")


if __name__ == "__main__":
    main()
