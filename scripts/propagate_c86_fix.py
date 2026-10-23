# -*- coding: utf-8 -*-
"""
Propagate the C86 verifier fix to all 8 corpus-builder shard copies.

Found by grepping every copy: only kernels/kernel-corpus64/build_corpus.py had the fix; all
eight kernel-corpus64-s<i>/build_corpus.py still carried the STRUCTURE-ONLY verdict
(`verdict = "OK" if np.median(c) > 0.9`). That is the same failure family as FACTS A170 ("the
load fix existed only in kernel-msft2") and A173 ("patching a file does not patch a running
process"): a fix that lands in ONE copy and not the others is not a fix.

Mechanical by design: the two replacements must each match exactly once per file, the result
must compile, and the file must not have been already patched.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

OLD1 = '''                c = corr_rows(v[keep], r[keep])
                verdict = "OK" if np.median(c) > 0.9 else "MISMATCH"
                print("  row %-5d slots %2d/%2d (comparable %2d)  corr min/med/max = %.3f / %.3f / %.3f  [%.0fs]" % (
                    row, int(m.sum()), v.shape[0], int(keep.sum()), c.min(), np.median(c), c.max(),
                    time.time() - t0), flush=True)'''

NEW1 = '''                c = corr_rows(v[keep], r[keep])
                # ---- C86 patch: the verdict was STRUCTURE-ONLY, and that is a real hole -------
                # corr_rows() mean-centres and normalises each row before correlating, so it is
                # INVARIANT to any global or monotone rescale. Measured 2026-09-25: our built
                # corpus had mean non-zero intensity 58.12 against the published 102.24 -- 43%
                # too dark, caused by min-max normalisation -- and this verifier still read
                # "FAITHFUL, median correlation 0.9813". A verifier that cannot see the defect it
                # exists to catch is worse than no verifier, because it is believed. So the
                # verdict now needs BOTH: structure (correlation) AND scale (intensity level).
                _ov = v[keep].astype(np.float64)
                _or = r[keep].astype(np.float64)
                _mv = float(_ov[_ov > 0].mean()) if (_ov > 0).any() else 0.0
                _mr = float(_or[_or > 0].mean()) if (_or > 0).any() else 0.0
                scale = (_mv / _mr) if _mr else float("nan")
                ok_corr = np.median(c) > 0.9
                ok_scale = bool(0.85 <= scale <= 1.15)      # 15% band; the observed defect was 43%
                if ok_corr and ok_scale:
                    verdict = "OK"
                elif ok_corr:
                    verdict = "OK-STRUCTURE-ONLY"           # structure right, brightness wrong
                else:
                    verdict = "MISMATCH"
                print("  row %-5d slots %2d/%2d (comparable %2d)  corr min/med/max = %.3f / %.3f / %.3f  "
                      "meanNZ ours/ref = %.2f / %.2f (scale %.3f)  [%.0fs]" % (
                          row, int(m.sum()), v.shape[0], int(keep.sum()), c.min(), np.median(c),
                          c.max(), _mv, _mr, scale, time.time() - t0), flush=True)'''

OLD2 = '''                                "corr_max": float(c.max()), "verdict": verdict,'''
NEW2 = '''                                "corr_max": float(c.max()), "verdict": verdict,
                                "meanNZ_ours": _mv, "meanNZ_ref": _mr, "scale": scale,
                                "ok_corr": bool(ok_corr), "ok_scale": ok_scale,'''


def main():
    targets = sorted(ROOT.glob("kernels/kernel-corpus64-s*/build_corpus.py"))
    if len(targets) != 8:
        print(f"WARNING: expected 8 shard copies, found {len(targets)}")
    done = 0
    for p in targets:
        src = p.read_text(encoding="utf-8")
        if "ok_scale" in src:
            print(f"  {p.parent.name}: already patched")
            continue
        for old, new, tag in ((OLD1, NEW1, "verdict block"), (OLD2, NEW2, "results dict")):
            n = src.count(old)
            if n != 1:
                raise SystemExit(f"FATAL {p.parent.name}/{tag}: anchor found {n} times, expected 1")
            src = src.replace(old, new, 1)
        p.write_text(src, encoding="utf-8")
        compile(src, str(p), "exec")            # compile(), not ast.parse
        done += 1
        print(f"  {p.parent.name}: patched + compile OK  ({len(src.encode())} bytes)")
    print(f"\n{done} shard copies patched; {len(targets)} checked")


if __name__ == "__main__":
    sys.exit(main())
