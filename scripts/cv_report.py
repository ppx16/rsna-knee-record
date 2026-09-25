"""One command for everything we do when a CV fold lands.

WHY (user feedback 2026-09-14: "经常忘记之前做过的事")
    The post-landing procedure used to live in my head and in §82.8: download the outputs,
    score them, check gate 1, read the epoch curve, test SWA/epens, then run the arm-join test.
    Anything that lives only in a procedure I have to REMEMBER gets skipped after a compaction.
    So the procedure is a script now, and it prints its own DECISION block.

    It also fixes the tag/fold naming trap: every fold is its OWN tag (cva0, cva1, ...) with
    `fold` always 0 inside the npz, so the human-readable unit is the GROUP (cva), not the tag.

Usage:
    python scripts/cv_report.py --group cva --download        # fetch + full report
    python scripts/cv_report.py --group cva                   # report on what is already local
    python scripts/cv_report.py --group cva --no-join         # skip the arm-join step
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
CV = os.path.join(ROOT, "downloads", "cv")


def sh(args, timeout=1800):
    p = subprocess.run(args, capture_output=True, text=True, timeout=timeout, cwd=ROOT)
    return ((p.stdout or "") + (p.stderr or "")).strip()


def ref_for(tag: str) -> str:
    """cva0 -> ppxl16/rsna-knee-cv-a-fold0 ; cvn0 -> ... (best effort, printed for checking)."""
    m = re.match(r"^cv([a-z]+)(\d+)$", tag)
    return "ppxl16/rsna-knee-cv-%s-fold%s" % (m.group(1), m.group(2)) if m else ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="cva", help="cva / cvg / cva-norm ... (tag prefix)")
    ap.add_argument("--download", action="store_true",
                    help="pull each tag's kernel output into downloads/cv/<tag> first")
    ap.add_argument("--gate", default="cvg", help="the control group for gate 1")
    ap.add_argument("--no-join", action="store_true")
    ap.add_argument("--boot", type=int, default=800)
    a = ap.parse_args(argv)

    if a.download:
        # ask Kaggle for the kernels whose tag starts with the group prefix
        rc = sh([ "kaggle", "kernels", "list", "--user", "ppxl16"])
        refs = sorted(set(re.findall(r"(ppxl16/rsna-knee-cv-[a-z0-9-]+fold\d+)", rc)))
        want = [r for r in refs if ("-cv-%s" % a.group[2:] if a.group.startswith("cv") else a.group) in r]
        print("== downloading %d kernel(s) matching the group ==" % len(want))
        for r in want:
            tag = r.split("rsna-knee-cv-")[1].replace("-fold", "")
            dst = os.path.join(CV, tag)
            os.makedirs(dst, exist_ok=True)
            out = sh(["kaggle", "kernels", "output", r, "-p", dst])
            print("  %-58s %s" % (r, out.splitlines()[-1][:70] if out else "(no output)"))

    have = sorted({os.path.basename(p) for p in glob.glob(os.path.join(CV, "*")) if os.path.isdir(p)})
    print("\n== local tag dirs under downloads/cv ==")
    print("  " + (", ".join(have) if have else "(none)"))

    print("\n" + "=" * 72)
    print("A. SCORE TABLE (grouped: folds are separate tags)")
    print("=" * 72)
    print(sh([PY, os.path.join("scripts", "cv_score.py"), "--dir", "downloads/cv", "--group"]))

    base = a.group
    print("\n" + "=" * 72)
    print("B. GATE 1 -- the 12-epoch arm must be clearly above the 1-epoch control")
    print("=" * 72)
    print(sh([PY, os.path.join("scripts", "cv_score.py"), "--dir", "downloads/cv",
              "--pair", base, a.gate, "--boot", str(a.boot)]))

    print("\n" + "=" * 72)
    print("C. PRE-REGISTERED WITHIN-RUN CHECKS (HANDOFF 85.3)")
    print("=" * 72)
    for kind, why in (("swa", "weight averaging over the top-K epochs"),
                      ("epens", "probability mean over the top-K epochs")):
        print("-- %s (%s)" % (kind, why))
        print(sh([PY, os.path.join("scripts", "cv_score.py"), "--dir", "downloads/cv",
                  "--pair", "%s:final" % base, "%s:%s" % (base, kind), "--boot", str(a.boot)]))
        # same comparison on the SHARPER instrument: soft truth, measured 36% tighter paired CI
        # (scripts/test_soft_auc.py).  Different scale -- read the CI, not the level.
        print("   [soft-truth metric, sharper]")
        print(sh([PY, os.path.join("scripts", "cv_score.py"), "--dir", "downloads/cv",
                  "--pair", "%s:final" % base, "%s:%s" % (base, kind), "--boot", str(a.boot),
                  "--soft"]))
        print()

    if not a.no_join:
        print("=" * 72)
        print("D. ARM-JOIN -- does OUR arm add anything to the author's 4-arm OOF?")
        print("   (this is the re-open condition for the parked integration idea, HANDOFF 85.4)")
        print("=" * 72)
        print(sh([PY, os.path.join("scripts", "oof_arm_join.py"), "--ours", os.path.join("downloads", "cv", base),
                  "--boot", "600"]))

    print("\n" + "=" * 72)
    print("E. DECISION BLOCK -- what to do with the numbers above")
    print("=" * 72)
    print("""  gate 1 (B): the control is 1 epoch vs 12; expectation is a gap of order 0.1.
              CLEAR gap  -> the CV resolves large effects; keep going.
              NO gap     -> DISCARD the instrument, do not iterate on it.
  SWA/epens (C): adopt only if the CI excludes 0 (a 58-study 'BETTER' is NOT a verdict, §85.2).
              Each check is printed TWICE: binarised truth (the competition analogue) and
              SOFT truth (a measured-sharper instrument, 36% tighter paired CI).  The two are
              on DIFFERENT SCALES -- compare the verdicts and the CI widths, never the levels.
              If the two disagree, say so rather than picking the friendlier one.
  arm-join (D): integration is worth re-opening only if OURS beats the author's arms AND the
              blend delta has a CI excluding 0 on the shared studies.
  Reminder   : the OOF truth is the rule parser's soft labels, so NONE of this can judge a
              label-function change, and every number here is a PROXY for the LB (HANDOFF 82.7).""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
