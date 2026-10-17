# -*- coding: utf-8 -*-
"""
List every fold-0 arm that a member census COULD see, and flag the ones the P1.1 census
(FACTS A158) never tested.

Why: A158 concluded "our existing checkpoints are exhausted as members" from a 24-arm
census -- but its scan only covered downloads/cv, downloads/rent, downloads/tpu and
downloads/a5lite_*. Our single best arm (msft2, tag msft0) writes to work/msft2_out/ and
was therefore INVISIBLE. Measuring it for the first time gives +0.0084
CI [+0.0060,+0.0107] on the strong base -- the only member ever to clear the >=0.005 bar.

So the census's conclusion needs re-testing over the FULL set, not the subset it could see.
This prints that full set.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PAT = re.compile(r"raptor_oof_([A-Za-z0-9]+?)(?:_(ep\d+|epens|swa))?_fold(\d+)\.npz$")

# what the A158 census could actually see
OLD_GLOBS = ["downloads/cv/**/raptor_oof_*.npz",
             "downloads/rent/**/raptor_oof_*.npz",
             "downloads/tpu/**/raptor_oof_*.npz",
             "downloads/a5lite_*/raptor_oof_*.npz"]

seen_old, seen_all = {}, {}
for pat in ("**/raptor_oof_*.npz",):
    for p in ROOT.glob(pat):
        m = PAT.search(p.name)
        if not m or m.group(2):
            continue
        key = "%s@f%s" % (m.group(1), m.group(3))
        seen_all.setdefault(key, p)
        rel = p.relative_to(ROOT).as_posix()
        if any(p.match(g) for g in OLD_GLOBS):
            seen_old.setdefault(key, p)

print(f"arms visible to the OLD census : {len(seen_old)}")
print(f"arms in the FULL set           : {len(seen_all)}")
missing = sorted(set(seen_all) - set(seen_old))
print(f"arms the census COULD NOT SEE  : {len(missing)}")
print()
print("=== fold-0 arms invisible to the A158 census ===")
f0 = [k for k in missing if k.endswith("@f0")]
for k in sorted(f0):
    print(f"  {k:<16} {seen_all[k].relative_to(ROOT).as_posix()}")
print()
print(f"({len(missing) - len(f0)} more on other folds: "
      f"{[k for k in missing if not k.endswith('@f0')]})")
print()
print("=== full fold-0 list (what a re-run census should cover) ===")
allf0 = sorted(k for k in seen_all if k.endswith("@f0"))
print("  " + ", ".join(allf0))
