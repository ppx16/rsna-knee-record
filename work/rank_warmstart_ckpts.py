# -*- coding: utf-8 -*-
"""
Rank every released Raptor checkpoint we hold locally as a WARM-START candidate.

Why: msft2 (a CoAtNet warm-started from maxspan v5) is the only member ever measured to
clear +0.005 as a member (+0.0084, CI [+0.0060,+0.0107], FACTS C91). A158's pattern says
only the CoAtNet family passes. So the one evidence-backed route left is another CoAtNet
warm start from a DIFFERENT released checkpoint -- and msft2's own guard tells us the bar
to clear: it refuses a warm start scoring below 0.8963 on our corpus (FACTS C92).

This prints, for each local checkpoint, the `gold_auc` the author reports and the geometry
it belongs to (scripts/dump_arms.py cross-checks those against the dataset descriptions).
The geometry column is the one that matters: our corpus is 44 slots.

  maxspan-v5        (336, 64 slots, span .02-.98) -> 62 windows   [msft2's start, scores 0.8963 here]
  native384dense-v10(384, 64 slots, span .02-.98) -> 62
  native384-v8      (384, 44 slots, span .06-.94) -> 42           <- MATCHES our 44-slot corpus
  finespacing-v9    (336, 80 slots, span .02-.98) -> 78           [reads 0.8907 here, guard refused]
"""
import pathlib
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = pathlib.Path(__file__).resolve().parent.parent

# geometry, from scripts/dump_arms.py (cross-checked against the author's dataset descriptions)
GEOM = {
    "v5": "336 / 64 slots / span .02-.98  (62 win)",
    "v8": "384 / 44 slots / span .06-.94  (42 win)   <-- matches our corpus",
    "v9": "336 / 80 slots / span .02-.98  (78 win)",
    "v10": "384 / 64 slots / span .02-.98  (62 win)",
    "384x": "see raptor-knee-arms-x description",
}

NAMES = [("v5", "raptor_ft_coatnet_v5_full_swa.pt"),
         ("v8", "raptor_ft_coatnet_v8_full_swa.pt"),
         ("v9", "raptor_ft_coatnet_v9_full.pt"),
         ("v10", "raptor_ft_coatnet_v10_full_swa.pt"),
         ("384x", "raptor_ft_coatnet384x.pt"),
         ("384", "raptor_ft_coatnet384.pt")]

found = {}
for p in ROOT.rglob("raptor_ft_coatnet*.pt"):
    found.setdefault(p.name, p)

import torch  # noqa: E402

print(f"{'tag':<6} {'gold_auc':>10} {'res':>4}  {'arch':<46} file")
print("-" * 120)
rows = []
for tag, fname in NAMES:
    p = found.get(fname)
    if p is None:
        for k, v in found.items():
            if tag in k and "swa" in k:
                p = v
                break
    if p is None:
        continue
    try:
        ck = torch.load(str(p), map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"{tag:<6} {'LOAD FAIL':>10}  {e}")
        continue
    st = ck
    for c in ("model", "student", "state_dict"):
        if isinstance(st, dict) and isinstance(st.get(c), dict):
            st = st[c]
            break
    n_bb = sum(1 for k in st if isinstance(k, str) and k.startswith("backbone."))
    rows.append((tag, ck.get("gold_auc"), ck.get("res"), ck.get("arch"), n_bb,
                 p.relative_to(ROOT).as_posix()))

rows.sort(key=lambda r: -(r[1] or 0))
for tag, ga, res, arch, n_bb, rel in rows:
    print(f"{tag:<6} {(ga if ga is not None else float('nan')):10.4f} {str(res):>4}  "
          f"{str(arch)[:46]:<46} {n_bb} bb  {rel}")
print()
print("geometry of each (from scripts/dump_arms.py):")
for tag, g in GEOM.items():
    print(f"  {tag:<6} {g}")
print()
print("msft2's guard bar on OUR corpus = 0.8963 (v5). v9 read 0.8907 and was REFUSED.")
print("=> a candidate must be evaluated on our corpus to know if it clears the bar.")
