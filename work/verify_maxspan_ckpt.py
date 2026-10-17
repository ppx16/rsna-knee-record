# -*- coding: utf-8 -*-
"""
Verify the MaxSpan v5 checkpoint against the patch, WITHOUT spending any quota.

This is the premise test for 方向一: if the released checkpoint's `model` dict is not
our RaptorClassifier's state_dict, the whole "start from >=0.90" plan is empty.

It answers, from the artefact rather than from a note:
  1. what containers the file actually has (this is the trap: load_raptor used to
     unwrap only "student", so a "model" file yielded {} and trained from timm);
  2. whether the OLD loader would have extracted ZERO tensors (the silent failure);
  3. whether the NEW loader extracts them (the fix);
  4. whether the head tensor names match our RaptorClassifier verbatim;
  5. the arch / res / lab / gold_auc the file reports.
"""
import json
import pathlib
import sys

CKPTS = [
    r"D:\kaggle\rsna-knee\work\above\tony\x\raptor-knee-maxspan\raptor_ft_coatnet_v5_full_swa.pt",
    r"D:\kaggle\rsna-knee\downloads\raptor_v9\raptor_ft_coatnet_v9_full.pt",
]

try:
    import torch
except Exception as e:                       # pragma: no cover
    print(f"NO TORCH locally ({e}); cannot inspect the checkpoint here")
    sys.exit(2)


def old_container(st):
    """Exactly what the shipped load_raptor did before the patch."""
    return st["student"] if "student" in st else st


def new_container(ck):
    """Exactly what the patched load_raptor does."""
    st = ck
    for c in ("model", "student", "state_dict", "net", "backbone", "ema"):
        if isinstance(st, dict) and isinstance(st.get(c), dict):
            st = st[c]
            break
    return st


for path in CKPTS:
    p = pathlib.Path(path)
    print("=" * 100)
    if not p.exists():
        print(f"MISSING {path}")
        continue
    print(f"{p.name}   {p.stat().st_size:,} B")
    ck = torch.load(str(p), map_location="cpu", weights_only=False)
    print(f"  top-level keys      : {list(ck)[:10]}")
    for k in ("arch", "res", "gold_auc", "lab", "epoch", "src", "swa_over"):
        if isinstance(ck, dict) and k in ck:
            v = ck[k]
            v = (v if not isinstance(v, (list, tuple)) or len(v) <= 14 else f"{list(v)[:14]}...")
            print(f"  {k:<20}: {v}")

    old = old_container(ck)
    old_bb = {k[len("backbone."):]: v for k, v in old.items()
              if isinstance(k, str) and k.startswith("backbone.")}
    new = new_container(ck)
    new_bb = {k[len("backbone."):]: v for k, v in new.items()
              if isinstance(k, str) and k.startswith("backbone.")}
    new_head = [k for k in new if isinstance(k, str) and not k.startswith("backbone.")]

    print(f"  OLD loader (student-only) -> {len(old_bb)} backbone tensors"
          f"   {'*** SILENT ZERO ***' if not old_bb else ''}")
    print(f"  NEW loader                -> {len(new_bb)} backbone tensors, "
          f"{len(new_head)} head tensors")
    if not new_bb:
        print("  *** NEW LOADER ALSO FAILS -- do not spend quota on this line ***")
        continue
    print(f"  head keys           : {sorted(new_head)}")
    tot = sum(v.numel() for v in new_bb.values() if hasattr(v, "numel"))
    print(f"  backbone params     : {tot/1e6:.2f} M")
    first = next(iter(new_bb))
    print(f"  first backbone key  : {first}  shape={tuple(new_bb[first].shape)}")
