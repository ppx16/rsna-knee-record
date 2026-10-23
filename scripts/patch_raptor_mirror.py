# -*- coding: utf-8 -*-
"""
Wire the anatomical mirror TTA into the Raptor eval path, as an ISOLATED, opt-in switch.

WHAT: the public 0.949 notebook averages a plain view with an "anatomical mirror" of it. Read
from its code, the mirror AXIS is chosen per window from the plane the window's CENTRE slice
sits in -- sagittal: reverse the 3 slice channels; coronal/axial: flip the width. A plain flip
would destroy medial/lateral, which is exactly why our own trainer says `NO flips`.

HOW: two anchors only, both confirmed from this file's source, not guessed:
  1. before `class StudyWindows(Dataset):`  -> the slot->plane table + the enable switch
  2. after `tri = np.stack([vol[c-1], vol[c], vol[c+1]], ...)` -> the mirror itself

The switch is an ENVIRONMENT VARIABLE (RAPTOR_MIRROR), so no constructor or argparse edit is
needed. Run eval twice -- without and with it -- and average the two gold npz files locally;
no averaging logic enters the kernel, so the mirror's effect is fully isolated.

WHY MIRRORING AFTER THE /255 IS CORRECT: the public implementation mirrors on uint8 before its
LUT ("mirror the knee, then normalise"). Reversing channels and flipping width are LINEAR INDEX
operations, so they commute with a scalar division -- mirroring after that line is
arithmetically identical. That is a property, not a preference.

Every anchor must match exactly once; the result is compiled BEFORE it is written.
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
TARGETS = [ROOT / "kernels" / "kernel-raptor-train" / "knee-mri-training-the-twelve-finding-model.py",
           ROOT / "kernels" / "kernel-raptor-train-rule" / "knee-mri-training-the-twelve-finding-model.py"]

A1_OLD = "class StudyWindows(Dataset):"
A1_NEW = '''# ---- anatomical mirror TTA: corpus slot -> plane ------------------------------------------
# Read VERBATIM from kernels/kernel-diag-slots/diag_slots.py, not inferred. NOTE sagittal
# appears TWICE (fluid=1 then fluid=0), so its block is 12+10 = 22 slots, not 12 -- the
# boundary that work/mirror_tta.py's self-test caught me getting wrong.
import os as _os
_SLOT_GROUPS_44 = [("Sagittal", 1, 12), ("Sagittal", 0, 10), ("Coronal", 1, 8),
                   ("Coronal", 0, 6), ("Axial", -1, 8)]
_SLOT_PLANES = [p.lower() for p, _f, k in _SLOT_GROUPS_44 for _ in range(k)]
assert len(_SLOT_PLANES) == 44, len(_SLOT_PLANES)
# opt-in: set RAPTOR_MIRROR=1 and run eval twice to isolate the effect
MIRROR_ENABLED = bool(_os.environ.get("RAPTOR_MIRROR", ""))


class StudyWindows(Dataset):'''

A2_OLD = '''            tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0  # (3,H,W)
            H, W = tri.shape[-2], tri.shape[-1]'''
A2_NEW = '''            tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0  # (3,H,W)
            # ---- anatomical mirror TTA (opt-in, default OFF) -----------------------------
            # The axis is chosen PER WINDOW from the plane of THIS window's centre slice, and
            # `c` indexes the corpus slot axis directly, so the plane is a plain lookup. A
            # batch-level width flip would be WRONG for sagittal windows: their L/R axis IS
            # the 3-slice axis. Mirrored after /255 deliberately -- flips are linear index ops
            # and commute with a scalar division, so this equals "mirror the knee, then
            # normalise" exactly.
            if MIRROR_ENABLED:
                if _SLOT_PLANES[c] == "sagittal":
                    tri = tri[::-1]            # L/R is the slice axis
                else:
                    tri = tri[:, :, ::-1]      # L/R is the image width
            H, W = tri.shape[-2], tri.shape[-1]'''


def patch(p: pathlib.Path) -> str:
    src = p.read_text(encoding="utf-8")
    if "MIRROR_ENABLED" in src:
        return f"{p.name}: already patched"
    for old, new, tag in ((A1_OLD, A1_NEW, "slot-plane table"),
                          (A2_OLD, A2_NEW, "mirror block")):
        n = src.count(old)
        if n != 1:
            raise SystemExit(f"FATAL {p.parent.name}/{tag}: anchor found {n} times, expected 1")
        src = src.replace(old, new, 1)
    compile(src, str(p), "exec")          # compile BEFORE writing -- changed after breaking a file once
    p.write_text(src, encoding="utf-8")
    return f"{p.name}: patched + compile OK"


def main():
    for p in TARGETS:
        if not p.exists():
            print(f"MISSING {p}")
            continue
        print(" ", patch(p))
        t = p.read_text(encoding="utf-8")
        for k in ("MIRROR_ENABLED", "_SLOT_PLANES", "tri[::-1]", "tri[:, :, ::-1]",
                  "RAPTOR_MIRROR"):
            print(f"      {k:<22} {'OK' if k in t else 'MISSING'}")


if __name__ == "__main__":
    main()
