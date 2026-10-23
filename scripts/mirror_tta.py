# -*- coding: utf-8 -*-
"""
Anatomical mirror TTA -- and the one thing that makes it "anatomical" rather than a flip.

WHY THIS FILE EXISTS
    A public notebook scoring 0.949 (nartaa/rsna-knee-0949-anatomical-mirror) averages two
    views: the plain one and an "anatomical mirror" of it. Read from its code, the mirror is
    NOT a plain flip and NOT a label swap -- it is a geometric mirror whose AXIS is chosen
    from the plane the window sits in:

        sagittal  (slots 0,1)  -> reverse the 3 slice channels   (L/R is the slice axis)
        coronal/axial (2,3,4)  -> flip the image width           (L/R is the image axis)

    done on uint8 BEFORE normalisation ("mirror the knee, then normalise"), the two views
    averaged with equal weight.

    This matters because our own Raptor trainer says, verbatim,
        `NO flips (laterality is signal)`
    -- a PLAIN flip destroys the medial/lateral distinction, which is why we never did this.
    An ANATOMICAL mirror preserves it. The public gains land exactly on the findings that cap
    us (Lateral Meniscus +0.071, Lateral OA +0.048, Medial Meniscus +0.035).

OUR SLOT TABLE (from kernels/kernel-corpus64/build_corpus.py, the 64-slot recipe)
    [("Sagittal",1,18), ("Sagittal",0,14), ("Coronal",1,12), ("Coronal",0,8), ("Axial",-1,12)]
    i.e. slot 0..17 sagittal, 18..31 coronal, 32..43 axial -- plus the 44-slot span variant.

SELF-TEST
    test_axis_choice() asserts the axis each slot picks, and that mirroring TWICE is the
    identity -- the invariance that catches a wrong axis far more reliably than eyeballing.
"""
from __future__ import annotations

import numpy as np

# the 64-slot recipe; a window's CENTRE slice decides the axis
SLOT_GROUPS_64 = [("Sagittal", 1, 18), ("Sagittal", 0, 14), ("Coronal", 1, 12),
                  ("Coronal", 0, 8), ("Axial", -1, 12)]
# THE 44-SLOT RECIPE -- this is the one OUR TRAINER reads (knee-raptor-corpus is 44 slots).
# Read verbatim from kernels/kernel-diag-slots/diag_slots.py, not inferred:
#   [("Sagittal",1,12), ("Sagittal",0,10), ("Coronal",1,8), ("Coronal",0,6), ("Axial",-1,8)]
# => sagittal 0..21, coronal 22..35, axial 36..43
SLOT_GROUPS_44 = [("Sagittal", 1, 12), ("Sagittal", 0, 10), ("Coronal", 1, 8),
                  ("Coronal", 0, 6), ("Axial", -1, 8)]


def slot_planes(groups=SLOT_GROUPS_64):
    """Plane label per slot index, in order."""
    out = []
    for plane, _fluid, k in groups:
        out.extend([plane.lower()] * k)
    return out


def mirror_axis(plane: str) -> str:
    """Which axis carries left/right for this plane.

    sagittal  -> 'slices' (the 3 stacked slices ARE the L/R axis)
    coronal / axial -> 'width' (L/R is the horizontal image axis)

    NOTE: this is the anatomical claim. It is asserted in the self-test against the exact
    behaviour the public implementation documents; it is NOT something I can verify from
    first principles without a run, and it is flagged as such in the SPEC.
    """
    p = plane.lower()
    if p.startswith("sag"):
        return "slices"
    return "width"


def anatomical_mirror(tri: np.ndarray, centre_slot: int, groups=SLOT_GROUPS_64) -> np.ndarray:
    """Mirror a (3, H, W) uint8 window about its anatomical L/R axis.

    tri          : (3, H, W) -- the 3 stacked neighbouring slices of one window
    centre_slot  : the corpus slot index of the window's CENTRE slice
    """
    planes = slot_planes(groups)
    if not (0 <= centre_slot < len(planes)):
        raise ValueError(f"centre_slot {centre_slot} outside 0..{len(planes) - 1}")
    ax = mirror_axis(planes[centre_slot])
    if ax == "slices":
        return tri[::-1, :, :].copy()        # reverse the 3 slice channels
    return tri[:, :, ::-1].copy()            # flip the image width


def mirror_batch(tri: np.ndarray, centre_slots, groups=SLOT_GROUPS_64) -> np.ndarray:
    """Mirror a (K, 3, H, W) batch, axis chosen PER WINDOW.

    Per-window is the whole point: a batch-level flip would apply the wrong axis to every
    window whose plane differs from the majority.
    """
    out = np.empty_like(tri)
    for i, c in enumerate(centre_slots):
        out[i] = anatomical_mirror(tri[i], int(c), groups)
    return out


def test_axis_choice() -> None:
    planes = slot_planes()
    assert len(planes) == 64, len(planes)
    # NOTE the grouping: sagittal appears TWICE (fluid=1 and fluid=0), so the sagittal block is
    # 18+14 = 32 slots, not 18. The self-test caught my first, wrong expectation of 18/32/64 --
    # which is exactly why the boundaries are asserted rather than assumed.
    assert planes[0] == "sagittal" and planes[31] == "sagittal", planes[:32]
    assert planes[32] == "coronal" and planes[51] == "coronal", planes[32:52]
    assert planes[52] == "axial" and planes[63] == "axial", planes[52:]

    rng = np.random.default_rng(0)
    for slot in (0, 31, 32, 51, 52, 63):
        tri = rng.integers(0, 256, size=(3, 8, 8), dtype=np.uint8)
        once = anatomical_mirror(tri, slot)
        twice = anatomical_mirror(once, slot)
        assert np.array_equal(tri, twice), f"mirror twice is not identity at slot {slot}"
        assert not np.array_equal(tri, once), f"mirror did nothing at slot {slot}"
        ax = mirror_axis(planes[slot])
        if ax == "slices":
            assert np.array_equal(once, tri[::-1]), f"slot {slot} should reverse slices"
        else:
            assert np.array_equal(once, tri[:, :, ::-1]), f"slot {slot} should flip width"

    # the trap: a batch-level width flip is WRONG for sagittal windows
    tri = rng.integers(0, 256, size=(1, 3, 8, 8), dtype=np.uint8)
    batch = np.repeat(tri, 4, axis=0)
    ok = mirror_batch(batch, [0, 18, 32, 63])
    naive = batch[:, :, :, ::-1]
    assert not np.array_equal(ok[0], naive[0]), "sagittal window must NOT use the width flip"
    assert np.array_equal(ok[3], naive[3]), "axial window should equal the width flip"
    print("test_axis_choice: OK  (axis per slot, involution, and the batch-flip trap)")
    for groups, label in ((SLOT_GROUPS_64, "64-slot"), (SLOT_GROUPS_44, "44-slot (OURS)")):
        pl = slot_planes(groups)
        n_sag = pl.count("sagittal")
        print(f"  {label}: {len(pl)} slots | sagittal 0..{n_sag - 1} -> slices | "
              f"coronal {n_sag}..{n_sag + 13} -> width | axial ... -> width")


if __name__ == "__main__":
    test_axis_choice()
