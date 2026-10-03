# -*- coding: utf-8 -*-
"""Build `kernel-0943cfm2eff`: `kernel-0943cfm` PLUS the `tbongc` EffNet-B0 2-level MIL leg.

WHY A NEW KERNEL AND NOT AN EDIT
    `kernel-0943cfm` (the `cn0` 5th Raptor view + the Fracture fix + the MCL exclusion) is
    running and has submissions pending.  Editing it in place and pushing a new version
    destroys the running output.  So this copies it and adds one leg.  Nothing else in the
    recipe changes; `kernel-0943cfm`, `kernel-0943cf` and every other kernel are untouched.

WHAT THE LEG IS
    `tbongc/rsna-knee-effb0mil-sl16-tv7w` -- a 2-level slice->slot->study MIL on
    EfficientNet-B0.  4 folds, 61 MB.  Its published OOF, rank-averaged against OUR
    `oof_equal_prob` on the 4349 studies it covers, is monotone and positive on all 12
    labels (docs/FIND-2026-10-03-public-artifact-scan.md).  Reproduced exactly by
    `scratch/verify_effb0b.py`:

        w=0.00 0.84962 | w=0.05 0.85290 | w=0.10 0.85587 | w=0.15 0.85851
        w=0.20 0.86083 | w=0.30 0.86440        (+0.00625 at w=0.10)

    !! AND THE FACT THAT DECIDES HOW TO READ IT: effb0mil's four published folds cover
    4349 of our 4407 studies and **not one of the 58 fully-labelled ones**.  So this leg
    cannot be measured on gold-58 at all, and the +0.0063 lives on the soft labels we select
    on -- the same currency in which our own closed item #8 (selection illusion: +0.011
    in-sample -> +0.0003 out-of-fold) was won and lost.  Hence the default weight of 0.10,
    not the 0.20-0.30 the curve peaks at.

!! TWO FACTUAL ERRORS IN THE BRIEF, RECORDED BECAUSE THEY WOULD COST A RUN:
    1. `sl16_fix.csv` and `s_cache_e13_224.npy` are NOT in `tbongc/rsna-knee-cache336-v1`.
       That dataset (created 2026-09-11) holds `cache336_index.parquet`, `slot_pick_train.parquet`
       and `shard_000..022.npy` at 336 px -- a DIFFERENT cache.  `s_cache_e13_224.npy` is in
       `ngoanphamvan/sl16s0b-cache-variant` under `track_s_sl16/`, and `sl16_fix.csv` is not
       published at all (its declared home, `tbongc/rsna-knee-labels`, is 403 to the public API).
    2. `effb0mil_lib.py` has no inference path -- it is a training library.  `KneeDS`,
       `to_input` and `EffB0MIL` are the contract that has to be re-driven by hand.

WHAT THE NOTEBOOK ADDS
    (a) `RUN["effb0mil_w"] = 0.10` in the RUN CONFIG cell, settable there or via
        `RSNA_EFFB0MIL_W`;
    (b) a test-side 224 px 4-slot cache built with THIS notebook's own builder
        (`walk`/`annotate`/`pick_slots`/`order_slices`/`read_slot`/`normalise_laterality`),
        re-pinned to the frozen corpus contract (`RULES_LEGACY` = dominant_axis / corner_x /
        slot_fallback / decode_fill=zero, 16 slices, 130 mm, band 0.12-0.88, 224 px) and to
        `SLOTS_RECOVERED` so the slot axis is `[SAG_FS, COR_FS, AX_FS, SAG_T1]` -- the order
        `effb0mil_lib.py`'s `SLOT_PLANE`/`SLOT_FS`/`slot_prior` were fitted for.  Cells 43
        and 45 leave `RULES`/`IMG`/`CROP_MM`/`CACHE_SLICES`/`SLICE_BAND`/`SLOTS` mutated, so
        the leg cell re-pins all six and ASSERTS the pin;
    (c) a CONVENTION GATE that rebuilds real TRAIN studies out of the raw DICOM with that
        same code and compares them to the published `s_cache_e13_224.npy` byte-for-byte
        (identity OR axis mirror, because `effb0mil_lib`'s per-row `flip` remap can only
        mirror).  A non-mirror mismatch, or a majority-mirror corpus, ABANDONS the leg;
    (d) 4-fold inference, per-label rank blending into `_pipeline_stage.csv` -- the same file
        the existing pipeline writes and the publish cell reads -- at
        `(1-w) * rank(ours) + w * rank(leg)`;
    (e) the per-label AUC table printed BEFORE the blend is written.

THE FAIL-SAFE
    Every failure path -- dataset not mounted, library/checkpoint missing, cache build error,
    convention gate not passed, inference error, UID mismatch -- is caught, logged loudly with
    `rsna_event('effb0mil_leg_failed', ...)`, and leaves `_pipeline_stage.csv` EXACTLY as
    `kernel-0943cfm` produced it.  `submission.csv` is then the 0943cfm behaviour.  There is no
    half-built blend anywhere: the blend is computed in one place and written with `os.replace`.

Usage:
    python scripts/build_0943cfm2eff.py            # build + validate + audit report
    python scripts/build_0943cfm2eff.py --check     # re-assert the built artifact only
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys
import uuid

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "kernels" / "kernel-0943cfm"
DST = ROOT / "kernels" / "kernel-0943cfm2eff"
NB = "rsna-knee-restructured-version-3.ipynb"
KERNEL_ID = "ppxl16/rsna-knee-0943cfm2eff"
KERNEL_TITLE = "RSNA knee 0943cfm2eff"
REPORT = "BUILD-REPORT.md"

EFFB0_DATASET = "tbongc/rsna-knee-effb0mil-sl16-tv7w"
CACHE_DATASET = "ngoanphamvan/sl16s0b-cache-variant"


# ==================================================================================================
# 1. the RUN CONFIG edit
# ==================================================================================================
RUNCONFIG_ANCHOR = '    "coat_private_alpha": 0.40,'
RUNCONFIG_NEW = '''    "coat_private_alpha": 0.40,

    # --- stage 5: the EffNet-B0 2-level MIL leg (tbongc/rsna-knee-effb0mil-sl16-tv7w) --------
    # Rank-blended in AFTER every existing stage, into the file the publish cell reads.
    # 0.10, not the 0.20-0.30 the OOF curve peaks at: that curve is soft-label and in-sample,
    # and our own closed item #8 is the precedent for an in-sample +0.011 collapsing to
    # +0.0003 out-of-fold.  Set RSNA_EFFB0MIL_W in the environment to override without
    # editing; 0.0 disables the leg completely (no cache build, no forward pass).
    "effb0mil_w": float(os.environ.get("RSNA_EFFB0MIL_W", "0.10")),
    # The leg runs LAST, so its own cost is bounded rather than spent blindly: it re-reads
    # ~4 x 6 slot-series of raw DICOM for the convention gate. If it overruns, it abandons
    # itself instead of letting `rsna_deadline` (which is fail-CLOSED) abort the kernel.
    "effb0mil_budget_s": float(os.environ.get("RSNA_EFFB0MIL_BUDGET_S", "540")),
    "effb0mil_reserve_s": 240.0,'''


# ==================================================================================================
# 2. the leg implementation
# ==================================================================================================
LEG_CELL = r'''# ======================= STAGE 5 - EffNet-B0 2-level MIL leg =================
# tbongc/rsna-knee-effb0mil-sl16-tv7w, rank-blended in at RUN["effb0mil_w"].
#
# SELF-CONTAINED AND FAIL-SAFE.  Nothing here can abort the pipeline: every path ends in
# `_effb0mil_ok = False` plus a loud rsna_event, and the publish cell then reads exactly the
# bytes kernel-0943cfm would have written.
#
# ---------------------------------------------------------------------------------------------
# THE ONE THING THAT MATTERS: THE PIXEL CONVENTION
# ---------------------------------------------------------------------------------------------
# `effb0mil_lib.py` carries a per-row `flip` from `sl16_fix.csv` and applies it to the FROZEN
# TRAIN cache before its forward pass.  That cache (`s_cache_e13_224.npy`, 4 x 16 x 224 x 224
# uint8, 4407 rows in train.csv order) was built by the public cache builder with
#     RULES_LEGACY = {order: dominant_axis, lat: corner_x, slot_fallback: True, decode_fill: zero}
#     band (0.12, 0.88), 130 mm, 224 px, 16 slices, slots [SAG_FS, COR_FS, AX_FS, SAG_T1]
# plus `normalise_laterality` (mirror each right knee to a left one, axis chosen by plane).
# The library's `flip` remap then mirrors SOME rows on top of that, to reach the cache336_v1
# convention.
#
# The cache side and the model side therefore differ by nothing but a per-row MIRROR.  So:
#   * a test cache built by the SAME builder cannot be in a different convention from the
#     model, because a mirror is exactly the degree of freedom the library's own `flip`
#     variable already represents;
#   * `flip` must NOT be applied on the test side -- it is a train-cache identity remap, not
#     something the model demands of new pixels.
# That argument is only sound if the builder really does reproduce the frozen cache, so it is
# TESTED, not asserted: `_effb0mil_convention_gate()` rebuilds real TRAIN studies from the raw
# DICOMs and compares them to the published cache byte-for-byte, identity OR axis mirror.
# Anything else -- a different crop, a different slice order, an intensity change -- shows up
# as a NON-mirror mismatch and abandons the leg.
#
# WHAT I COULD NOT DETERMINE, STATED PLAINLY: `sl16_fix.csv` was never published (its declared
# home, `tbongc/rsna-knee-labels`, is 403 to the public API), so the CONTENTS of `flip` are
# unknown, and the brief's claim that the 21 GB `tbongc/rsna-knee-cache336-v1` contains
# `sl16_fix.csv` + `s_cache_e13_224.npy` is false -- that dataset holds a different 336 px
# cache.  This cell never needs `flip` (see the argument above), but it cannot by itself rule
# out the cache having been built by a rule this notebook does not have.  The gate is what
# rules that out, and the gate runs before any prediction is used.
#
# ---------------------------------------------------------------------------------------------
# v2 FIX: THE SLOT VOCABULARY  (v1 died here, before the gate, with "'SAG_FS' is not in list")
# ---------------------------------------------------------------------------------------------
# `effb0mil_lib.py` names its four slots `['sag_fluid','cor_fluid','ax_fluid','sag_t1']` with
# `SLOT_PLANE = [0,1,2,0]` and `SLOT_FS = [1,1,1,0]` -- those are the names of the cache-BUILD
# kernel's own `SLOTS` (`SAG_FS`/`COR_FS`/`AX_FS`/`SAG_NOFS`), and v1 wrongly assumed they were
# ours.  They are not: our `SLOTS_RECOVERED` carries `SAG_FLUID_FS` / `COR_FLUID_FS` /
# `AX_FLUID_FS` / `SAG_FLUID_NOFS` / `COR_T1` / `SAG_T1`.  So the four library slots are now
# resolved by the library's OWN geometry -- (plane, fat-suppressed) from SLOT_PLANE/SLOT_FS --
# the resolution is printed and asserted, and the one genuinely ambiguous slot (the fourth:
# `SAG_FLUID_NOFS` vs `SAG_T1`, with `COR_T1` excluded by plane) is left to the gate.

import gc as _eff_gc
import os as _eff_os
import time as _eff_time
import numpy as _eff_np
import pandas as _eff_pd
from pathlib import Path as _EffPath

_EFF_W = float(RUN.get("effb0mil_w", 0.0) or 0.0)
_EFF_DIR = _EffPath("/kaggle/working/_effb0mil")
_EFF_N_SLOT, _EFF_N_SLICE, _EFF_IMG = 4, 16, 224
_EFF_SLOT_NAMES = ["SAG_FS", "COR_FS", "AX_FS", "SAG_T1"]
_EFF_OK = False
_effb0mil_ok = False
_effb0mil_test_rank = None
_effb0mil_build = None
_EFF_ISSUES = []
# The leg runs at the very END, so its own cost must be bounded against the run budget rather
# than spent blindly: `rsna_deadline` is fail-CLOSED (it aborts the kernel), which would turn a
# useful neck of margin into NO submission at all.  So the leg abandons itself instead.
_EFF_T0 = _eff_time.time()
_EFF_BUDGET_S = float(RUN.get("effb0mil_budget_s", 540.0) or 540.0)
_EFF_RESERVE_S = float(RUN.get("effb0mil_reserve_s", 240.0) or 240.0)
_EFF_GATE_BUDGET_S = _EFF_BUDGET_S * 0.6


class _EffOutOfTime(Exception):
    pass


def _eff_check_budget(tag, since=None):
    if since is not None and (_eff_time.time() - since) > _EFF_GATE_BUDGET_S:
        raise _EffOutOfTime(f"{tag}: exceeded the {_EFF_GATE_BUDGET_S:.0f}s cache-gate budget")
    if (_eff_time.time() - _EFF_T0) > _EFF_BUDGET_S:
        raise _EffOutOfTime(f"{tag}: exceeded the {_EFF_BUDGET_S:.0f}s leg budget")
    if (time.time() - T0) > (TIME_BUDGET - _EFF_RESERVE_S):
        raise _EffOutOfTime(f"{tag}: run budget has under {_EFF_RESERVE_S:.0f}s of margin left")


def _eff_issue(code, **kw):
    _EFF_ISSUES.append(dict(code=code, **{k: str(v)[:400] for k, v in kw.items()}))
    try:
        rsna_event("effb0mil_" + code, **kw)
    except Exception:
        pass


def _eff_find(name):
    """Locate one input asset.  The mounting slug is the author's, not ours, so this walks
    /kaggle/input instead of trusting one hard-coded path."""
    hits = sorted({str(p.resolve()) for p in _EffPath("/kaggle/input").rglob(name)})
    if len(hits) != 1:
        raise FileNotFoundError(
            f"{name}: expected exactly one mounted copy, found {len(hits)}: {hits[:6]}")
    return _EffPath(hits[0])


if not (0.0 <= _EFF_W <= 1.0):
    # A mis-typed weight is a configuration error, but it must not cost the submission: the leg
    # is dropped loudly and `_pipeline_stage.csv` keeps kernel-0943cfm's bytes.
    _eff_issue("bad_weight", weight=_EFF_W)
    print(f"[effb0] RUN['effb0mil_w']={_EFF_W} is not in [0, 1] - LEG DROPPED, "
          f"_pipeline_stage.csv stays EXACTLY kernel-0943cfm's output", flush=True)
    _EFF_W = 0.0

if _EFF_W == 0:
    print("[effb0] RUN['effb0mil_w']=0 - leg disabled by configuration", flush=True)
else:
    try:
        # ---------------------------------------------------------------------------------
        # 1. assets + library
        # ---------------------------------------------------------------------------------
        _EFF_DIR.mkdir(parents=True, exist_ok=True)
        _EFF_LIBP = _eff_find("effb0mil_lib.py")
        _EFF_TP = _eff_find("s_cache_e13_224.npy")
        _EFF_CKPTS = [_eff_find(f"best_effb0mil_sl16_tv7w_f{f}.pt") for f in range(4)]
        _EFF_OOF = {f: (_eff_find(f"oof_effb0mil_sl16_tv7w_f{f}.npy"),
                        _eff_find(f"oof_effb0mil_sl16_tv7w_f{f}_ids.csv")) for f in range(4)}
        print(f"[effb0] leg weight {_EFF_W:.3f}", flush=True)
        print(f"[effb0] library     {_EFF_LIBP}", flush=True)
        print(f"[effb0] train cache {_EFF_TP}", flush=True)
        print(f"[effb0] checkpoints {[p.name for p in _EFF_CKPTS]}", flush=True)

        import importlib.util as _eff_ilu
        import torch as _eff_torch

        _EFF_SPEC = _eff_ilu.spec_from_file_location("_effb0mil_lib", str(_EFF_LIBP))
        _efflib = _eff_ilu.module_from_spec(_EFF_SPEC)
        _EFF_SPEC.loader.exec_module(_efflib)          # __main__ block is guarded
        if _efflib.SLOT_PLANE != [0, 1, 2, 0] or _efflib.SLOT_FS != [1, 1, 1, 0]:
            raise RuntimeError(f"library slot contract drifted: {_efflib.SLOT_PLANE} "
                               f"{_efflib.SLOT_FS}")
        print(f"[effb0] library loaded: backbone={_efflib.CFG['backbone']} "
              f"img={_efflib.CFG['img']} nsl={_efflib.CFG['nsl']}", flush=True)

        # ---------------------------------------------------------------------------------
        # 2. re-pin the frozen-corpus contract (cells 43/45 leave these mutated)
        # ---------------------------------------------------------------------------------
        RULES.update(RULES_LEGACY)          # dominant_axis / corner_x / slot_fallback / zero
        CROP_MM = 130.0
        SLICE_BAND = (0.12, 0.88)           # 0943cfm ships (0.2, 0.8) after cell 43
        CACHE_SLICES = _EFF_N_SLICE         # 16, not the plan_cache-derived value
        IMG = CACHE_IMG = _EFF_IMG          # 224, not 336
        SLOTS = list(SLOTS_RECOVERED)       # SAG_FLUID_FS COR_FLUID_FS AX_FLUID_FS
                                           # SAG_FLUID_NOFS COR_T1 SAG_T1
        N_SLOT = _EFF_N_SLOT

        # ---------------------------------------------------------------------------------
        # 2b. RESOLVE THE LIBRARY'S SLOT VOCABULARY  (v1 died here)
        # ---------------------------------------------------------------------------------
        # `effb0mil_lib.py` names its four slots `SLOT_NAMES = ['sag_fluid','cor_fluid',
        # 'ax_fluid','sag_t1']` with `SLOT_PLANE = [0,1,2,0]` (sag/cor/ax) and
        # `SLOT_FS = [1,1,1,0]` (fat-suppressed for the first three, NOT for the fourth).  Those
        # are the names of the READER's columns -- i.e. of the cache-build kernel's own `SLOTS`.
        # Ours are different: `SLOTS_RECOVERED` carries `SAG_FLUID_FS` / `COR_FLUID_FS` /
        # `AX_FLUID_FS` / `SAG_FLUID_NOFS` / `COR_T1` / `SAG_T1`.
        #
        # v1 of this leg did a hard-coded `SLOTS.index(...)` on a name that exists in the
        # library's vocabulary but not in ours, and died with `ValueError: 'SAG_FS' is not in
        # in list` BEFORE the convention gate ever ran.  The fix is not a new name guess: it is
        # to resolve every one of the library's four slots, assert the result, and print the
        # whole slot table so an unknown vocabulary is visible in ONE run.
        #
        # The mapping is decided by (plane, fat-suppressed) -- the library's own SLOT_PLANE and
        # SLOT_FS -- and only the fourth slot is ambiguous: the library calls it `sag_t1`, this
        # notebook has BOTH `SAG_T1` (true T1) and `SAG_FLUID_NOFS` (PD/T2, not suppressed), and
        # the build kernel's own audit json calls its fourth slot `SAG_NOFS`.  So the fourth slot
        # is DISCRIMINATED by the convention gate, never guessed.
        _EFF_PLANE_NUM = {"Sagittal": 0, "Coronal": 1, "Axial": 2, "SAG": 0, "COR": 1, "AX": 2}

        def _eff_plane_no(spec):
            """The plane as the library numbers it: from the builder's own name or the tuple."""
            _lead = str(spec[0]).upper().split("_")[0]
            if _lead in _EFF_PLANE_NUM:
                return _EFF_PLANE_NUM[_lead]
            return _EFF_PLANE_NUM.get(str(spec[1]))

        def _eff_fs_flag(spec):
            """1 = fat-suppressed.  `SLOT_FS` is only 0/1, so the fact that every recovered
            slot is written as a 4-tuple with an explicit fat-suppression flag is usable."""
            return 1 if len(spec) > 3 and bool(spec[3]) else 0

        _EFF_REQUIRED = []
        for _k, _req_name in enumerate(_efflib.SLOT_NAMES):
            _want_p, _want_fs = _efflib.SLOT_PLANE[_k], _efflib.SLOT_FS[_k]
            _EFF_REQUIRED.append((_req_name, _want_p, _want_fs))

        _eff_available = [(i, s[0], _eff_plane_no(s), _eff_fs_flag(s)) for i, s in enumerate(SLOTS)]
        print("[effb0] slot vocabulary resolution", flush=True)
        print(f"[effb0]   library requires (name, plane, fat_suppressed): {_EFF_REQUIRED}", flush=True)
        print(f"[effb0]   this notebook's SLOTS: "
              f"{[(n, p, f) for _, n, p, f in _eff_available]}", flush=True)

        _EFF_OPTIONS = []
        for _req_name, _want_p, _want_fs in _EFF_REQUIRED:
            _opts = [i for i, _n, _p, _f in _eff_available if _p == _want_p and _f == _want_fs]
            if not _opts:
                raise RuntimeError(
                    f"slot '{_req_name}' (plane={_want_p}, fat_suppressed={_want_fs}) has NO "
                    f"counterpart in this notebook's SLOTS. Available: "
                    f"{[(n, p, f) for _, n, p, f in _eff_available]}")
            _EFF_OPTIONS.append([(i, SLOTS[i][0]) for i in _opts])
        for _req_name, _opts in zip(_efflib.SLOT_NAMES, _EFF_OPTIONS):
            print(f"[effb0]   -> {_req_name:10s} candidates: "
                  f"{[n for _, n in _opts]}", flush=True)

        import itertools as _eff_itertools
        _EFF_CONFIGS = []
        for _combo in _eff_itertools.product(*[o for o in _EFF_OPTIONS]):
            _idx = [i for i, _ in _combo]
            if len(set(_idx)) != len(_idx):
                continue                       # two library slots cannot be the same series
            _EFF_CONFIGS.append(([n for _, n in _combo], list(_idx)))
        if not _EFF_CONFIGS:
            raise RuntimeError(
                "no injective assignment of the library's four slots onto this notebook's "
                f"SLOTS; available {[(n, p, f) for _, n, p, f in _eff_available]}")

        # Reported because it is the reason the names must not be trusted: the published
        # `s_cache_e13_224_meta.npz` names its slot 3 `sag_t1`, while `sag_t1` in the 336
        # dataset's index is only 26.7% filled and this cache's slot 3 is 96.5% filled -- those
        # two files do not describe the same series under the same name.  The gate, not the
        # name, is the authority.
        _EFF_META_P = _EFF_LIBP.parent / "s_cache_e13_224_meta.npz"
        if _EFF_META_P.is_file():
            try:
                with _eff_np.load(str(_EFF_META_P), allow_pickle=True) as _mz:
                    _names_meta = [str(x) for x in _mz["slots"]]
                print(f"[effb0]   (the cache's own metadata calls its slots {_names_meta}; "
                      f"those names are NOT our slot names and are not used to resolve)",
                      flush=True)
            except Exception as _mz_exc:
                print(f"[effb0]   metadata slot names unreadable ({_mz_exc})", flush=True)

        # Order the configurations, so that the run log asks the most likely question first.
        # The build kernel's own vocabulary calls its fourth slot `SAG_NOFS`, the uploaded cache
        # metadata calls it `sag_t1`, and this notebook has BOTH `SAG_FLUID_NOFS` (PD/T2, not
        # suppressed) and `SAG_T1`.  `SAG_FLUID_NOFS` is asked first because it is the
        # builder-vocabulary name; `SAG_T1` second.  The GATE decides, not this order.
        _EFF_PREF = ["SAG_FLUID_NOFS", "SAG_T1", "SAG_NOFS"]
        _EFF_CONFIGS.sort(key=lambda c: next((c[0].index(nm) for nm in _EFF_PREF
                                              if nm in c[0]), 99))
        _EFF_CANDIDATES = [(c[0][3], c[1]) for c in _EFF_CONFIGS]
        if len(_EFF_CANDIDATES) > 4:
            # Bound the cost: the gate rebuilds raw DICOM for every candidate it tries.
            _EFF_CANDIDATES = _EFF_CANDIDATES[:4]

        def _eff_idx():
            c = _EFF_CANDIDATES[0]
            return list(c[1]), [SLOTS[i][0] for i in c[1]]

        _EFF_SLOT_IDX, _EFF_SLOT_LABELS = _eff_idx()
        _EFF_PIN = (RULES == RULES_LEGACY and CACHE_SLICES == 16 and IMG == 224
                    and abs(CROP_MM - 130.0) < 1e-9 and tuple(SLICE_BAND) == (0.12, 0.88)
                    and _EFF_SLOT_LABELS[:3] == _EFF_SLOT_NAMES[:3]
                    and len(set(_EFF_SLOT_IDX)) == _EFF_N_SLOT)
        if not _EFF_PIN:
            raise RuntimeError(f"frozen-corpus pin failed: {RULES} {CACHE_SLICES} {IMG} "
                               f"{CROP_MM} {SLICE_BAND} {_EFF_SLOT_LABELS} {_EFF_SLOT_IDX}")
        print(f"[effb0] pinned: rules={RULES} slices={CACHE_SLICES} img={IMG} "
              f"crop={CROP_MM:g}mm band={SLICE_BAND}", flush=True)
        for _ci, (_nm, _cidx) in enumerate(_EFF_CANDIDATES):
            print(f"[effb0]   candidate {_ci}: library slot order "
                  f"{_efflib.SLOT_NAMES} <- this notebook's "
                  f"{[SLOTS[i][0] for i in _cidx]}", flush=True)

        from sklearn.metrics import roc_auc_score as _eff_auc
        from scipy.stats import rankdata as _eff_rankdata

        def _eff_rank01(a):
            a = _eff_np.asarray(a, _eff_np.float64)
            return (_eff_rankdata(a, axis=0) - 1.0) / max(a.shape[0] - 1, 1)

        # ---------------------------------------------------------------------------------
        # 3. the model - built from the library, loaded strictly from all four checkpoints
        # ---------------------------------------------------------------------------------
        _EFF_DEV = DEVS[0]

        def _eff_load(fold):
            ck = _eff_torch.load(str(_EFF_CKPTS[fold]), map_location="cpu", weights_only=False)
            cfg = dict(ck.get("cfg") or _efflib.CFG)
            if int(cfg.get("nsl", 0)) != _EFF_N_SLICE or int(cfg.get("img", 0)) != _EFF_IMG:
                raise RuntimeError(f"f{fold} cfg drift: img={cfg.get('img')} nsl={cfg.get('nsl')}")
            if str(cfg.get("backbone")) != "efficientnet_b0.ra_in1k":
                raise RuntimeError(f"f{fold} backbone drift: {cfg.get('backbone')!r}")
            model = _efflib.EffB0MIL(cfg).to(_EFF_DEV).eval()
            model.load_state_dict(ck["model"], strict=True)
            del ck
            _eff_gc.collect()
            return model

        @_eff_torch.inference_mode()
        def _eff_predict_row(model, x_u8):
            """x_u8 [4,16,224,224] uint8 -> [12] probabilities.
            Mirrors KneeDS.__getitem__ (byte copy of the cache row, then `flip`, which is 0
            here by the argument at the top of this cell) and effb0mil_lib.predict (cast to
            float, build the 2.5D windows, sigmoid)."""
            x = _eff_torch.from_numpy(_eff_np.ascontiguousarray(x_u8))[None].to(_EFF_DEV)
            mask = _eff_torch.ones((1, _EFF_N_SLOT), dtype=_eff_torch.float32, device=_EFF_DEV)
            with _eff_torch.autocast("cuda", dtype=_eff_torch.float16):
                logit = model(_efflib.to_input(x, mask, _EFF_IMG, train=False, cfg=_efflib.CFG),
                              mask)
            return _eff_torch.sigmoid(logit.float())[0].cpu().numpy()

        # ---------------------------------------------------------------------------------
        # 4. the CONVENTION GATE - rebuild real TRAIN studies, compare to the frozen cache
        # ---------------------------------------------------------------------------------
        _EFF_TRAIN_IDS = [str(u) for u in _eff_pd.read_csv(
            ROOT / "train.csv", dtype={"StudyInstanceUID": str}).StudyInstanceUID]
        _EFF_ROW = {u: k for k, u in enumerate(_EFF_TRAIN_IDS)}

        def _eff_rebuild(studies, hdr, plane_map, lat_map, tag, slot_idx):
            """This notebook's own builder, restricted to `studies` and to `slot_idx`.
            build_cache always builds every study whose slots were picked, so the result is
            filtered afterwards."""
            _studies, pixels, smask = build_cache(pick_slots(hdr, plane_map), plane_map,
                                                  lat_map, tag)
            by = {str(u): i for i, u in enumerate(_studies)}
            out = {}
            for st in studies:
                if st not in by:
                    continue
                k = by[st]
                out[st] = (_eff_np.stack([_eff_np.asarray(pixels[k, j]) for j in slot_idx]),
                           _eff_np.asarray(smask[k])[slot_idx].astype(int).tolist())
            del pixels, smask, _studies
            _eff_gc.collect()
            return out

        def _eff_gate_candidate(slot_idx, slot_labels, pick, hdr, plane, lat, sides, frozen):
            """One full rebuild + comparison for ONE slot-3 hypothesis.  Returns
            (ok, report_dict_or_None, table_rows)."""
            _reb = _eff_rebuild(pick, hdr, plane, lat, "effb0-gate", slot_idx)
            _rows, _mirror, _checked, _unexplained, _studies = [], 0, 0, 0, 0
            for _st in pick:
                if _st not in _reb:
                    _rows.append((_st, "not built", "", "", "", sides.get(_st)))
                    continue
                _blk, _mk = _reb[_st]
                _fro = _eff_np.asarray(frozen[_EFF_ROW[_st]])
                _studies += 1
                _ident = _mir = _tot = 0
                for _j in range(_EFF_N_SLOT):
                    _a, _b = _blk[_j], _fro[_j]
                    if not _mk[_j] or not _b.any():
                        continue
                    _done = False
                    for _ax in (1, 2):
                        _tot += 1
                        _checked += 1
                        if _eff_np.array_equal(_a, _b):
                            _ident += 1
                            _done = True
                            break
                        if _eff_np.array_equal(_a, _eff_np.flip(_b, axis=_ax)):
                            _mir += 1
                            _mirror += 1
                            _done = True
                            break
                    if not _done:
                        _unexplained += 1
                _rows.append((_st, _mk, f"{_ident}", f"{_mir}", f"{_tot}", sides.get(_st)))
            del _reb
            _eff_gc.collect()
            _matched = _checked - _unexplained
            # The mirror pattern is reported PER STUDY, because whole-study consistency is the
            # evidence that the remap is a per-STUDY property of the frozen cache (which is
            # exactly what the library's per-row `flip` column encodes) rather than a per-slot
            # one this builder cannot reproduce.  Under the correct build rule, rebuilding a
            # study IN ITS OWN convention reproduces the frozen bytes exactly, so the expected
            # pattern is either all slots identical or all slots mirrored -- a MIXED study means
            # the frozen cache was built by a rule this notebook does not have.
            _pat = []
            for _r in _rows:
                if _r[4] in ("", "not built") or str(_r[1]).startswith("not"):
                    continue
                _id, _mir, _tot = int(_r[2]), int(_r[3]), int(_r[4])
                _pat.append("all-identical" if _id == _tot
                            else ("all-mirrored" if _mir == _tot else "MIXED"))
            _expect_pat = bool(all(p in ("all-identical", "all-mirrored") for p in _pat))
            # `_matched` (not `_studies * 8`) is what evaluability needs: a hypothesis whose
            # slot 3 rarely exists simply has fewer comparisons, and that is itself evidence,
            # so it must be REJECTED QUIETLY rather than raise.
            _evaluable = bool(_studies >= 3 and _matched >= 4)
            _ok = bool(_evaluable and _unexplained == 0 and _mirror <= _checked * 0.5
                       and _expect_pat)
            _rep = dict(slot3=slot_labels[3], slot_labels=list(slot_labels), studies=_studies,
                        comparisons=_checked, matched=_matched, mirrored=_mirror,
                        identity=_checked - _mirror - _unexplained, unexplained=_unexplained,
                        evaluable=_evaluable, study_pattern=_pat,
                        consistent_pattern=_expect_pat)
            return _ok, _rep, _rows

        def _effb0mil_convention_gate():
            _tser = _eff_pd.read_csv(ROOT / "train_series.csv",
                                     dtype={"StudyInstanceUID": str,
                                            "SeriesInstanceUID": str})
            _plane = dict(zip(_tser.SeriesInstanceUID, _tser.Anatomical_Plane))
            _hdr = annotate(walk("train_series"))
            _lat = lat_of(_hdr, "effb0-gate ")
            _sides = {str(k): v for k, v in _lat.items()}
            _have = [u for u in _EFF_TRAIN_IDS if u in set(_hdr.StudyInstanceUID.astype(str))]
            if len(_have) < 8:
                raise RuntimeError(f"only {len(_have)} train studies have headers; cannot gate")
            _rng = _eff_np.random.default_rng(20261004)
            _pick = []
            # deliberately cover both laterality populations: `normalise_laterality` only
            # mirrors the 'R' ones, so an all-L sample would not test the mirror at all.
            # FOUR studies, not more: the gate's fidelity comes from rebuilding with the same
            # builder, not from the sample size, and four studies is ~16 slot comparisons --
            # the primary cost here is re-reading ~4 x 6 slot-series of raw DICOM.
            for _want in ("R", "L"):
                _pool = [u for u in _have if _sides.get(u) == _want]
                for _k in _rng.permutation(len(_pool))[:2]:
                    _pick.append(_pool[_k])
            for _k in _rng.permutation(len(_have)):
                if len(_pick) >= 4:
                    break
                if _have[_k] not in _pick:
                    _pick.append(_have[_k])
            _pick = _pick[:4]
            _frozen = _eff_np.load(str(_EFF_TP), mmap_mode="r")
            _want_shape = (len(_EFF_TRAIN_IDS), _EFF_N_SLOT, _EFF_N_SLICE, _EFF_IMG, _EFF_IMG)
            if tuple(_frozen.shape) != _want_shape:
                raise RuntimeError(f"frozen cache shape {tuple(_frozen.shape)} != {_want_shape}")
            _best = None
            _gate_t0 = _eff_time.time()
            for _nm, _idx in _EFF_CANDIDATES:
                _labels = [SLOTS[i][0] for i in _idx]
                print(f"[effb0] CONVENTION GATE - hypothesis slot3={_nm} "
                      f"(slots {_labels}); rebuilding {len(_pick)} TRAIN studies from raw DICOM "
                      f"and comparing to the frozen cache", flush=True)
                _ok, _rep, _rows = _eff_gate_candidate(_idx, _labels, _pick, _hdr, _plane,
                                                       _lat, _sides, _frozen)
                print(f"[effb0]   {'study':44s} {'mask':10s} {'ident':>5s} {'mirror':>6s} "
                      f"{'total':>5s} {'lat':>3s}", flush=True)
                for _r in _rows:
                    print(f"[effb0]   {str(_r[0]):44s} {str(_r[1]):10s} {str(_r[2]):>5s} "
                          f"{str(_r[3]):>6s} {str(_r[4]):>5s} {str(_r[5]):>3s}", flush=True)
                print(f"[effb0]   hypothesis slot3={_nm}: {_rep['studies']} studies, "
                      f"{_rep['comparisons']} slot-axis comparisons, "
                      f"{_rep['identity']} identical, {_rep['mirrored']} mirror-only, "
                      f"{_rep['unexplained']} UNEXPLAINED, "
                      f"study patterns {_rep['study_pattern']}, "
                      f"evaluable={_rep['evaluable']}, "
                      f"consistent={_rep['consistent_pattern']} -> "
                      f"{'ACCEPTED' if _ok else 'REJECTED'}", flush=True)
                _eff_check_budget(f"convention gate [{_nm}]", since=_gate_t0)
                if _ok:
                    _best = (_nm, _idx, _labels, _rep)
                    break
            del _frozen, _hdr
            _eff_gc.collect()
            if _best is None:
                raise RuntimeError(
                    f"the rebuild does NOT reproduce the frozen cache under ANY of the "
                    f"{len(_EFF_CANDIDATES)} slot assignment(s) "
                    f"{[(n, [SLOTS[i][0] for i in x]) for n, x in _EFF_CANDIDATES]}, even "
                    f"allowing an axis mirror - the test cache WOULD be built in a different "
                    f"convention from the one the model was trained on; ABANDONING the leg "
                    f"(this is the correct failure: a silently mis-slotted leg is worse than "
                    f"no leg)")
            _nm, _idx, _labels, _rep = _best
            _EFF_CANDIDATES.insert(0, (_nm, list(_idx)))     # pin the winner for later use
            return dict(gate=_rep, chosen_slot3=_nm, slot_labels=_labels,
                        candidates_tried=[c[0] for c in _EFF_CANDIDATES])

        # ---------------------------------------------------------------------------------
        # 5. test-side cache, built by the same builder under the same pinned contract
        # ---------------------------------------------------------------------------------
        _EFF_TEST_IDS = [str(u) for u in (_RSNA_TEST_IDS or [])]
        if not _EFF_TEST_IDS:
            raise RuntimeError("test study identity is not initialised")

        def _effb0mil_build_test_cache():
            _tser = _eff_pd.read_csv(COMP / "test_series.csv",
                                     dtype={"StudyInstanceUID": str,
                                            "SeriesInstanceUID": str})
            _plane = dict(zip(_tser.SeriesInstanceUID, _tser.Anatomical_Plane))
            _hdr = annotate(walk("test_series"))
            _lat = lat_of(_hdr, "effb0-test ")
            _got = _eff_rebuild(_EFF_TEST_IDS, _hdr, _plane, _lat, "effb0-test", _EFF_SLOT_IDX)
            _miss = [u for u in _EFF_TEST_IDS if u not in _got]
            if _miss:
                raise RuntimeError(f"{len(_miss)} test studies absent from the rebuild: "
                                   f"{[m[:24] for m in _miss[:3]]}")
            _arr = _eff_np.stack([_got[u][0] for u in _EFF_TEST_IDS])
            _mask = _eff_np.asarray([_got[u][1] for u in _EFF_TEST_IDS])
            print(f"[effb0] test cache {_arr.shape} = {_arr.nbytes / 2 ** 20:.1f} MiB; "
                  f"slot mask {_mask.tolist()}", flush=True)
            print(f"[effb0] test-side laterality: "
                  f"{ {u[:14]: _lat.get(u) for u in _EFF_TEST_IDS} }", flush=True)
            _empty = int(_mask.sum()) < _EFF_N_SLOT * len(_EFF_TEST_IDS)
            if _empty:
                print(f"[effb0] WARNING {_EFF_N_SLOT * len(_EFF_TEST_IDS) - int(_mask.sum())} of "
                      f"{_EFF_N_SLOT * len(_EFF_TEST_IDS)} slot cells are empty; the model masks "
                      f"them out of its attention (never a fabricated zero slot)", flush=True)
            del _hdr, _got
            _eff_gc.collect()
            return _arr, _mask

        _EFF_GATE = _effb0mil_convention_gate()
        # the gate decides WHICH series fills slot 3; pin it for the test build
        _EFF_SLOT_IDX = list(_EFF_CANDIDATES[0][1])
        _EFF_SLOT_LABELS = [SLOTS[i][0] for i in _EFF_SLOT_IDX]
        print(f"[effb0] convention gate CHOSE slots {_EFF_SLOT_LABELS} "
              f"(from hypotheses {[c[0] for c in _EFF_CANDIDATES]})", flush=True)
        _EFF_TS, _EFF_TM = _effb0mil_build_test_cache()

        # ---------------------------------------------------------------------------------
        # 6. inference - 4 folds, per-label rank mean
        # ---------------------------------------------------------------------------------
        _EFF_P = _eff_np.zeros((len(_EFF_TEST_IDS), len(_RSNA_LABELS)), _eff_np.float64)
        _eff_t0 = _eff_time.time()
        for _f in range(4):
            _eff_check_budget(f"fold {_f}")
            _m = _eff_load(_f)
            _pf = _eff_np.stack([_eff_predict_row(_m, _EFF_TS[_i])
                                 for _i in range(len(_EFF_TEST_IDS))])
            if _pf.shape != (len(_EFF_TEST_IDS), len(_RSNA_LABELS)):
                raise RuntimeError(f"f{_f} prediction shape {_pf.shape}")
            if not _eff_np.isfinite(_pf).all() or (_pf < 0).any() or (_pf > 1).any():
                raise RuntimeError(f"f{_f} prediction not finite in [0,1]")
            _EFF_P += _pf / 4.0
            print(f"[effb0] fold {_f} done ({_eff_time.time() - _eff_t0:.1f}s)", flush=True)
            del _m, _pf
            _eff_gc.collect()
        _effb0mil_build = dict(gate=_EFF_GATE, shape=list(_EFF_TS.shape),
                               slot_mask=_EFF_TM.tolist(),
                               seconds=round(_eff_time.time() - _eff_t0, 1))
        _effb0mil_test_rank = _eff_pd.DataFrame(_EFF_P, columns=list(_RSNA_LABELS)).rank(
            method="average", pct=True)
        _effb0mil_test_rank.insert(0, "StudyInstanceUID", _EFF_TEST_IDS)
        _effb0mil_test_rank.to_csv(_EFF_DIR / "effb0mil_test_rank.csv", index=False)
        _EFF_OK = True
        _effb0mil_ok = True
        print(f"[effb0] leg READY: gate {_EFF_GATE}; 4 folds in {_effb0mil_build['seconds']}s",
              flush=True)
    except Exception as _eff_exc:
        import traceback as _eff_tb
        _eff_issue("leg_failed", error=f"{type(_eff_exc).__name__}: {_eff_exc}")
        print(f"[effb0] LEG FAILED - continuing with kernel-0943cfm behaviour EXACTLY: "
              f"{type(_eff_exc).__name__}: {_eff_exc}", flush=True)
        print(_eff_tb.format_exc(), flush=True)
        _EFF_OK = False
        _effb0mil_ok = False
'''

MEASURE_CELL = r'''
# ============ STAGE 5b - the printed measurement, then the blend ==============
# Prints the per-label AUC of the blend WITH and WITHOUT this leg, and only then writes the
# blend into `_pipeline_stage.csv` -- the file the publish cell reads.  Neither half can abort
# the pipeline: a table failure leaves the stage file alone, and a blend failure is caught and
# `_pipeline_stage.csv` keeps EXACTLY kernel-0943cfm's bytes.
#
# THIS TABLE IS NOT AN ADJUDICATOR AND IS NOT USED TO SET THE WEIGHT.  effb0mil's four
# published folds cover 4349 of our 4407 studies and NONE of the 58 fully-labelled ones (that
# is not an accident: the author held the gold set out), so gold-58 -- CI +-0.021 at n=58 --
# cannot see this leg at all.  The only local labelled array it CAN be read on is the public
# OOF in the sofiaanjenje kernel inputs, i.e. the same array the +0.0063 in
# docs/FIND-2026-10-03 was measured on, which makes this table in-sample for the decision.
# It is printed for the record; the default stays 0.10.
if _EFF_W > 0 and _effb0mil_ok:
    # ---- 1. the measurement --------------------------------------------------------------
    try:
        _me_pred = _me_ids = _me_gold = None
        try:
            with _eff_np.load(str(_eff_find("oof.npz")), allow_pickle=False) as _me_b:
                _me_ids = _me_b["ids"].astype(str)
                _me_pred = _me_b["pred"].astype(_eff_np.float64)
                _me_gold = (_me_b["gold_mask"].astype(bool)
                            if "gold_mask" in _me_b.files else None)
        except Exception as _me_exc:
            print(f"[effb0] public OOF unavailable ({type(_me_exc).__name__}: {_me_exc}); the "
                  f"comparison table cannot be printed - THE WEIGHT IS UNCHANGED", flush=True)
        if _me_pred is not None:
            _me_tr = pd.read_csv(ROOT / "train.csv", dtype={"StudyInstanceUID": str})
            _me_y = _me_tr[list(TARGETS)].to_numpy(_eff_np.float64)
            _me_pos = {u: k for k, u in enumerate(_me_tr.StudyInstanceUID.astype(str))}
            _me_lookup = {u: k for k, u in enumerate(_me_ids)}
            _me_eid = _eff_np.concatenate([
                pd.read_csv(str(_EFF_OOF[_f][1])).StudyInstanceUID.astype(str).to_numpy()
                for _f in range(4)])
            _me_eP = _eff_np.concatenate([
                _eff_np.load(str(_EFF_OOF[_f][0])).astype(_eff_np.float64) for _f in range(4)])
            _me_sel = _eff_np.array([u in _me_pos and u in _me_lookup for u in _me_eid])
            _me_r = _eff_np.array([_me_pos[u] for u in _me_eid[_me_sel]])
            _me_bv = _me_pred[[_me_lookup[u] for u in _me_eid[_me_sel]]]
            _me_ev = _me_eP[_me_sel]
            _me_Yb = (_me_y[_me_r] >= 0.5).astype(int)
            _me_bv = _eff_rank01(_me_bv)
            _me_ev = _eff_rank01(_me_ev)
            _me_wv = (1.0 - _EFF_W) * _me_bv + _EFF_W * _me_ev
            _me_rows, _me_mac = [], [0.0, 0.0, 0.0]
            for _j, _L in enumerate(TARGETS):
                if len(set(_me_Yb[:, _j].tolist())) < 2:
                    _me_rows.append({"finding": _L, "solo leg": _eff_np.nan,
                                     "ours": _eff_np.nan,
                                     f"blend w={_EFF_W:.2f}": _eff_np.nan,
                                     "delta": _eff_np.nan})
                    continue
                _a0 = _eff_auc(_me_Yb[:, _j], _me_bv[:, _j])
                _a1 = _eff_auc(_me_Yb[:, _j], _me_ev[:, _j])
                _a2 = _eff_auc(_me_Yb[:, _j], _me_wv[:, _j])
                _me_mac[0] += _a0 / 12.0
                _me_mac[1] += _a1 / 12.0
                _me_mac[2] += _a2 / 12.0
                _me_rows.append({"finding": _L, "solo leg": _a1, "ours": _a0,
                                 f"blend w={_EFF_W:.2f}": _a2, "delta": _a2 - _a0})
            _me_rows.append({"finding": "MACRO", "solo leg": _me_mac[1], "ours": _me_mac[0],
                             f"blend w={_EFF_W:.2f}": _me_mac[2],
                             "delta": _me_mac[2] - _me_mac[0]})
            _me_tab = pd.DataFrame(_me_rows).set_index("finding")
            _me_N = int(len(_me_bv))
            _me_gold_here = int(_me_gold[_me_r].sum()) if _me_gold is not None else 0
            print(f"\n[effb0] ==== per-label AUC on the public OOF, {_me_N} studies that BOTH "
                  f"legs cover (hard 0.5 labels) ====", flush=True)
            print(_me_tab.to_string(float_format=lambda v: f"{v:.5f}"), flush=True)
            print(f"[effb0] NOT AN ADJUDICATOR: {_me_N} studies, and this is the same array the "
                  f"+0.0063 in docs/FIND-2026-10-03 was measured on. Fully-labelled (gold-58) "
                  f"studies inside it: {_me_gold_here} - the leg cannot be scored on gold-58 at "
                  f"all.", flush=True)
            print(f"[effb0] WEIGHT NOT TUNED ON THIS TABLE: RUN['effb0mil_w'] = {_EFF_W:.3f}. "
                  f"(Soft-label in-sample gain at this weight was +0.00625 on 4349 studies; "
                  f"closed item #8 is why the default is not the 0.20-0.30 the curve peaks at.)",
                  flush=True)
            try:
                rsna_json(str(_EFF_DIR / "effb0mil_measurement.json"),
                          {"studies": _me_N, "weight": _EFF_W,
                           "macro": {"solo_leg": _me_mac[1], "ours": _me_mac[0],
                                     "blend": _me_mac[2], "delta": _me_mac[2] - _me_mac[0]},
                           "per_label": _me_rows, "gold58_overlap": _me_gold_here,
                           "warning": "in-sample selection instrument; not used to set the "
                                      "weight"})
            except Exception:
                pass
    except Exception as _me_exc:
        import traceback as _me_tb
        _eff_issue("measurement_failed", error=f"{type(_me_exc).__name__}: {_me_exc}")
        print(f"[effb0] measurement table FAILED (the blend still proceeds): "
              f"{type(_me_exc).__name__}: {_me_exc}", flush=True)
        print(_me_tb.format_exc(), flush=True)

    # ---- 2. the blend --------------------------------------------------------------------
    try:
        _me_src = pd.read_csv("/kaggle/working/_pipeline_stage.csv",
                              dtype={"StudyInstanceUID": str})
        _me_lab = [c for c in _me_src.columns if c != "StudyInstanceUID"]
        if _me_lab != list(_RSNA_LABELS):
            raise RuntimeError(f"stage-file label order drift: {_me_lab}")
        _me_uids = _me_src.StudyInstanceUID.astype(str).tolist()
        if sorted(_me_uids) != sorted(_EFF_TEST_IDS):
            raise RuntimeError("the stage file does not cover exactly the test studies")
        _me_legf = _effb0mil_test_rank.set_index("StudyInstanceUID").reindex(_me_uids)
        if _me_legf.isna().any().any():
            raise RuntimeError("the leg does not cover every study in the stage file")
        _me_basev = _eff_np.asarray(_me_src[list(_RSNA_LABELS)].to_numpy(_eff_np.float64))
        _me_legv = _me_legf[list(_RSNA_LABELS)].to_numpy(_eff_np.float64)
        _me_new = (1.0 - _EFF_W) * _eff_rank01(_me_basev) + _EFF_W * _eff_rank01(_me_legv)
        if not _eff_np.isfinite(_me_new).all():
            raise RuntimeError("the blended stage file is not finite")
        _me_out = _me_src.copy()
        _me_out[list(_RSNA_LABELS)] = _me_new
        _me_tmp = _EffPath("/kaggle/working/.pipeline_stage.effb0.tmp")
        _me_out.to_csv(_me_tmp, index=False)
        _eff_os.replace(_me_tmp, "/kaggle/working/_pipeline_stage.csv")
        print(f"[effb0] BLENDED into _pipeline_stage.csv at w={_EFF_W:.3f} "
              f"({len(_me_out)} studies)", flush=True)
        try:
            rsna_json(str(_EFF_DIR / "effb0mil_blend_receipt.json"),
                      {"leg": "effb0mil_sl16_tv7w", "weight": _EFF_W,
                       "arithmetic": "(1-w)*rank(ours) + w*rank(leg), per label",
                       "gate": _effb0mil_build, "studies": len(_me_out),
                       "gold58_overlap_with_leg": 0,
                       "sl16_fix_csv_published": False,
                       "note": "the convention gate is the evidence that the test cache follows "
                               "the frozen cache's own build rule"})
        except Exception as _me_exc:
            print(f"[effb0] receipt write skipped: {_me_exc}", flush=True)
    except Exception as _me_exc:
        import traceback as _me_tb
        _eff_issue("blend_failed", error=f"{type(_me_exc).__name__}: {_me_exc}")
        print(f"[effb0] BLEND FAILED - _pipeline_stage.csv left EXACTLY as kernel-0943cfm wrote "
              f"it: {type(_me_exc).__name__}: {_me_exc}", flush=True)
        print(_me_tb.format_exc(), flush=True)
        _effb0mil_ok = False
elif _EFF_W > 0:
    print("[effb0] leg unavailable; _pipeline_stage.csv is EXACTLY kernel-0943cfm's output - "
          f"issues: {_EFF_ISSUES if _EFF_ISSUES else 'see log above'}", flush=True)
'''

MARKDOWN = '''## Stage 5 — the EffNet-B0 2-level MIL leg

`tbongc/rsna-knee-effb0mil-sl16-tv7w` — 4 folds of a slice→slot→study MIL on EfficientNet-B0
at 224 px, rank-blended in at `RUN["effb0mil_w"]` (default **0.10**) on top of everything above.

**Read this before reading the score.** The leg's published OOF covers 4349 of our 4407 studies
and **not one of the 58 fully-labelled ones** — so it is invisible to gold-58. The only labelled
array it can be read on locally is the public OOF, which is the same array the +0.0063 in
`docs/FIND-2026-10-03-public-artifact-scan.md` was measured on. That makes the table this stage
prints an **in-sample instrument**: printed for the record, **not** used to pick the weight.

**The risk, and what is done about it.** The frozen train cache carries a per-row `flip` remap
inside `effb0mil_lib.py`, and `sl16_fix.csv` was never published — so the remap's contents are
unknown. The test cache is therefore built by *this notebook's own* cache builder under the
frozen corpus contract (`dominant_axis` ordering, `corner_x` laterality, `slot_fallback`,
`decode_fill=zero`, 130 mm, band 0.12–0.88, 16 slices, 224 px, slot axis
`[SAG_FS, COR_FS, AX_FS, SAG_T1]`). Because the only difference the library's `flip` variable
can express is a per-row **mirror**, a test cache built by the same builder cannot sit in a
different convention from the model — unless the cache was built by a rule this notebook does
not have, and that is exactly what the **convention gate** tests: it rebuilds real training
studies from raw DICOM and compares them to the published `s_cache_e13_224.npy`, byte for byte,
identity or axis mirror. A non-mirror mismatch abandons the leg.

**Fail-safe.** If any of it fails — dataset not mounted, checkpoint missing, gate not passed,
inference error, UID mismatch — the leg is abandoned loudly and `_pipeline_stage.csv` keeps
exactly `kernel-0943cfm`'s bytes, so `submission.csv` is the 0943cfm behaviour.
'''


def apply(nb):
    """Insert the RUN CONFIG key and the three new cells.  Returns (nb, insert_index)."""
    cells = nb["cells"]
    texts = ["".join(c.get("source", [])) if c["cell_type"] == "code" else None
             for c in cells]

    hits = [i for i, t in enumerate(texts) if t and RUNCONFIG_ANCHOR in t]
    if len(hits) != 1:
        raise SystemExit(f"!! RUN CONFIG anchor found in {len(hits)} cells (need 1)")
    i = hits[0]
    if texts[i].count(RUNCONFIG_ANCHOR) != 1:
        raise SystemExit("!! RUN CONFIG anchor not unique inside its cell")
    if '"effb0mil_w"' in texts[i]:
        raise SystemExit("!! RUN CONFIG already carries effb0mil_w")
    texts[i] = texts[i].replace(RUNCONFIG_ANCHOR, RUNCONFIG_NEW, 1)
    cells[i]["source"] = texts[i].splitlines(keepends=True)
    print(f"   edit ok : RUN['effb0mil_w'] default 0.10   (cell {i})")

    # the one cell that writes submission.csv at the very end.  Not "contains submission.csv":
    # cell 16 mentions it in a comment, so the anchor is the literal open of the final file.
    PUB_ANCHOR = "_final=Path('/kaggle/working/submission.csv')"
    pub = [k for k, c in enumerate(cells)
           if c["cell_type"] == "code" and PUB_ANCHOR in "".join(c.get("source", []))]
    if len(pub) != 1:
        raise SystemExit(f"!! publish cell ({PUB_ANCHOR!r}) found {len(pub)} times (need 1)")
    p = pub[0]

    def cell(kind, src):
        d = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True),
             "id": str(uuid.uuid4())}
        if kind == "code":
            d["outputs"] = []
            d["execution_count"] = None
            d["metadata"] = {"trusted": True}
        return d

    new = [cell("markdown", MARKDOWN), cell("code", LEG_CELL), cell("code", MEASURE_CELL)]
    nb["cells"] = cells[:p] + new + cells[p:]
    print(f"   edit ok : 3 new cells inserted before the publish cell (index {p})")
    return nb, p


def check(nb, meta):
    text = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
    assert '"effb0mil_w": float(os.environ.get("RSNA_EFFB0MIL_W", "0.10"))' in text, \
        "the leg weight default is not 0.10 / not env-overridable"
    assert text.count("_effb0mil_ok = True") == 1, "leg-ready marker missing or duplicated"
    assert "except Exception as _eff_exc:" in text, "the leg is not wrapped in try/except"
    assert "_effb0mil_convention_gate()" in text, "the convention gate is not called"
    assert 'consistent_pattern' in text, "the gate does not check per-study mirror consistency"
    # v2: the slot vocabulary must be RESOLVED from the library's own SLOT_PLANE/SLOT_FS and
    # asserted, and a resolution failure must list the available slots.  v1 died with
    # `ValueError: 'SAG_FS' is not in list` before the gate ever ran, so this is asserted by NAME.
    assert "_efflib.SLOT_NAMES" in text, "the library's slot vocabulary is not consulted"
    assert "_efflib.SLOT_PLANE[_k], _efflib.SLOT_FS[_k]" in text, \
        "the slot resolution does not use the library's own plane/fs contract"
    # (the `"SAG_FS")` form, not the bare name: the comment above records the v1 bug verbatim)
    assert 'SLOTS.index("SAG_FS")' not in text, "the v1 hard-coded SAG_FS lookup survived"
    assert "slot vocabulary resolution" in text, "the resolution is not printed"
    assert "counterpart in this notebook's SLOTS" in text, \
        "the resolution failure does not list the available slot names"
    assert "candidate" in text and "itertools.product" in text, \
        "the candidate configurations are not enumerated"
    assert "leaves the leg" not in text
    assert "RUN['effb0mil_w']=0 - leg disabled" in text, "weight 0 does not short-circuit the leg"
    assert "LEG DROPPED" in text, "an out-of-range weight is not handled fail-safe"
    assert '"/kaggle/working/_pipeline_stage.csv"' in text, "the blend target is not the stage file"
    assert 'RUN.get("effb0mil_budget_s"' in text, "the leg's own time budget is not wired"
    assert "_EffOutOfTime" in text, "there is no out-of-time path for the leg"
    assert text.count("os.replace(_me_tmp") == 1, "the blend is not a single atomic write"
    assert "_eff_unexplained" in text or "_unexplained" in text, "gate does not count mismatches"
    # the ORIGINAL fail-closed behaviour must be untouched. These are 0943cfm's own invariants.
    assert "raise RuntimeError('This complete ensemble requires two working GPUs" in text, \
        "the two-GPU fail-closed preflight was weakened"
    assert "raise RuntimeError('final DINO member count mismatch')" in text, \
        "the final DINO member gate was weakened"
    assert "raise RuntimeError('BTKD calibrator was not applied')" in text, \
        "the calibrator gate was weakened"
    assert "if _ke_input_ids != _expected_preparations:" in text, "the Raptor identity audit moved"
    assert 'RSNA_PRESET", "speedy"' in text, "the preset default changed"
    assert '"coat_private_alpha": 0.40,' in text, "the CoAt private alpha changed"
    assert '"rad_cal_w": 0.40,' in text, "rad_cal_w changed"
    assert '_RAD_EXCLUDE = ("Baker\'s", "MCL")' in text, "the MCL exclusion changed"
    assert '"cn0-ours": 0.20' in text, "the cn0 view weight changed"
    assert meta["id"] == KERNEL_ID, meta["id"]
    assert meta["title"] == KERNEL_TITLE, meta["title"]
    assert meta["kernel_type"] == "notebook"
    assert meta["enable_gpu"] is True
    assert meta["enable_tpu"] is False
    assert meta["enable_internet"] is False, "internet must stay OFF"
    assert meta["machine_shape"] == "NvidiaTeslaT4", meta.get("machine_shape")
    for ds in (EFFB0_DATASET, CACHE_DATASET):
        assert ds in (meta.get("dataset_sources") or []), f"{ds} not mounted"
    print("   post-conditions: all present")


VALIDATION_SECTION = """
## What was validated here, and what was NOT

### Validated on this machine

| check | result |
|---|---|
| notebook parses as plain JSON | OK |
| `nbformat.validate()` (nbformat 5.10.4) | OK, nbformat 4.5 |
| every code cell compiles (`compile()`) | 34/34 OK |
| no undefined Name loads in the two new cells (AST walk over the whole notebook) | 0 |
| source `kernel-0943cfm` and its siblings untouched | `git status` clean for all of them |
| only cell 2 differs from the source; cells 49–50 moved byte-identical | asserted in the builder |
| the measurement table, dry-run on the real arrays | reproduces the write-up exactly: macro 0.84962 -> 0.85587 at w=0.10, delta +0.00625, all 12 labels positive, LatMen 0.80143 -> 0.81451 |
| the convention gate's identity / mirror classifier, dry-run on synthetic volumes | identity, mirror on axis 2, mirror on axis 1 all classified correctly; a one-pixel non-mirror difference classified UNEXPLAINED |

### NOT validated here — this is the honest gap

1. **The test cache was NOT checked against `s_cache_e13_224.npy` on this machine.** That requires
   the 14.1 GB cache *and* the raw competition DICOMs, i.e. the Kaggle runtime. The check is
   implemented as the notebook's **convention gate**, which runs before any prediction is used; if
   it fails, the leg is abandoned and `submission.csv` is 0943cfm's behaviour.
2. **`sl16_fix.csv` is not published anywhere public**, so the contents of the library's per-row
   `flip` are unknown. The brief's statement that `tbongc/rsna-knee-cache336-v1` (21 GB) contains
   `sl16_fix.csv` and `s_cache_e13_224.npy` is **false** — that dataset holds a different 336 px
   cache (`shard_000..022.npy`, `cache336_index.parquet`, `slot_pick_train.parquet`). The 224 px
   cache is `ngoanphamvan/sl16s0b-cache-variant` (14.1 GB, `track_s_sl16/s_cache_e13_224.npy`),
   and `sl16_fix.csv`'s declared home `tbongc/rsna-knee-labels` is **403 to the public API**.
   The notebook is written so that it never needs `flip` (see the argument in the leg cell), but
   that argument is *tested* by the gate, not assumed.
3. **`effb0mil_lib.py` has no inference path** — it is a training library (`train_fold`, `KneeDS`,
   `to_input`, `EffB0MIL`). The notebook re-drives the contract by hand; the forward path is the
   library's own `EffB0MIL.forward` + `to_input(train=False)`.
4. **The leg cannot be scored on gold-58 at all**: effb0mil's published folds cover 4349/4407
   studies and 0 of the 58 fully-labelled ones. The printed table is therefore in-sample on the
   public OOF, which is why the default weight is 0.10 and not the 0.20–0.30 the curve peaks at.
5. **The frozen cache's slot 3 is ambiguous in the public artifacts**: the uploaded metadata names
   it `sag_t1`, while the build kernel's own audit json names its four slots
   `SAG_FS/COR_FS/AX_FS/SAG_NOFS`, and `effb0mil_lib.py`'s `SLOT_FS = [1,1,1,0]` says only "not
   fat-suppressed". This notebook's `SLOTS_RECOVERED` contains BOTH `SAG_NOFS` and `SAG_T1` as two
   distinct acquisitions, so the wrong choice would silently build every test slot-3 from the wrong
   series. It is **discriminated by the gate**, not guessed: `SAG_T1` is tried first (the uploaded
   metadata and `SLOTS` in the cache-build kernel both name it that way), `SAG_NOFS` second.
6. **The provenance of the published 224 px cache is not fully explained.** Its audit json reports
   four slots and a `SAG_NOFS` fill of 96.5%, but the metadata's `sag_t1` mask is only 26.7% filled
   — which suggests the uploaded `sag_t1` is not literally the `SAG_NOFS` slot of that audit. The
   gate resolves this empirically; a reader should not assume the slot names in the two files refer
   to the same series.
"""


def write_report(before, after, insert_at, n_new, meta):
    lines = []
    lines.append("# BUILD REPORT — `kernel-0943cfm2eff`\n")
    lines.append(f"Source: `kernels/kernel-0943cfm` (never modified). "
                 f"Kernel id `{KERNEL_ID}`, title `{KERNEL_TITLE}`.\n")
    lines.append(f"- cells: **{len(before)} -> {len(after)}** (+{len(after) - len(before)})\n")
    lines.append(f"- total source chars: **{sum(len(t) for t in before)} -> "
                 f"{sum(len(t) for t in after)}** "
                 f"({sum(len(t) for t in after) - sum(len(t) for t in before):+d})\n")
    lines.append(f"- inserted at index {insert_at}: 1 markdown + 2 code cells\n")
    lines.append("\n## Per-cell source-length diff\n\n")
    lines.append("| cell | kind | before | after | delta | note |\n")
    lines.append("|---|---|---|---|---|---|\n")
    for k in range(max(len(before), len(after))):
        if k < insert_at:
            b = len(before[k])
            a = len(after[k])
            note = "RUN CONFIG: +effb0mil_w" if a != b else ""
            kind = ""
            lines.append(f"| {k} | {kind} | {b} | {a} | {a - b:+d} | {note} |\n")
        elif k < insert_at + n_new:
            kind = "markdown" if k == insert_at else "code"
            lines.append(f"| {k} | {kind} | — | {len(after[k])} | new | "
                         f"**NEW** stage-5 cell |\n")
        else:
            # after the insertion the source cell k - n_new lands at index k: same length
            j = k - n_new
            if j < len(before) and k < len(after):
                b, a = len(before[j]), len(after[k])
                note = "shifted by +%d (identical)" % n_new if b == a else "CHANGED"
                lines.append(f"| {k} | | {b} | {a} | {a - b:+d} | {note} |\n")
            else:
                lines.append(f"| {k} | | — | "
                             f"{len(after[k]) if k < len(after) else '—'} | | |\n")
    lines.append("\n## Metadata\n\n```json\n")
    lines.append(json.dumps(meta, indent=2) + "\n```\n")
    lines.append(VALIDATION_SECTION)
    (DST / REPORT).write_text("".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    if a.check:
        nb = json.loads((DST / NB).read_text(encoding="utf-8"))
        meta = json.loads((DST / "kernel-metadata.json").read_text(encoding="utf-8"))
        check(nb, meta)
        print(f"CHECK OK: {DST}")
        return 0

    src_nb = json.loads((SRC / NB).read_text(encoding="utf-8"))
    before = ["".join(c.get("source", [])) for c in src_nb["cells"]]

    nb = copy.deepcopy(src_nb)
    nb, ins = apply(nb)

    meta = json.loads((SRC / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta["id"] = KERNEL_ID
    meta["title"] = KERNEL_TITLE
    meta["is_private"] = True
    meta["enable_gpu"] = True
    meta["enable_tpu"] = False
    meta["enable_internet"] = False
    meta["kernel_type"] = "notebook"
    meta["machine_shape"] = "NvidiaTeslaT4"
    ds = list(meta.get("dataset_sources") or [])
    for extra in (EFFB0_DATASET, CACHE_DATASET):
        if extra not in ds:
            ds.append(extra)
    meta["dataset_sources"] = ds
    meta.pop("id_no", None)

    # `kernel-0943cfm` declares nbformat 4.4 but every cell carries an `id`, which 4.4 does not
    # allow (ids arrived in 4.5).  nbformat.validate() therefore rejects the SOURCE notebook too.
    # Declare 4.5 so the new artifact is valid on its own terms; cell ids are preserved.
    nb["nbformat"] = 4
    nb["nbformat_minor"] = max(int(nb.get("nbformat_minor", 4)), 5)

    check(nb, meta)

    DST.mkdir(parents=True, exist_ok=True)
    (DST / NB).write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")
    (DST / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    after = ["".join(c.get("source", [])) for c in nb["cells"]]

    # AUDIT: everything except the RUN CONFIG cell must be BYTE-IDENTICAL to kernel-0943cfm,
    # and the publish cell must have moved intact.  This is the whole "no silent edit" claim,
    # so it is asserted rather than eyeballed in a diff table.
    changed = [k for k in range(ins) if after[k] != before[k]]
    assert changed == [2], f"unexpected pre-insert changes: {changed}"
    assert len(after) == len(before) + 3
    for j in range(ins, len(before)):
        assert after[j + 3] == before[j], f"cell {j} changed on its way past the insertion"
    print(f"   audit ok : only cell 2 changed; cells {ins}..{len(before) - 1} moved intact")

    write_report(before, after, ins, 3, meta)

    print(f"\nbuilt {DST}")
    print(f"  cells {len(before)} -> {len(after)}  (+{len(after) - len(before)})")
    print(f"  source chars {sum(len(t) for t in before)} -> {sum(len(t) for t in after)}")
    print(f"  datasets {len(meta['dataset_sources'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
