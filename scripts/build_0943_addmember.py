# -*- coding: utf-8 -*-
"""Build `kernel-0943c`: the validated 0943 recipe with OUR OWN CoAtNet member added as a 5th view.

WHY A NEW ARM AND NOT AN EDIT
    `kernels/kernel-0943` is the recipe that scored 0.943 and whose hidden rerun is proven to
    complete.  Editing it would put that at risk for a change worth about +0.002 at a 0.917 base,
    so this builder copies it and adds one view.  `kernel-0943` is not touched.

WHAT IS ADDED, AND THE FIVE EDITS IT TAKES
    Our `cn0` (CoAtNet-rmlp-2_rw_384 @384, trained on the rented box, rescued to the Kaggle dataset
    `ppxl16/rsna-knee-cn0-weights`) is the one arm this repo has measured ABOVE the frontier's own
    DINOv2 seed members (0.8769 vs 0.8676-0.8697, scripts/arm_parity.py), and adding it to a
    0.917-quality blend measured +0.0028 with the weight taken out of `public20`
    (scripts/member_add_test.py, FACTS A131/A137).

    The submission notebook already contains the machinery for exactly this kind of member -- the
    same `RaptorClassifier`, the same `build_backbone`, the same `eval_windows`, and an `ARMS` table
    whose four entries are ALL `coatnet_rmlp_2_rw_384`.  So the work is registration, not porting:

      1. `ARMS` gains a fifth entry (same arch/img/slots/span as `native384-v8`, because cn0 trained
         on the 44-slot corpus at res 384).
      2. `RUN["raptor_view_w"]` gains its weight.  The blend normalises by the sum of weights
         (cell 45: `weights /= weights.sum()`), so 0.20 gives it an effective share of 0.2/1.2.
      3. the windowing call site honours a per-arm `norm`, and
      4. the `raptor_k_eval` override honours a per-arm `k_eval_fixed`;
      5. `gpu_one` runs it on the GPU that already runs the single arms.
    and the kernel mounts the weights dataset.

    ⚠️ EDITS 3 AND 4 ARE THE POINT.  Both are cases where the notebook's global setting is right for
    the frontier's views and wrong for ours, and both would degrade our member SILENTLY:
      * `NORM = "imagenet"` there vs `--norm none` (the trainer default, [0,1]) for ours.  The only
        arm in this repo that ever ran imagenet norm is the dedicated `norm0` ablation, so every
        other arm -- including cn0 -- used the default.  Measured from the kernels, FACTS A143.
      * `RUN["raptor_k_eval"] = 94` overrides every view's own k_eval, but 24 is what cn0 was SCORED
        at (its OOF used `k_eval=24`), and it is 4x cheaper -- which matters because this stage is
        fail-closed on the 8-hour budget (`rsna_deadline`).

    ⚠️ The mapping from the measured "+0.0028 at weight 0.2" to "0.2 inside the CoAt arm" is not
    exact: the measurement blended cn0 against the transformer-stack stand-in, while here it joins
    the CoAt arm, which is itself blended against the stack at `coatnet_w` ~0.65.  Its effective
    weight therefore lands near 0.65 * 0.167 ~ 0.11 rather than 0.2 -- same direction, smaller dose.

Usage:
    python scripts/build_0943_addmember.py            # build (asserts every edit)
    python scripts/build_0943_addmember.py --check     # re-assert the built artifact only
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "kernels" / "kernel-0943"
DST = ROOT / "kernels" / "kernel-0943c"
NB = "rsna-knee-restructured-version-3.ipynb"
MEMBER_DATASET = "ppxl16/rsna-knee-cn0-weights"

# ---- the five edits, as whitespace-tolerant regexes (each must match exactly once) -------------
# Literal strings were tried first and failed on indentation: a notebook cell is a list of lines and
# the indentation inside a dict is easy to miscount.  The discipline is unchanged -- EXACTLY one
# match, or refuse to write -- but the match no longer depends on counting spaces by eye.
VIEW_W_OLD = r'"raptor_view_w":\s*\{[^}]*\}'
VIEW_W_NEW = ('"raptor_view_w": {"maxspan-v5": 0.60, "native384dense-v10": 0.10,\n'
              '                      "maxspan-v5-reverse": 0.10, "native384-v8": 0.20,\n'
              '                      "cn0-ours": 0.20}')

ARM4_OLD = r'(\{"name": "native384-v8".*?"reverse": False, "w": 0\.20\},)\s*\]'
ARM4_NEW = r'''\1
    # ADDED 2026-09-26 by scripts/build_0943_addmember.py.  Our own CoAtNet member, trained on the
    # rented box, rescued to ppxl16/rsna-knee-cn0-weights.  Geometry copied from native384-v8 above
    # (same arch/img/slots/span) because cn0 trained on the 44-slot corpus at res 384.  The two
    # extra keys are not cosmetic: both override a notebook-wide default that is right for the
    # frontier's views and wrong for ours, and both would degrade this member SILENTLY.
    {"name": "cn0-ours", "file": "raptor_ft_cn0.pt",
     "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384,
     "img": 384, "slots": _SLOTS44, "span": (0.06, 0.94), "k_eval": 24,
     "k_eval_fixed": 24, "norm": "none",
     "reverse": False, "w": 0.20},
]'''

NORM_OLD = r"norm=_KE_NS\['NORM'\]\)"
NORM_NEW = "norm=arm.get('norm', _KE_NS['NORM']))"

KEVAL_OLD = (r"for _arm in _KE_NS\['ARMS'\]:\s*\n"
             r"\s*_arm\['parent_k_eval'\] = _arm\['k_eval'\]\s*\n"
             r"\s*_arm\['k_eval'\] = int\(RUN\['raptor_k_eval'\] or _arm\['parent_k_eval'\]\)")
KEVAL_NEW = ("for _arm in _KE_NS['ARMS']:\n"
             "    _arm['parent_k_eval'] = _arm['k_eval']\n"
             "    # a view may pin its own window count (`k_eval_fixed`): RUN['raptor_k_eval']=94 is\n"
             "    # right for the frontier's views, but our member was SCORED at 24 and is 4x cheaper.\n"
             "    _arm['k_eval'] = int(_arm.get('k_eval_fixed') or RUN['raptor_k_eval']\n"
             "                         or _arm['parent_k_eval'])")

# ⚠️ The added view goes on GPU 0, NOT GPU 1.  Measured on the visible set: putting it on GPU 1
# (which already runs arms 1 and 3) took the Raptor stage from 28.9 s to 44.0 s -- +52%, because
# GPU 1 went from 2 arms to 3 while GPU 0 kept only the shared MaxSpan pass.  GPU 0 loads ONE model
# and does two forward passes per study (normal + flipped), so it is the lighter side; riding it
# costs much less critical-path time than making GPU 1 the 3-arm straggler.  This stage is
# fail-closed on the 8-hour budget, which is the one risk that cannot be fixed after the fact.
GPU0_OLD = (r"(def gpu_zero\(\):\s*\n"
            r"\s*with torch\.cuda\.device\(0\):\s*\n"
            r"\s*run_shared_maxspan\(torch\.device\('cuda:0'\)\))")
GPU0_NEW = (r"\1\n"
            r"            run_single(4, torch.device('cuda:0'))   # our added cn0 view (index 4)")

ASSET_OLD = r"(ASSET_ROOTS = \[)(.*?)(\])"
ASSET_NEW = (r"\1\2, '/kaggle/input/datasets/ppxl16/rsna-knee-cn0-weights', "
             r"'/kaggle/input/rsna-knee-cn0-weights'\3")

# The SEVENTH edit, only for the `-cf` flavour: stack the second thing this repo has a *measurement*
# for on top of the first.  `_RAD_EXCLUDE` dropping 'Fracture' is worth +0.001 macro at n=4185
# (C81), and the member is worth +0.0028 on a 0.917 base (A137).  They act on different parts of
# the pipeline (the Rad blend's label set vs the CoAt view set), so the two are at least plausibly
# additive -- which is the only reason it is worth a slot rather than a knob.
# ⚠️ The replacement must NOT be a raw string.  The first version used r'_RAD_EXCLUDE = ("Baker\'s",)'
# and wrote the backslash into the notebook, producing `("Baker\'s",)` -- a plain string with an
# escaped apostrophe instead of a 1-element tuple, i.e. exactly the silent substring-matching bug the
# preflight's own comment warns about.  `check()` caught it because it asserts the RESULT, not the edit.
RADEX_OLD = r"_RAD_EXCLUDE = \(\"Baker's\", 'Fracture'\)"
RADEX_NEW = "_RAD_EXCLUDE = (\"Baker's\",)"

# ---- the routed variant: member + fracture + the FLATTENED outer routing -------------------
# WHY `0943cflat` EXISTS.  The pre-registered pair is `0943cf` + `0943p`, but those two differ on
# THREE axes at once (routing, our member, the fracture fix): if `0943p` wins we cannot say why, and
# that arm carries NEITHER of the two changes this repo has measurements for.  `0943cflat` differs
# from `0943cf` on EXACTLY ONE axis -- routing -- so the final pair becomes a clean single-axis
# hedge with the measured changes in BOTH arms.  It also weakly dominates `0943p`: same flattened
# routing, plus the +0.0028 member and the +0.00099 fracture fix (A131/A137, C81).
# ⚠️ It is still a CONCENTRATED bet if only one slot is ever scored -- which is why this ADDS an
# option rather than replacing the recommendation (docs/FINAL-SELECTION.md).
# ⚠️⚠️ THE ANCHORS IN `apply_edits` ARE REGEXES, NOT LITERALS.  This bit me here: I first wrote
# PRESET_OLD as the plain text 'PRESET = os.environ.get("RSNA_PRESET", "speedy")', and its unescaped
# `(` and `)` were consumed as a GROUP DELIMITER, so the pattern required `get` followed immediately
# by `"RSNA_PRESET"...` while the text has `get("RSNA_PRESET"...` -- the parens are never matched.
# `old in text` was True while `re.search(old, text)` was None, which is exactly the confusing part.
# The builder's raise ("found in 0 cells") is what caught it; use re.escape for literal anchors.
PRESET_SPEEDY_TXT = 'PRESET = os.environ.get("RSNA_PRESET", "speedy")'
PRESET_PARENT_TXT = 'PRESET = os.environ.get("RSNA_PRESET", "parent")'
PRESET_OLD = re.escape(PRESET_SPEEDY_TXT)
PRESET_NEW = PRESET_PARENT_TXT

EDITS = [
    # ⚠️ SIXTH EDIT, and the first smoke run is why it exists.  `_asset_find_asset` does NOT search
    # /kaggle/input; it walks an explicit whitelist, `ASSET_ROOTS`.  So mounting the dataset is
    # NECESSARY BUT NOT SUFFICIENT: v1 of this arm pushed, ran the whole DINO stage (20/20
    # fingerprints, fail-closed gate PASS) and then died fail-closed in 2.5 minutes with
    #     RuntimeError: expected one pinned asset raptor_ft_cn0.pt, got []
    # which is a good failure -- cheap, early, and it names the missing thing.
    ("ASSET_ROOTS whitelists our dataset", ASSET_OLD, ASSET_NEW),
    ("RUN['raptor_view_w'] gains cn0-ours", VIEW_W_OLD, VIEW_W_NEW),
    ("ARMS gains the cn0-ours entry", ARM4_OLD, ARM4_NEW),
    ("windowing honours a per-arm norm", NORM_OLD, NORM_NEW),
    ("k_eval override honours k_eval_fixed", KEVAL_OLD, KEVAL_NEW),
    ("gpu_zero runs index 4 (the lighter GPU)", GPU0_OLD, GPU0_NEW),
]


def apply_edits(texts, extra_fracture=False, flat=False):
    """Apply each edit to the ONE cell that contains it.

    ⚠️ The first version called this per cell and required all five edits inside that cell -- but
    they live in different cells (the view weights in cell 2, the rest in cell 45), so it
    refused on cell 0.  Correct shape: locate the unique cell per edit, then assert n == 1 within it.
    """
    edits = list(EDITS)
    if extra_fracture:
        edits.append(("_RAD_EXCLUDE drops 'Fracture' (the C81 fix)", RADEX_OLD, RADEX_NEW))
    if flat:
        # the whole preset line, so the assert below is on the RESULT and not on a substring
        edits.append(("outer routing flattened (the `parent` preset)", PRESET_OLD, PRESET_NEW))
    for name, old, new in edits:
        hits = [i for i, t in enumerate(texts) if t is not None and re.search(old, t, re.S)]
        if len(hits) != 1:
            raise SystemExit("!! edit %r found in %d cells (need exactly 1) -- refusing to write"
                             % (name, len(hits)))
        texts[hits[0]], n = re.subn(old, new, texts[hits[0]], flags=re.S)
        if n != 1:
            raise SystemExit("!! edit %r matched %d times inside cell %d" % (name, n, hits[0]))
        print("   edit ok : %-38s (cell %d)" % (name, hits[0]))
    return texts


def check(nb, meta, fracture=False, flat=False):
    """Post-conditions on the built artifact.  Every one of these is a thing that would be SILENT."""
    text = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
    assert '"cn0-ours": 0.20' in text, "view weight missing"
    assert text.count('"name": "cn0-ours"') == 1, "cn0-ours arm entry missing or duplicated"
    assert "run_single(4, torch.device('cuda:0'))" in text, "index 4 not run on the lighter GPU"
    # ⚠️ Why the dataset must be in BOTH places.  `dataset_sources` makes Kaggle MOUNT the files;
    # `ASSET_ROOTS` is what `_asset_find_asset` actually walks.  v1 of this arm had only the first,
    # and died fail-closed in 2.5 minutes with "expected one pinned asset raptor_ft_cn0.pt, got []"
    # after passing the DINOv2 fingerprint gate.  Mounting is necessary but not sufficient.
    assert "/kaggle/input/rsna-knee-cn0-weights" in text, "our dataset is not in ASSET_ROOTS"
    assert "norm=arm.get('norm', _KE_NS['NORM']))" in text, "per-arm norm not honoured"
    assert "_arm.get('k_eval_fixed')" in text, "k_eval_fixed not honoured"
    assert '"k_eval_fixed": 24' in text, "our view does not pin k_eval"
    assert '"norm": "none"' in text, "our view does not pin norm"
    # the recipe's own invariants must still hold
    want_preset = PRESET_PARENT_TXT if flat else PRESET_SPEEDY_TXT
    assert want_preset in text, "preset line is not %r" % want_preset
    if flat:
        # the dispatch must still CONTAIN the parent branch -- baking the default without the
        # branch would silently fall through to `speedy` and produce a fitted-routing arm
        assert 'PRESET == "parent"' in text, "parent preset branch is missing"
    if fracture:
        assert '_RAD_EXCLUDE = ("Baker\'s",)' in text, "the fracture fix did not take"
    else:
        assert '_RAD_EXCLUDE = ("Baker\'s", \'Fracture\')' in text, "unexpected fracture fix"
    assert '"rad_cal_w": 0.40,' in text, "rad_cal_w changed"
    # ⚠️ `coatnet_w`'s LITERAL stays 0.65 even under `parent` -- the preset flattens it at RUNTIME
    # through the dispatch block.  So this asserts the literal, and the branch assert above is what
    # proves the flattening will actually happen.
    assert '"coatnet_w": {"__default__": 0.65' in text, "outer routing literal changed"
    assert meta["enable_internet"] is False, "internet must be OFF (A78)"
    assert MEMBER_DATASET in (meta.get("dataset_sources") or []), "weights dataset not mounted"
    print("   post-conditions: all present")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="assert the built artifact, build nothing")
    ap.add_argument("--fracture", action="store_true",
                    help="also build `kernel-0943cf`: the member PLUS the _RAD_EXCLUDE fracture fix "
                         "(the two changes this repo actually has measurements for)")
    ap.add_argument("--flat", action="store_true",
                    help="also build `kernel-0943cflat`: the member + the fracture fix + the "
                         "FLATTENED outer routing, so the final pair can differ on ONE axis")
    a = ap.parse_args()

    if a.check:
        for d, frac, flat in ((DST, False, False),
                              (ROOT / "kernels" / "kernel-0943cf", True, False),
                              (ROOT / "kernels" / "kernel-0943cflat", True, True)):
            if d.is_dir():
                check(json.loads((d / NB).read_text(encoding="utf-8")),
                      json.loads((d / "kernel-metadata.json").read_text(encoding="utf-8")),
                      fracture=frac, flat=flat)
                print("CHECK OK: %s" % d)
        return 0

    targets = [(DST, "ppxl16/rsna-knee-0943c", "RSNA knee 0943c", False, False)]
    if a.fracture:
        targets.append((ROOT / "kernels" / "kernel-0943cf", "ppxl16/rsna-knee-0943cf",
                        "RSNA knee 0943cf", True, False))
    if a.flat:
        targets.append((ROOT / "kernels" / "kernel-0943cflat", "ppxl16/rsna-knee-0943cflat",
                        "RSNA knee 0943cflat", True, True))

    for dst, kid, title, frac, flat in targets:
        nb = json.loads((SRC / NB).read_text(encoding="utf-8", errors="replace"))
        texts = ["".join(c["source"]) if "source" in c else None for c in nb["cells"]]
        texts = apply_edits(texts, extra_fracture=frac, flat=flat)
        for c, t in zip(nb["cells"], texts):
            if t is not None:
                c["source"] = t.splitlines(keepends=True)

        meta = json.loads((SRC / "kernel-metadata.json").read_text(encoding="utf-8"))
        meta["id"], meta["title"] = kid, title
        meta["is_private"] = True
        meta["enable_internet"] = False
        ds = list(meta.get("dataset_sources") or [])
        if MEMBER_DATASET not in ds:
            ds.append(MEMBER_DATASET)
        meta["dataset_sources"] = ds
        meta.pop("id_no", None)

        check(nb, meta, fracture=frac, flat=flat)
        dst.mkdir(parents=True, exist_ok=True)
        (dst / NB).write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")
        (dst / "kernel-metadata.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("\nbuilt %s  (fracture=%s flat=%s, %d datasets)" % (dst, frac, flat, len(ds)))

    print("   reminder: the member view is 24 windows on purpose; the other four stay at 94.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
