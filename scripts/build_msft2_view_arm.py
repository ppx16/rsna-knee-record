"""Build `kernel-0943cfmsft2`: `0943cf` plus OUR `msft2` CoAtNet member as a SIXTH Raptor view.

WHY THIS EXISTS (and the honest case against it)
    `msft2` (SWA, gold58 **0.9069**) is the only artifact this repo owns that clears the A167 member
    gate (solo >= ~0.90), and it has **never been in the ensemble**. Every recorded reason for
    closing it is either inapplicable or weak:
      * D5's rule is "**sub-0.90** solo members are net harmful" -- 0.9069 is ABOVE that line;
      * D7 closed "our raptor member" on different-architecture evidence, but msft2 is a CoAtNet,
        i.e. the SAME family as everything already in the stack.
    ⚠️ AND THAT IS ALSO THE CASE AGAINST IT, which must not be buried:
      * the trainer reads `all_vols.npy` -- the SAME 44-slot corpus as `cn0` -- and msft2 was
        initialised from `raptor_ft_coatnet_v5_full_swa.pt`, which is `maxspan-v5`, ALREADY a view
        here. So it is correlated with both.
      * HANDOFF 13.2 measured a **0.910** member at 10-15 % weight changing the LB by **0.000**, and
        msft2 at 0.9069 is exactly that band. 0.20/1.40 = 14.3 % share, i.e. we are testing the
        configuration that measurement already says does nothing.
    ==> Expected value is ~0. It is built because it is cheap, it is the one gate-passing artifact we
    own, and "never tested" is not the same as "closed". It is NOT expected to break 0.943.

GEOMETRY -- taken from what it TRAINED on, not from its warm start
    msft2's warm start (`maxspan-v5`) is 64-slot / span (0.02,0.98) / img 336. But the trainer
    renders from the 44-slot corpus at res 384, so the member's expected input is `_SLOTS44`,
    span (0.06,0.94), img 384 -- identical to the `cn0-ours` entry, which is why this entry is a
    copy of that one with only the file name changed. Getting this wrong is the A143 class of bug:
    a geometry mismatch degrades a member SILENTLY and passes every gate.

Usage:
    python scripts/build_msft2_view_arm.py           # build (asserts every edit)
    python scripts/build_msft2_view_arm.py --push    # build + push
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "kernels" / "kernel-0943cf"
DST = ROOT / "kernels" / "kernel-0943cfmsft2"
NB = "rsna-knee-restructured-version-3.ipynb"
SLUG = "rsna-knee-0943cfmsft2"
TITLE = "RSNA knee 0943cfmsft2"
CKPT_DATASET = "ppxl16/rsna-knee-msft2-ckpt"
MEMBER_FILE = "raptor_ft_msft0_swa.pt"
MEMBER_W = 0.20

VIEW_W_OLD = r'("cn0-ours": 0\.20)\}'
VIEW_W_NEW = ('\\1,\n'
              '                      "msft2-ours": %s}' % MEMBER_W)

ARM5_OLD = r'(\{"name": "cn0-ours".*?"reverse": False, "w": 0\.20\},\s*)\]'
ARM5_NEW = r'''\1
    # ADDED 2026-10-04 by scripts/build_msft2_view_arm.py.  `msft2` = our CoAtNet fine-tune whose SWA
    # reached gold58 0.9069 -- the only artifact this repo owns that clears the A167 member gate, and
    # never previously in the ensemble.  Geometry is a copy of `cn0-ours` above because the trainer
    # renders from the SAME 44-slot corpus at res 384; only the file name differs.
    # ⚠️ Expected value ~0: it is the same family as everything else here and it was initialised from
    # `maxspan-v5`, already view 0.  This tests "never tried", not "expected to help".
    {"name": "msft2-ours", "file": "%s",
     "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384,
     "img": 384, "slots": _SLOTS44, "span": (0.06, 0.94), "k_eval": 24,
     "k_eval_fixed": 24, "norm": "none",
     "reverse": False, "w": %s},
]''' % (MEMBER_FILE, MEMBER_W)

GPU0_OLD = r"(\n\s*run_single\(4, torch\.device\('cuda:0'\)\)\s*#[^\n]*\n)"
GPU0_NEW = (r"\1"
            r"            run_single(5, torch.device('cuda:0'))   "
            r"# our added msft2 view (index 5)\n")

ASSET_OLD = r"(ASSET_ROOTS = \[)(.*?)(\])"
ASSET_NEW = (r"\1\2, '/kaggle/input/datasets/ppxl16/rsna-knee-msft2-ckpt', "
             r"'/kaggle/input/rsna-knee-msft2-ckpt'\3")

EDITS = [
    ("ASSET_ROOTS whitelists the msft2 dataset", ASSET_OLD, ASSET_NEW),
    ("RUN['raptor_view_w'] gains msft2-ours", VIEW_W_OLD, VIEW_W_NEW),
    ("ARMS gains the msft2-ours entry", ARM5_OLD, ARM5_NEW),
    ("gpu_zero runs index 5", GPU0_OLD, GPU0_NEW),
]


def apply_edits(texts):
    for name, old, new in EDITS:
        hits = [i for i, t in enumerate(texts) if t is not None and re.search(old, t, re.S)]
        if len(hits) != 1:
            raise SystemExit("!! edit %r found in %d cells (need exactly 1) -- refusing to write"
                             % (name, len(hits)))
        texts[hits[0]], n = re.subn(old, new, texts[hits[0]], flags=re.S)
        if n != 1:
            raise SystemExit("!! edit %r matched %d times inside cell %d" % (name, n, hits[0]))
        print("   edit ok : %-42s (cell %d)" % (name, hits[0]))
    return texts


def check(nb, meta):
    """Post-conditions on the RESULT, not on the edits."""
    text = "\n".join("".join(c.get("source", [])) if isinstance(c.get("source"), list)
                     else (c.get("source") or "") for c in nb["cells"])
    assert '"msft2-ours": %s' % MEMBER_W in text, "view weight missing"
    assert text.count('"name": "msft2-ours"') == 1, "msft2 arm entry missing or duplicated"
    assert text.count('"name": "cn0-ours"') == 1, "cn0 arm entry lost"
    assert MEMBER_FILE in text, "member file name missing from ARMS"
    assert "run_single(5, torch.device('cuda:0'))" in text, "index 5 not run on the lighter GPU"
    assert "run_single(4, torch.device('cuda:0'))" in text, "index 4 (cn0) lost"
    assert "/kaggle/input/rsna-knee-msft2-ckpt" in text, "msft2 dataset not in ASSET_ROOTS"
    assert "/kaggle/input/rsna-knee-cn0-weights" in text, "cn0 dataset lost from ASSET_ROOTS"
    assert "norm=arm.get('norm', _KE_NS['NORM']))" in text, "per-arm norm not honoured"
    assert "_arm.get('k_eval_fixed')" in text, "k_eval_fixed not honoured"
    # the recipe's own invariants
    assert 'PRESET = os.environ.get("RSNA_PRESET", "speedy")' in text, "preset changed"
    assert '_RAD_EXCLUDE = ("Baker\'s",)' in text, "the fracture fix was lost"
    assert '"rad_cal_w": 0.40,' in text, "rad_cal_w changed"
    assert '"coatnet_w": {"__default__": 0.65' in text, "outer routing literal changed"
    assert meta["enable_internet"] is False, "internet must be OFF (A78)"
    ds = meta.get("dataset_sources") or []
    assert CKPT_DATASET in ds, "msft2 checkpoint dataset not mounted"
    assert "ppxl16/rsna-knee-cn0-weights" in ds, "cn0 dataset lost"
    assert len(ds) == len(set(ds)), "duplicate dataset source"
    print("   post-conditions: all present (%d datasets)" % len(ds))


def main() -> int:
    nb = json.loads((SRC / NB).read_text(encoding="utf-8", errors="replace"))
    texts = ["".join(c["source"]) if "source" in c else None for c in nb["cells"]]
    n_before = len(nb["cells"])
    texts = apply_edits(texts)
    for c, t in zip(nb["cells"], texts):
        if t is not None:
            c["source"] = t.splitlines(keepends=True)
    assert len(nb["cells"]) == n_before

    meta = json.loads((SRC / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta["id"], meta["title"] = "ppxl16/" + SLUG, TITLE
    meta["is_private"] = True
    meta["enable_gpu"] = True
    meta["enable_tpu"] = False
    meta["enable_internet"] = False
    ds = list(meta.get("dataset_sources") or [])
    if CKPT_DATASET not in ds:
        ds.append(CKPT_DATASET)
    meta["dataset_sources"] = ds
    meta.pop("id_no", None)

    check(nb, meta)
    DST.mkdir(parents=True, exist_ok=True)
    (DST / NB).write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")
    (DST / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\nbuilt %s" % DST)
    print("   view weights: 0.60 + 0.10 + 0.10 + 0.20 + 0.20 + %.2f = %.2f (normalised)"
          % (MEMBER_W, 1.20 + MEMBER_W))
    print("   msft2 share = %.4f  (13.2 measured 0.910-members at 0.10-0.15 as 0.000)"
          % (MEMBER_W / (1.20 + MEMBER_W)))

    if "--push" not in sys.argv:
        print("\nnot pushed (pass --push).")
        return 0
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    try:
        api.kernels_push(str(DST))
        print("pushed -> ppxl16/%s" % SLUG)
    except Exception as e:
        print("push failed: %s: %s" % (type(e).__name__, str(e)[:500]))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
