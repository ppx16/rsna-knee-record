"""Publish the msft2 fine-tuned checkpoints as a dataset WE own.

WHY THIS IS THE FIRST ACTION, NOT A LATER ONE
    The exp085 line died on exactly this: its weights were a public dataset, the author made it
    private mid-session, and `kernels_push` then **silently dropped the source while reporting
    success**. Anything a submission must mount has to live in a dataset under our own account
    *before* anything depends on it.

    `work/msft2_out/` already holds the pulled kernel output, so no re-download is needed.

Usage:  python work/publish_msft2_dataset.py
"""
from __future__ import annotations

import json
import pathlib
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "work" / "msft2_out"
STAGE = ROOT / "work" / "msft2_dataset"
SLUG = "rsna-knee-msft2-ckpt"

KEEP = ["raptor_ft_msft0.pt", "raptor_ft_msft0_swa.pt", "raptor_ft_msft0.json",
        "raptor_gold_msft0.npz", "raptor_gold_msft0_swa.npz", "raptor_gold_msft0_epens.npz"]


def main() -> int:
    import shutil
    STAGE.mkdir(parents=True, exist_ok=True)
    missing = []
    for f in KEEP:
        s = SRC / f
        if not s.is_file():
            missing.append(f)
            continue
        shutil.copyfile(s, STAGE / f)
        print("staged %-28s %12d" % (f, s.stat().st_size))
    if missing:
        print("!! missing from %s: %s" % (SRC, missing))
        print("   (files present: %s)" % sorted(p.name for p in SRC.iterdir()))
        return 2

    meta = {
        "title": "RSNA Knee msft2 fine-tuned checkpoints",
        "id": "ppxl16/%s" % SLUG,
        "licenses": [{"name": "CC0-1.0"}],
        "description": (
            "Fine-tuned checkpoints from ppxl16/rsna-knee-msft2 v4 (TPU, 4/4 epochs, "
            "partial=false). Warm start: raptor_ft_coatnet_v5_full_swa.pt with its 8 head tensors "
            "CARRIED, verified at zero training (zero-shot gold58 macro-AUC 0.8961 against our own "
            "measured reload bar of 0.8963). Gold58 macro-AUC by epoch: ep0 0.8903, ep1 0.8946, "
            "ep2 0.9007, ep3 0.9048; SWA over epochs [3,2,1] = 0.9069, which clears the A167 member "
            "gate (solo >= 0.90). Published here because a submission must mount assets from a "
            "dataset we control -- the exp085 line died when its weights dataset was withdrawn."
        ),
    }
    (STAGE / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote dataset-metadata.json -> %s" % meta["id"])

    if "--push" not in sys.argv:
        print("\nstaged only. Re-run with --push to create the dataset.")
        return 0

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    try:
        r = api.dataset_create_new(str(STAGE), public=False, quiet=False, dir_mode="skip")
        print("created:", getattr(r, "ref", None) or r)
    except Exception as e:
        print("create failed: %s: %s" % (type(e).__name__, str(e)[:400]))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
