# -*- coding: utf-8 -*-
"""
Build a TPU self-training arm: convnext on the 64-slot corpus, with the ANATOMICAL MIRROR used
as a TRAINING AUGMENTATION (not as TTA).

WHY THIS ARCHITECTURE IS FORCED
    FACTS D17: training `coatnet_rmlp_2_rw_384` on torch_xla dies in timm's additive relative
    position bias (`pos_embed_rel.py:440 attn + self.get_bias()`), and disabling fused_attn
    only moves the same addition elsewhere. So on TPU the only usable family is convnext. That
    is a hard constraint, not a preference.
    ⚠️ HONEST PRIOR: FACTS A158's census found NO non-CoAtNet arm passing the member gate, so a
    convnext arm starts from a weak prior. Stated here rather than buried.

WHY THIS ONE IS GENUINELY UNTRIED
    The existing TPU convnext arm (`d64nx3360`, solo +0.0068 over `nx3360`, REAL) used NO
    mirror. The mirror is new here -- and unlike TTA (which measured -0.0129 on v5, C102), an
    augmentation is a different hypothesis: it regularises rather than averaging away variance.

    ⚠️ SECOND HONEST NOTE: the 64-slot corpus this trains on was built BEFORE the normalisation
    fix (FACTS C86: min-max made it 43% too dark vs the published 102.24 mean). So this arm is
    confounded exactly the way C86 says the density line was. Fixing that means rebuilding the
    corpus (8 shards), which is a separate, larger job.

THE PLANE TABLE MUST BE THE 64-SLOT ONE
    work/mirror_tta.py's _SLOT_PLANES is the 44-slot table. This kernel reads the 64-slot
    corpus, so the table here is [("Sagittal",1,18),("Sagittal",0,14),("Coronal",1,12),
    ("Coronal",0,8),("Axial",-1,12)] -> 64 entries. Getting this wrong picks the wrong mirror
    axis -- the very failure the caveat in SPEC-2026-10-08 warns about.
"""
import json
import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "kernels" / "kernel-d64pilot-tpu"
DST_DIR = ROOT / "kernels" / "kernel-tpu-cnxmirror"
CODE = "knee-mri-training-the-twelve-finding-model.py"

PLANES = '''# ---- anatomical mirror as a TRAINING AUGMENTATION ----------------------------------------
# 64-SLOT table (this kernel reads the 64-slot corpus). Read VERBATIM from
# kernels/kernel-diag-slots/diag_slots.py. NOTE sagittal appears TWICE, so its block is 18+14.
# Using the 44-slot table here would pick the WRONG mirror axis -- see the caveat in
# docs/SPEC-2026-10-08-anatomical-mirror-tta.md.
import os as _os
_SLOT_GROUPS_64 = [("Sagittal", 1, 18), ("Sagittal", 0, 14), ("Coronal", 1, 12),
                   ("Coronal", 0, 8), ("Axial", -1, 12)]
_SLOT_PLANES = [p.lower() for p, _f, k in _SLOT_GROUPS_64 for _ in range(k)]
assert len(_SLOT_PLANES) == 64, len(_SLOT_PLANES)


class StudyWindows(Dataset):'''

MIRROR = '''            tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0  # (3,H,W)
            # ---- anatomical mirror AUGMENTATION (train only) -----------------------------
            # Axis chosen PER WINDOW from this window's centre-slice plane; `c` indexes the
            # corpus slot axis directly. A batch-level width flip would be WRONG for sagittal
            # windows, whose L/R axis IS the 3-slice axis. Applied TRAIN-ONLY: this is an
            # augmentation, so eval must stay plain or the gold/AUC numbers stop being
            # comparable to every earlier arm.
            if self.train and _os.environ.get("RAPTOR_MIRROR"):
                if _SLOT_PLANES[c] == "sagittal":
                    tri = tri[::-1]
                else:
                    tri = tri[:, :, ::-1]
            H, W = tri.shape[-2], tri.shape[-1]'''


def main():
    if not SRC_DIR.exists():
        raise SystemExit(f"FATAL: base kernel missing: {SRC_DIR}")
    src = (SRC_DIR / CODE).read_text(encoding="utf-8")
    if "MIRROR_TRAIN" in src or "_SLOT_PLANES" in src:
        raise SystemExit("FATAL: base already patched -- refusing to double-apply")

    a1 = "class StudyWindows(Dataset):"
    a2 = ("            tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) "
          "/ 255.0  # (3,H,W)\n            H, W = tri.shape[-2], tri.shape[-1]")
    for old, new, tag in ((a1, PLANES, "slot planes"), (a2, MIRROR, "mirror augmentation")):
        n = src.count(old)
        if n != 1:
            raise SystemExit(f"FATAL/{tag}: anchor found {n} times, expected 1")
        src = src.replace(old, new, 1)
    src = src.replace("    # ---- anatomical mirror as a TRAINING AUGMENTATION",
                      "    # MIRROR_TRAIN ---- anatomical mirror as a TRAINING AUGMENTATION", 1)
    compile(src, "tpu_cnxmirror", "exec")

    DST_DIR.mkdir(parents=True, exist_ok=True)
    (DST_DIR / CODE).write_text(src, encoding="utf-8")

    meta = json.loads((SRC_DIR / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta["id"] = "ppxl16/rsna-knee-tpu-cnxmirror"
    meta["title"] = "RSNA knee tpu cnxmirror"
    meta["enable_tpu"] = True
    meta["enable_gpu"] = False
    meta["machine_shape"] = "Tpu1VmV38"      # we WANT a TPU session (A168 trap 3: this wins)
    (DST_DIR / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")
    json.loads((DST_DIR / "kernel-metadata.json").read_text(encoding="utf-8"))

    print(f"built {DST_DIR.relative_to(ROOT)}/{CODE}  bytes={len(src.encode())}")
    for k in ("_SLOT_PLANES", "assert len(_SLOT_PLANES) == 64", "self.train and _os.environ",
              "tri[::-1]", "tri[:, :, ::-1]", "MIRROR_TRAIN"):
        print(f"  {k:<34} {'OK' if k in src else 'MISSING'}")
    print(f"id={meta['id']} tpu={meta['enable_tpu']} gpu={meta['enable_gpu']} "
          f"shape={meta['machine_shape']} kernel_sources={len(meta['kernel_sources'])}")
    print("NOTE: augmentation is TRAIN-ONLY -- eval stays plain so the AUC stays comparable.")


if __name__ == "__main__":
    main()
