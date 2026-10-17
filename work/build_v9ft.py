# -*- coding: utf-8 -*-
"""
Build 方向一's one untried variant: warm-start the msft2 recipe from v9 finespacing.

Clean A/B by construction. kernels/kernel-msft2 already ran the SAME recipe from v5
maxspan and produced the only board-measured gain we have (solo 0.9069, +0.00025 as a
member by the 11x decay law). This file is kernel-msft2's trainer with EXACTLY TWO
changes, so the only variable is the starting checkpoint:

  1. the _warm_c glob:     v5_full_swa  ->  v9_full
  2. --tag:                msft0        ->  v9ft0

Everything else -- arch coatnet_rmlp_2_rw_384 @384, 4 epochs, bs 2, k 12, k_eval 24,
--grad_ckpt, seed 42, folds 5 fold 0, workers 0 -- is byte-identical.

Why v9 and not another try at v5: v5 has been done. v9 is the author's STRONGEST single
view (0.932 public LB solo vs maxspan v5's 0.927) and is the endpoint of the only
sampling axis we measured as a large lever (C47: 44 -> 64 -> 80 slices = 0.926 -> 0.928
-> 0.932; our own gap0 independently reproduced the opposite direction at -0.012).

The msft2 loader is ALREADY the correct one -- it unwraps model/student/state_dict,
raises on zero tensors, AND carries the 8 head tensors. That last part is not optional:
msft2's own comment records that dropping the head made the 2026-09-28 run start from an
untrained head and score 0.8795 at epoch 0 against a parent at 0.9214.
"""
import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "kernels" / "kernel-msft2" / "knee-mri-training-the-twelve-finding-model.py"
DST_DIR = ROOT / "kernels" / "kernel-v9ft"
DST = DST_DIR / "knee-mri-training-the-twelve-finding-model.py"

OLD_GLOB = ("_warm_c = glob.glob('/kaggle/input/**/raptor_ft_coatnet_v5_full_swa.pt', "
            "recursive=True)")
NEW_GLOB = ("_warm_c = glob.glob('/kaggle/input/**/raptor_ft_coatnet_v9_full.pt', "
            "recursive=True)")
OLD_ERR = ("    raise SystemExit('warm-start checkpoint raptor_ft_coatnet_v5_full_swa.pt "
           "not found under /kaggle/input (dataset mounted?)')")
NEW_ERR = ("    raise SystemExit('warm-start checkpoint raptor_ft_coatnet_v9_full.pt "
           "not found under /kaggle/input -- is dreaddevelopment/raptor-knee-finespacing "
           "attached?')")
OLD_TAG = "'--seed', '42', '--tag', 'msft0', '--folds', '5', '--fold', '0',"
NEW_TAG = "'--seed', '42', '--tag', 'v9ft0', '--folds', '5', '--fold', '0',"


def main():
    src = SRC.read_text(encoding="utf-8")
    for old, new, what in ((OLD_GLOB, NEW_GLOB, "glob"),
                           (OLD_ERR, NEW_ERR, "error message"),
                           (OLD_TAG, NEW_TAG, "tag")):
        n = src.count(old)
        if n != 1:
            raise SystemExit(f"FATAL: {what} anchor found {n} times, expected 1")
        src = src.replace(old, new, 1)

    # Prove the diff is exactly the two intended changes.
    a = SRC.read_text(encoding="utf-8").splitlines()
    b = src.splitlines()
    if len(a) != len(b):
        raise SystemExit("FATAL: line count changed -- more than the intended edit")
    diff = [(i + 1, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
    print(f"diff vs kernel-msft2: {len(diff)} line(s)")
    for i, x, y in diff:
        print(f"  line {i}\n    -  {x.strip()}\n    +  {y.strip()}")
    if len(diff) != 3:
        raise SystemExit(f"FATAL: expected 3 changed lines, got {len(diff)}")

    DST_DIR.mkdir(parents=True, exist_ok=True)
    DST.write_text(src, encoding="utf-8")
    compile(src, str(DST), "exec")

    meta = json.loads((ROOT / "kernels" / "kernel-msft2" / "kernel-metadata.json")
                      .read_text(encoding="utf-8"))
    meta["id"] = "ppxl16/rsna-knee-v9ft"
    meta["title"] = "RSNA knee v9ft"
    # msft2 ran on TPU, but D17 records that training coatnet_rmlp_2_rw_384 on torch_xla
    # dies in timm's additive relative position bias. Run this one on GPU, and carry NO
    # machine_shape: an inherited machine_shape makes enable_gpu:true still count as a
    # TPU session and the push then returns kernelId 0 (A168 trap 3).
    meta["enable_tpu"] = False
    meta["enable_gpu"] = True
    meta.pop("machine_shape", None)
    ds = list(meta["dataset_sources"])
    if "dreaddevelopment/raptor-knee-finespacing" not in ds:
        ds.append("dreaddevelopment/raptor-knee-finespacing")
    meta["dataset_sources"] = ds
    (DST_DIR / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")
    json.loads((DST_DIR / "kernel-metadata.json").read_text(encoding="utf-8"))

    print(f"\nbuilt {DST.relative_to(ROOT)}  bytes={len(src.encode())}  "
          f"sha256={hashlib.sha256(src.encode()).hexdigest()[:16]}")
    print(f"built {DST_DIR.relative_to(ROOT)}/kernel-metadata.json")
    print(f"  gpu={meta['enable_gpu']} tpu={meta['enable_tpu']} "
          f"machine_shape={meta.get('machine_shape')} datasets={len(ds)}")
    for d in ds:
        print(f"    - {d}")


if __name__ == "__main__":
    main()
