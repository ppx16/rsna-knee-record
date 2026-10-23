# -*- coding: utf-8 -*-
"""
kernel-msft2-long: the SAME msft2 recipe, trained LONGER.

User's direction: "不用开新实验...把旧方向的臂训练个更好的就很满足了" -- no new directions,
just train a better arm in the old one. So this changes NO mechanism. It changes one number.

WHY LONGER, FROM OUR OWN DATA (not a hunch)
    msft2 used --epochs 4. Its per-epoch snapshots on the 870-study instrument:
        msft0_ep1  0.8720
        msft0_ep2  0.8793
        msft0      0.8857   <- the selected best
    Monotone rising across every point we can see, with the best at the END of the run. The
    run was cut short, not converged. (work/largen_instrument.py prints this table.)

WHAT IS IDENTICAL (deliberately -- this is not a new experiment)
    arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k @384, bs 2, k 12, k_eval 24, --grad_ckpt,
    seed 42, folds 5 fold 0, the SAME warm start (maxspan v5, head carried), the same
    --ckpt / --grad_ckpt / workers. The guard's bar still applies and must still pass
    (v5 reads 0.8963 on our corpus; C92).

WHAT CHANGES
    --epochs 4 -> 12. Plus --topk stays at its train.py default so the SWA/epoch-ensemble
    artefacts are still produced, and the tag becomes msft0L so nothing collides.

PRE-REGISTERED (write the bar down BEFORE the run)
    1. The per-epoch gold_auc curve must still be RISING at the end; if it has flattened or
       rolled over, longer was not the problem and this line stops.
    2. The decisive number is the MEMBER delta on the 870 instrument (zero GPU):
           python scripts/member_add_test.py --arm <tag>@f0 --strong
       The bar to beat is msft2's +0.0084 (C91, CI [+0.0060,+0.0107]).
    3. Solo is NOT the criterion -- C90: a solo +0.0065 arm measured +0.0007 as a member.
    4. Overfit check: if gold rises while the TRAIN loss keeps falling and gold then rolls
       over, take the peak epoch, not the last one -- topk/SWA already retains it.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "kernels" / "kernel-msft2"
DST_DIR = ROOT / "kernels" / "kernel-msft2-long"
CODE = "knee-mri-training-the-twelve-finding-model.py"

OLD = "'--res', '384', '--epochs', '4', '--bs', '2', '--k', '12', '--k_eval', '24',"
NEW = ("'--res', '384', '--epochs', '12', '--bs', '2', '--k', '12', '--k_eval', '24',\n"
       "            # msft2 used 4 and was still rising at the end (ep1 0.8720 -> ep2 0.8793 ->\n"
       "            # best 0.8857 on the 870 instrument). 12 gives the curve room to peak.\n"
       "            # --topk (train.py default 3) keeps the per-epoch snapshots and the SWA\n"
       "            # average, so the peak epoch is retained even if the last one is worse.")
OLD_TAG = "'--seed', '42', '--tag', 'msft0', '--folds', '5', '--fold', '0',"
NEW_TAG = "'--seed', '42', '--tag', 'msft0L', '--folds', '5', '--fold', '0',"


def main():
    if not SRC_DIR.exists():
        raise SystemExit(f"FATAL: base missing: {SRC_DIR}")
    src = (SRC_DIR / CODE).read_text(encoding="utf-8")
    if "msft0L" in src:
        raise SystemExit("FATAL: already built")
    for old, new, tag in ((OLD, NEW, "epochs"), (OLD_TAG, NEW_TAG, "tag")):
        n = src.count(old)
        if n != 1:
            raise SystemExit(f"FATAL/{tag}: anchor found {n} times, expected 1")
        src = src.replace(old, new, 1)
    compile(src, "msft2_long", "exec")        # compile BEFORE writing

    DST_DIR.mkdir(parents=True, exist_ok=True)
    (DST_DIR / CODE).write_text(src, encoding="utf-8")

    meta = json.loads((SRC_DIR / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta["id"] = "ppxl16/rsna-knee-msft2-long"
    meta["title"] = "RSNA knee msft2 long"
    meta["enable_tpu"] = False
    meta["enable_gpu"] = True
    meta.pop("machine_shape", None)           # A168 trap 3: this would force a TPU session
    (DST_DIR / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")
    json.loads((DST_DIR / "kernel-metadata.json").read_text(encoding="utf-8"))

    print(f"built {DST_DIR.relative_to(ROOT)}/{CODE}  bytes={len(src.encode())}")
    for k in ("'--epochs', '12'", "'--tag', 'msft0L'", "msft2 used 4 and was still rising"):
        print(f"  {k:<34} {'OK' if k in src else 'MISSING'}")
    print(f"id={meta['id']} gpu={meta['enable_gpu']} tpu={meta['enable_tpu']} "
          f"shape={meta.get('machine_shape')}")
    print("bar to beat: msft2 member +0.0084 (member test is zero-GPU)")


if __name__ == "__main__":
    main()
