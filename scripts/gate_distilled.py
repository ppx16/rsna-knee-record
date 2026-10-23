"""Publish the distilled CoAtNet as OUR dataset, and build a ZERO-SHOT GATE kernel for it.

WHY. The community's score ladder is `0.924 -> 0.941 -> 0.942 -> 0.943`, and **every +0.001 came from
adding a CoAt family member**. The "4th" member (Repair-v1) turned out to have no published checkpoint at
all, so the ladder has no public next rung -- except one checkpoint we rejected too fast:

`kawaharataishi/rsna-knee-coatnet-distilled-checkpoint` is the SAME architecture family as ours
(`coatnet_rmlp_2_rw_384`, res 384, head `clsW (12,1024)`, tail `backbone.norm`) and its metadata claims
**`auc 0.9231`** -- higher than our msft2's gold-58 0.9069. It was rejected last round for having **no
documented geometry**, which is the `hunter_pool` trap.

⚠️ BUT THAT REJECTION IS TESTABLE RATHER THAN FINAL. The author's trainer has a `--zero_shot_only` gate:
it loads a warm start, runs ZERO gradient steps, and reports the macro-AUC on the 58 gold studies. If the
distilled checkpoint scores near its claimed 0.9231 **under our geometry**, then its geometry IS ours and
the checkpoint is real. If it scores far below, the geometry differs and it is rejected for good -- with
evidence instead of a guess.

This is a GATE, not an arm: no submission is spent, and a failure costs one queued run.

⚠️ Hosting first, on our own account. The exp085 line died because its weights were a public dataset that
its owner made private mid-session, and `kernels_push` **silently dropped the source while reporting
success**. Anything a kernel mounts must live under our control.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import build_msft2_kernel as m2  # noqa: E402  (the four validated msft2 patches)

CKPT = ROOT / "work" / "coat_probe" / "rsna-knee-coatnet-distilled-checkpoint" / \
    "coatnet_384_full_cohort_swa.pt"
STAGE = ROOT / "work" / "distilled_dataset"
DS_SLUG = "rsna-knee-coat-distilled-ckpt"
KID = "ppxl16/rsna-knee-msft2-gate-distilled"
DST = ROOT / "kernels" / "kernel-msft2-gate-distilled"

GLOB_OLD = "_warm_c = glob.glob('/kaggle/input/**/raptor_ft_coatnet_v5_full_swa.pt', recursive=True)\n" \
           "if not _warm_c:\n" \
           "    raise SystemExit('warm-start checkpoint raptor_ft_coatnet_v5_full_swa.pt not found " \
           "under /kaggle/input (dataset mounted?)')\n"
GLOB_NEW = "_warm_c = glob.glob('/kaggle/input/**/coatnet_384_full_cohort_swa.pt', recursive=True)\n" \
           "if not _warm_c:\n" \
           "    raise SystemExit('warm-start checkpoint coatnet_384_full_cohort_swa.pt not found " \
           "under /kaggle/input (dataset mounted?)')\n"


def publish(push: bool) -> int:
    if not CKPT.is_file():
        print("!! missing %s -- run work/probe_coat_ckpts.py first" % CKPT)
        return 2
    STAGE.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CKPT, STAGE / CKPT.name)
    print("staged %s (%.1f MB)" % (CKPT.name, CKPT.stat().st_size / 1e6))
    meta = {
        "title": "RSNA Knee CoAtNet distilled checkpoint",
        "id": "ppxl16/%s" % DS_SLUG,
        "licenses": [{"name": "CC0-1.0"}],
        "description": (
            "Mirror of kawaharataishi/rsna-knee-coatnet-distilled-checkpoint, re-hosted under our own "
            "account so a kernel can mount it without depending on a third party's dataset remaining "
            "public (the exp085 line died when exactly that happened). Same bytes, unmodified. "
            "Its own metadata reports auc 0.9231; architecture coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k at "
            "res 384, head clsW(12,1024)/clsb/norm/att, tail backbone.norm -- the same family our ARMS "
            "loader handles. Its preprocessing geometry is NOT documented, which is why it is being "
            "zero-shot gated against the 58 gold studies before any use."
        ),
    }
    (STAGE / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if not push:
        print("staged only (pass --push)")
        return 0
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    try:
        r = api.dataset_create_new(str(STAGE), public=False, quiet=False, dir_mode="skip")
        print("dataset created:", getattr(r, "ref", None) or r)
        return 0
    except Exception as e:  # noqa: BLE001
        print("create failed: %s: %s" % (type(e).__name__, str(e)[:400]))
        return 1


def build(push: bool) -> int:
    src = m2.SRC / m2.CODE
    text = m2.patch(src.read_text(encoding="utf-8"), "gate")
    n = text.count(GLOB_OLD)
    if n != 1:
        raise SystemExit("ANCHOR FAILURE [warm-start glob]: expected 1 match, found %d" % n)
    text = text.replace(GLOB_OLD, GLOB_NEW, 1)
    # ⚠️ FIRST VERSION FLAGGED ANY OCCURRENCE, INCLUDING A COMMENT. The trainer's line 397 is
    # `# ⚠️ The published raptor checkpoints (e.g. raptor_ft_coatnet_v5_full_swa.pt) wrap the tensors`
    # -- prose, not a reference. A guard that fires on prose trains you to weaken the guard, so it is
    # narrowed to FUNCTIONAL lines only: strip the comment part and look there.
    live = [ln for ln in text.splitlines()
            if "raptor_ft_coatnet_v5_full_swa.pt" in ln.split("#", 1)[0]]
    if live:
        raise SystemExit("!! the v5 filename survives on a live line -- the gate would load the wrong "
                         "model: %s" % live[:3])

    probes = [("coatnet_384_full_cohort_swa.pt", "the distilled checkpoint glob"),
              ("--zero_shot_only", "the gate flag"),
              ("[zeroshot] VERDICT", "the verdict print"),
              ("_HEAD_STATE", "msft2's head-carrying fix"),
              ("shape_mismatch", "the head shape check")]
    missing = [w for p, w in probes if p not in text]
    if missing:
        raise SystemExit("!! built payload missing: %s" % missing)

    if push:
        DST.mkdir(parents=True, exist_ok=True)
        (DST / m2.CODE).write_text(text, encoding="utf-8")
    tmp = ROOT / "work" / "_gate_distilled_preview.py"
    tmp.write_text(text, encoding="utf-8")
    import py_compile
    py_compile.compile(str(tmp), doraise=True)
    print("PATTERN CHECK OK: compiled, all %d probes present" % len(probes))

    if not push:
        print("(--check only)")
        return 0

    meta = json.loads((m2.SRC / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta["id"] = KID
    meta["title"] = "RSNA knee msft2 gate distilled"
    meta["code_file"] = m2.CODE
    meta["enable_tpu"] = True
    meta["enable_gpu"] = False
    meta.pop("machine_shape", None)
    meta["is_private"] = True
    ds = list(meta.get("dataset_sources") or [])
    # ⚠️ MUST BE `owner/slug`. The first version appended the bare slug and `kernels_push` died with
    # "Dataset must be specified in the form of '{username}/{dataset-slug}'" -- before pushing anything,
    # so nothing was wasted, but it is the same class as the other silent-format traps in this repo.
    ds_ref = "ppxl16/%s" % DS_SLUG
    if ds_ref not in ds:
        ds.append(ds_ref)
    meta["dataset_sources"] = ds
    (DST / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("built %s (%d datasets, tpu=%s)" % (DST.relative_to(ROOT), len(ds), meta["enable_tpu"]))

    if "--push-kernel" in sys.argv:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.kernels_push(str(DST))
        print("PUSHED %s" % KID)
    else:
        print("kernel NOT pushed (pass --push-kernel)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="store_true", help="stage (and with --push, create) the dataset")
    ap.add_argument("--kernel", action="store_true", help="build (and with --push-kernel, push) the gate")
    ap.add_argument("--push", action="store_true", help="create the dataset on Kaggle")
    ap.add_argument("--push-kernel", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    rc = 0
    if a.dataset or a.push:
        rc |= publish(a.push)
    if a.kernel or a.check or a.push_kernel:
        rc |= build(a.push_kernel)
    if not (a.dataset or a.push or a.kernel or a.check or a.push_kernel):
        ap.print_help()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
