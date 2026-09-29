"""Build `kernels/kernel-msft2` — the corrected MaxSpan-v5 fine-tune, starting with its missing gate.

WHY (docs/RESULT-2026-09-29-msft-correction.md)
    `kernel-tpu-msft` fine-tuned the published `raptor_ft_coatnet_v5_full_swa.pt` (gold_auc **0.9214**)
    and was rejected because epoch 0 scored 0.8795, read as "the warm start was damaged immediately".
    That reading is wrong. The checkpoint holds **486 tensors = 478 `backbone.*` + 8 head tensors**
    (`clsW (12,1024)`, `clsb`, `norm.*`, `att.0.*`, `att.3.*`), our `RaptorClassifier` uses the **same
    names and nesting**, and `F_dim = bb.num_features` gives the right 1024 — but `load_raptor()` keeps
    only `backbone.` keys, so **the head was never loaded and was re-initialised from zeros/random**.
    Epoch 0 was therefore an UNTRAINED HEAD, which must start below 0.9214, and the trajectory was still
    rising (0.8795 -> 0.8938 -> 0.8968) when the 12 h cap killed it at 3 of 4 epochs.

WHAT THIS KERNEL ADDS
    1. `load_raptor` also CARRIES the 8 head tensors out of the checkpoint (the empty-backbone guard is
       kept — it exists because `strict=False` once silently accepted an empty dict).
    2. The head is loaded into the model with a **shape check and an absence check that RAISE**, rather
       than a `strict=False` that reports nothing.
    3. ⭐ **A ZERO-SHOT GATE**: evaluate the freshly loaded model BEFORE any gradient step and print a
       greppable verdict against the parent's 0.9214. `--zero_shot_only` stops there. This is the check
       whose absence made the whole comparison ambiguous, and it costs one forward pass over the gold set.
    4. The gate REFUSES to train when the warm start is not reproduced, unless `--allow_bad_warmstart`,
       because training on top of a bad warm start is exactly the 10.7 h that was wasted.
    5. TPU is requested properly (`enable_tpu` true / `enable_gpu` false). The first run was pushed as a
       GPU kernel while being called `tpu-msft`, so it ran on a T4 at 3.7 h/epoch.

⚠️ A NOTE ON THE SOFTWARE LIMIT, because it is deliberate: with the head loaded, the PESSIMISTIC reading
of a low zero-shot score is no longer available. If the gate fails, the cause is a genuine mismatch
(wrong arch, wrong F_dim, wrong wrapper key) and the printout names it (missing/unexpected/shape), so the
next step is a diagnosis rather than another training run.

Usage:
    python scripts/build_msft2_kernel.py --check
    python scripts/build_msft2_kernel.py --build [--push]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import py_compile
import shutil
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "kernels" / "kernel-tpu-msft"
DST = ROOT / "kernels" / "kernel-msft2"
CODE = "knee-mri-training-the-twelve-finding-model.py"
KID = "ppxl16/rsna-knee-msft2"
PARENT_BAR = 0.9214


def replace_once(text, old, new, label):
    n = text.count(old)
    if n != 1:
        raise SystemExit("ANCHOR FAILURE [%s]: expected exactly 1 match, found %d" % (label, n))
    return text.replace(old, new, 1)


def patch(text: str) -> str:
    # ── 1. carry the head out of the checkpoint ─────────────────────────────────────
    text = replace_once(
        text,
        '        print("[raptor] RANDOM-INIT control (no SSL weights)", flush=True)\n'
        '        return "random-init"\n',
        '        print("[raptor] RANDOM-INIT control (no SSL weights)", flush=True)\n'
        '        return "random-init", {}\n',
        "raptor-random-init-return",
    )
    text = replace_once(
        text,
        '    if ckpt_path in ("timm", "pretrained"):\n        return "timm-pretrained"\n',
        '    if ckpt_path in ("timm", "pretrained"):\n        return "timm-pretrained", {}\n',
        "raptor-timm-return",
    )
    text = replace_once(
        text,
        '          f"loaded {len(bbst)} tensors, missing {len(missing)}, unexpected {len(unexpected)}", flush=True)\n'
        '    return f"{os.path.basename(ckpt_path)}@ep{ep}"\n',
        '          f"loaded {len(bbst)} tensors, missing {len(missing)}, unexpected {len(unexpected)}", flush=True)\n'
        '    # ---- msft2 patch: the 8 HEAD tensors are part of the warm start too ------------------\n'
        '    # They are not optional. Dropping them is what made the 2026-09-28 run start from an\n'
        '    # untrained head and score 0.8795 at epoch 0 against a parent at 0.9214.\n'
        '    _head_state = {k: v for k, v in ck.items()\n'
        '                   if not k.startswith("backbone.") and torch.is_tensor(v)}\n'
        '    print(f"[raptor] head tensors carried: {len(_head_state)} "\n'
        '          f"{sorted(_head_state)}", flush=True)\n'
        '    if not _head_state:\n'
        '        print("[raptor] WARNING: checkpoint carries NO head tensors; the head will be random",\n'
        '              flush=True)\n'
        '    return f"{os.path.basename(ckpt_path)}@ep{ep}", _head_state\n',
        "raptor-carry-head",
    )
    text = replace_once(text, "    src = load_raptor(bb, a.ckpt)\n",
                        "    src, _HEAD_STATE = load_raptor(bb, a.ckpt)\n", "raptor-call-site")

    # ── 2. load the head with checks that RAISE ─────────────────────────────────────
    text = replace_once(
        text,
        "    model = RaptorClassifier(bb, F_dim=F_dim).to(dev)\n",
        "    model = RaptorClassifier(bb, F_dim=F_dim).to(dev)\n"
        "    # ---- msft2 patch: load the head, and CHECK it rather than trusting strict=False --------\n"
        "    if _HEAD_STATE:\n"
        "        _msd = model.state_dict()\n"
        "        _absent = sorted(k for k in _HEAD_STATE if k not in _msd)\n"
        "        _badshape = {k: (tuple(_HEAD_STATE[k].shape), tuple(_msd[k].shape))\n"
        "                     for k in _HEAD_STATE\n"
        "                     if k in _msd and tuple(_HEAD_STATE[k].shape) != tuple(_msd[k].shape)}\n"
        "        if _absent or _badshape:\n"
        "            raise SystemExit('[msft2] head is INCOMPATIBLE with this architecture: '\n"
        "                             'absent=%s shape_mismatch=%s (arch=%s F_dim=%s)'\n"
        "                             % (_absent, _badshape, a.arch, F_dim))\n"
        "        model.load_state_dict(_HEAD_STATE, strict=False)\n"
        "        print('[msft2] HEAD loaded: %d/%d tensors, names and shapes verified'\n"
        "              % (len(_HEAD_STATE), len(_HEAD_STATE)), flush=True)\n"
        "    else:\n"
        "        print('[msft2] WARNING: no head tensors available -> the head is RANDOM, which is '\n"
        "              'exactly the condition that invalidated the 2026-09-28 comparison', flush=True)\n",
        "load-head",
    )

    # ── 3. the zero-shot gate ───────────────────────────────────────────────────────
    # ⚠️ Build this block by CONCATENATION, never with `%` or `.format()` on it: the inserted code is
    # full of literal `%s`/`%d` conversions of its own, and running a format operator over the whole
    # string makes the builder try to interpret them (the first version died on exactly that).
    bar = "%.4f" % PARENT_BAR
    gate = (
        "    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)\n"
        "\n"
        "    # ---- msft2 patch: THE GATE THAT WAS MISSING ---------------------------------------\n"
        "    # Evaluate the freshly loaded model BEFORE any gradient step.  The 2026-09-28 run had no\n"
        "    # such check, so its epoch-0 score of 0.8795 was read as a damaged warm start when it was\n"
        "    # really an untrained head.  One forward pass settles it.\n"
        "    _zs, _zaucs, _zP, _zY = evaluate()\n"
        "    print('[zeroshot] zero training steps: GOLD macro-AUC %.4f over %d studies'\n"
        "          % (_zs, len(gold_ids)), flush=True)\n"
        "    print('[zeroshot] per-label %s'\n"
        "          % json.dumps({k: round(v, 4) for k, v in _zaucs.items()}), flush=True)\n"
        "    _ok = _zs >= " + bar + " - 0.002\n"
        "    print('[zeroshot] VERDICT: %s (parent bar " + bar + ", got %.4f)'\n"
        "          % ('PASS' if _ok else 'FAIL', _zs), flush=True)\n"
        "    if a.zero_shot_only:\n"
        "        print('[zeroshot] --zero_shot_only: stopping before any training', flush=True)\n"
        "        return\n"
        "    if not _ok and not a.allow_bad_warmstart:\n"
        "        raise SystemExit('[msft2] refusing to train on a warm start that scored %.4f vs bar "
        + bar + " ; pass --allow_bad_warmstart to override' % _zs)\n"
    )
    text = replace_once(text, "    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)\n", gate, "zero-shot-gate")

    # ── 4. the flags ────────────────────────────────────────────────────────────────
    text = replace_once(
        text,
        '    ap.add_argument("--smoke", action="store_true")\n',
        '    ap.add_argument("--smoke", action="store_true")\n'
        '    ap.add_argument("--zero_shot_only", action="store_true",\n'
        '                    help="load the warm start, evaluate it with ZERO training steps, print a "\n'
        '                         "verdict against the parent bar, and exit")\n'
        '    ap.add_argument("--allow_bad_warmstart", action="store_true",\n'
        '                    help="train even when the zero-shot gate fails (default: refuse)")\n',
        "flags",
    )
    # ── 5. run the gate, not the training ───────────────────────────────────────────
    # The script builds its own sys.argv in a prelude, so the flag has to go in there, not on a
    # command line Kaggle never gives us.
    text = replace_once(
        text,
        "            '--grad_ckpt', '--ckpt', _WARM,\n",
        "            '--grad_ckpt', '--ckpt', _WARM, '--zero_shot_only',\n",
        "prelude-argv",
    )
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--device", default="gpu", choices=["tpu", "gpu"],
                    help="'gpu' is the RIGHT choice for the zero-shot gate: it is one forward pass over "
                         "the gold set, so the 3.7 h/epoch cost that made GPU wrong for TRAINING does not "
                         "apply. It also avoids the account's TPU cap of 1, which a queued TPU kernel "
                         "holds -- pushing a TPU kernel while one is queued fails outright with "
                         "'Maximum batch TPU session count of 1 reached' and creates nothing.")
    a = ap.parse_args()

    src = SRC / CODE
    if not src.is_file():
        raise SystemExit("!! missing %s" % src)
    DST.mkdir(parents=True, exist_ok=True)
    out = DST / CODE
    text = patch(src.read_text(encoding="utf-8"))
    (out if a.build else SRC.parent / "_patched_preview.py").write_text(text, encoding="utf-8")

    py_compile.compile(str(out if a.build else SRC.parent / "_patched_preview.py"), doraise=True)
    for probe, why in (("_HEAD_STATE", "head carried"), ("[zeroshot] VERDICT", "zero-shot gate"),
                       ("--zero_shot_only", "the flag"), ("shape_mismatch", "the shape check"),
                       ("head tensors carried", "the head inventory print")):
        if probe not in text:
            raise SystemExit("!! patched payload is missing %s" % why)
    print("PATTERN CHECK OK: compiled, and all 5 probes present")

    if not a.build:
        print("(--check only: wrote _patched_preview.py next to the source)")
        return 0

    meta = json.loads((SRC / "kernel-metadata.json").read_text(encoding="utf-8"))
    # ⚠️ The first run was a GPU kernel named `tpu-msft`, so it ran on a T4 at 3.7 h/epoch and spent
    # 10.7 h of the 30 h weekly GPU quota. Force the flags to agree with the intent.
    meta["id"] = KID
    meta["title"] = "RSNA knee msft2"
    meta["code_file"] = CODE
    meta["enable_tpu"] = (a.device == "tpu")
    meta["enable_gpu"] = (a.device == "gpu")
    if "machine_shape" in meta:
        del meta["machine_shape"]          # an inherited value once overrode the flags
    meta["is_private"] = True
    (DST / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("built %s  (enable_tpu=%s enable_gpu=%s, %d datasets, competition=%s)"
          % (DST.relative_to(ROOT), meta["enable_tpu"], meta["enable_gpu"],
             len(meta.get("dataset_sources", [])), meta.get("competition_sources")))

    if a.push:
        from kaggle import api
        api.kernels_push(str(DST))
        print("PUSHED %s" % KID)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
