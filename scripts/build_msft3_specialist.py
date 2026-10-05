"""Build `kernels/kernel-msft3-spec` — a SELF-TRAINED hard-label specialist. NOT PUSHED BY DEFAULT.

WHY THIS EXISTS
    `docs/PREREG-2026-10-05-self-trained-specialist.md`. Short version: if the pending meniscus
    arm shows that label-specialists work at all, then the self-trained analogue is the next
    experiment the stop rule actually permits.

WHAT MAKES IT A DIFFERENT AXIS FROM msft2
    msft2 is a full-12-label fine-tune of the published v5 and is our only member that moved the
    rank. A third one of those is what the stop rule forbids ("同语料同配方的 CoAtNet 成员" -- the
    measured decay already quantifies it). This differs in THREE independent ways:
      1. the LOSS is masked to a label group (no gradient on the other labels),
      2. the EPOCH SELECTION is on that group's AUC, not the global macro,
      3. the BLEND TARGET is that group only.
    Two of those three are patches below; the third is an inference-side wiring choice.

WHAT IT PATCHES, ON TOP OF build_msft2_kernel.py's FOUR PATCHES
    msft2 already: carries the 8 head tensors out of the checkpoint, loads them with checks that
    RAISE, adds a zero-shot gate before any gradient step, and adds --zero_shot_only /
    --allow_bad_warmstart.  All four are REUSED VERBATIM -- the head-carrying fix is the only
    reason msft2's numbers mean anything and there is no reason to re-derive it.
    This builder adds:
      A. a MASKED loss (`_spec_mask`), replacing `lossf = nn.BCEWithLogitsLoss(pos_weight=pw)`
      B. GROUP-AWARE epoch selection, replacing `if au > best:` and the save guard
      C. `--spec_group` / `--spec_other_weight` flags
      D. `--freeze_backbone` + an optimizer/scheduler rebuild that tolerates an empty param group
         (the author's `AdamW([{bb...},{head...}])` RAISES if every backbone param is frozen, so
         a head-only run cannot be expressed with `--freeze_blocks`, which only freezes the first N)

⚠️ WHY (B) IS NOT OPTIONAL. `evaluate()` returns the macro AUC over ALL labels and the loop keeps
    the best epoch by that number. Adding a masked loss without changing the selection criterion
    gives an epoch chosen for the labels we are not specialising in -- i.e. it silently trains one
    objective and selects on another. That is the same class of error as the unloaded head.

Usage:
    python scripts/build_msft3_specialist.py --check           # anchors + compile, nothing written
    python scripts/build_msft3_specialist.py --build           # write kernels/kernel-msft3-spec
    python scripts/build_msft3_specialist.py --build --push    # only once the prereg's gate is met
"""
from __future__ import annotations

import argparse
import json
import pathlib
import py_compile
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "kernels" / "kernel-tpu-msft"
DST = ROOT / "kernels" / "kernel-msft3-spec"
CODE = "knee-mri-training-the-twelve-finding-model.py"
KID = "ppxl16/rsna-knee-msft3-spec"

# The six labels the base's OWN log names as the ones where the arms genuinely differ:
#   "Medial Meniscus / Medial OA / Lateral OA / Effusion / Synovitis / Contusion
#    是各臂真正不同、外层权重只在这里起作用的地方"
# This is the same sentence the meniscus prereg is built on; using it here keeps the two
# experiments on one hypothesis instead of two.
SPEC_LABELS = ["Medial Meniscus", "Medial OA", "Lateral OA",
               "Effusion", "Synovitis", "Contusion"]

# Reused verbatim from build_msft2_kernel.py -- same parent, same measured bar.
GATE_BAR = 0.8963


def replace_once(text, old, new, label):
    n = text.count(old)
    if n != 1:
        raise SystemExit("ANCHOR FAILURE [%s]: expected exactly 1 match, found %d" % (label, n))
    return text.replace(old, new, 1)


def msft2_patches(text: str) -> str:
    """The four patches build_msft2_kernel.py applies, copied so this builder is self-contained.

    ⚠️ If msft2's anchors ever change, `--check` fails loudly here rather than producing a kernel
    that silently lacks the head-carrying fix (which is the bug that wasted 10.7 h in September).
    """
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
    text = replace_once(
        text,
        "    model = RaptorClassifier(bb, F_dim=F_dim).to(dev)\n",
        "    model = RaptorClassifier(bb, F_dim=F_dim).to(dev)\n"
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
        "        print('[msft2] WARNING: no head tensors available -> the head is RANDOM', flush=True)\n",
        "load-head",
    )
    bar = "%.4f" % GATE_BAR
    gate = (
        "\n"
        "    _zs, _zaucs, _zP, _zY = evaluate()\n"
        "    print('[zeroshot] zero training steps: GOLD macro-AUC %.4f over %d studies'\n"
        "          % (_zs, len(gold_ids)), flush=True)\n"
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
    text = replace_once(
        text,
        "        return float(np.mean(list(aucs.values()))), aucs, P, Y\n",
        "        return float(np.mean(list(aucs.values()))), aucs, P, Y\n"
        "\n"
        "    # ---- the gate, inserted AFTER evaluate is defined ----\n"
        + gate,
        "zero-shot-gate",
    )
    text = replace_once(
        text,
        '    ap.add_argument("--smoke", action="store_true")\n',
        '    ap.add_argument("--smoke", action="store_true")\n'
        '    ap.add_argument("--zero_shot_only", action="store_true")\n'
        '    ap.add_argument("--allow_bad_warmstart", action="store_true")\n',
        "msft2-flags",
    )
    return text


def spec_patches(text: str) -> str:
    spec_lit = repr(list(SPEC_LABELS))

    # ── C. the flags ────────────────────────────────────────────────────────────────
    text = replace_once(
        text,
        '    ap.add_argument("--allow_bad_warmstart", action="store_true")\n',
        '    ap.add_argument("--allow_bad_warmstart", action="store_true")\n'
        '    ap.add_argument("--spec_group", default=None,\n'
        '                    help="comma-separated label names to specialise on; default is the six "\n'
        '                         "the base names as where the arms differ")\n'
        '    ap.add_argument("--spec_other_weight", type=float, default=0.0,\n'
        '                    help="loss weight for labels OUTSIDE the group (0.0 = pure specialist)")\n'
        '    ap.add_argument("--freeze_backbone", action="store_true",\n'
        '                    help="head-only run: freeze every backbone param. --freeze_blocks "\n'
        '                         "cannot express this (it only freezes the FIRST N stage groups)")\n',
        "spec-flags",
    )

    # ── A. the masked loss ──────────────────────────────────────────────────────────
    text = replace_once(
        text,
        "    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)\n",
        "    _SPEC_LABELS = " + spec_lit + "\n"
        "    if a.spec_group:\n"
        "        _SPEC_LABELS = [s.strip() for s in a.spec_group.split(',') if s.strip()]\n"
        "    _missing = [s for s in _SPEC_LABELS if s not in LAB]\n"
        "    if _missing:\n"
        "        raise SystemExit('[spec] label group entries not in LAB: %s (LAB=%s)'\n"
        "                         % (_missing, LAB))\n"
        "    _spec_idx = [LAB.index(s) for s in _SPEC_LABELS]\n"
        "    print('[spec] SPECIALIST GROUP %s -> indices %s of %d labels'\n"
        "          % (_SPEC_LABELS, _spec_idx, len(LAB)), flush=True)\n"
        "    _spec_mask = torch.full((len(LAB),), float(a.spec_other_weight), device=dev)\n"
        "    for _j in _spec_idx:\n"
        "        _spec_mask[_j] = 1.0\n"
        "    _spec_denom = float(_spec_mask.sum().item())\n"
        "    if _spec_denom <= 0:\n"
        "        raise SystemExit('[spec] every loss weight is zero -> no gradient would flow')\n"
        "    print('[spec] loss weights %s (denominator %.4f)'\n"
        "          % (np.round(_spec_mask.detach().cpu().numpy(), 4).tolist(), _spec_denom), flush=True)\n"
        "    _bce_none = nn.BCEWithLogitsLoss(pos_weight=pw, reduction='none')\n"
        "\n"
        "    def lossf(logits, y):\n"
        "        return (_bce_none(logits, y) * _spec_mask).sum() / _spec_denom\n"
        "    # ==== end specialist loss ====\n",
        "spec-loss",
    )

    # ── B. group-aware epoch selection ──────────────────────────────────────────────
    text = replace_once(
        text,
        "        if au > best:\n",
        "        _spec_auc = float(np.mean([aucs[k] for k in _SPEC_LABELS if k in aucs]))\n"
        "        _nspec = sum(1 for k in _SPEC_LABELS if k in aucs)\n"
        "        print('[spec] epoch %d: gold %.4f  SPEC %.4f over %d labels'\n"
        "              % (ep, au, _spec_auc, _nspec), flush=True)\n"
        "        if _nspec != len(_SPEC_LABELS):\n"
        "            print('[spec] WARNING: only %d/%d group labels had both classes in the gold set'\n"
        "                  % (_nspec, len(_SPEC_LABELS)), flush=True)\n"
        "        if _spec_auc > best:\n",
        "spec-selection",
    )
    text = replace_once(
        text,
        "        if au >= best and best_state is not None:\n",
        "        if _spec_auc >= best and best_state is not None:\n",
        "spec-save-guard",
    )
    # record the specialist score alongside the global one, so the artifact is self-describing
    text = replace_once(
        text,
        '                          "arch": a.arch, "lab": LAB, "epoch": ep}\n',
        '                          "arch": a.arch, "lab": LAB, "epoch": ep,\n'
        '                          "spec_group": _SPEC_LABELS, "spec_auc": _spec_auc}\n',
        "spec-record",
    )

    # ── D. optional whole-backbone freeze + an optimizer that tolerates an empty group ──
    old_opt = (
        "    head_params = [p for n_, p in model.named_parameters() if not n_.startswith(\"backbone.\") and p.requires_grad]\n"
        "    bb_params = [p for n_, p in model.named_parameters() if n_.startswith(\"backbone.\") and p.requires_grad]\n"
        "    opt = torch.optim.AdamW([{\"params\": bb_params, \"lr\": a.bb_lr},\n"
        "                             {\"params\": head_params, \"lr\": a.head_lr}], weight_decay=a.wd)\n"
        "    steps = max(len(tl) * a.epochs, 1)\n"
        "    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.bb_lr, a.head_lr],\n"
        "                                                total_steps=steps, pct_start=0.15)\n"
    )
    new_opt = (
        "    # ---- specialist: optional head-only run ----\n"
        "    if a.freeze_backbone:\n"
        "        _nf = 0\n"
        "        for _nm, _p in model.backbone.named_parameters():\n"
        "            _p.requires_grad = False\n"
        "            _nf += _p.numel()\n"
        "        print('[spec] BACKBONE FROZEN: %.2fM parameters, head-only run' % (_nf / 1e6), flush=True)\n"
        "    head_params = [p for n_, p in model.named_parameters() if not n_.startswith(\"backbone.\") and p.requires_grad]\n"
        "    bb_params = [p for n_, p in model.named_parameters() if n_.startswith(\"backbone.\") and p.requires_grad]\n"
        "    _groups = []\n"
        "    _maxlr = []\n"
        "    if bb_params:\n"
        "        _groups.append({\"params\": bb_params, \"lr\": a.bb_lr}); _maxlr.append(a.bb_lr)\n"
        "    if head_params:\n"
        "        _groups.append({\"params\": head_params, \"lr\": a.head_lr}); _maxlr.append(a.head_lr)\n"
        "    if not _groups:\n"
        "        raise SystemExit('[spec] no trainable parameters at all (freeze_backbone with a frozen head?)')\n"
        "    print('[spec] optimizer groups: %d (bb=%d head=%d)'\n"
        "          % (len(_groups), len(bb_params), len(head_params)), flush=True)\n"
        "    opt = torch.optim.AdamW(_groups, weight_decay=a.wd)\n"
        "    steps = max(len(tl) * a.epochs, 1)\n"
        "    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=_maxlr,\n"
        "                                                total_steps=steps, pct_start=0.15)\n"
    )
    text = replace_once(text, old_opt, new_opt, "spec-optimizer")

    # ── prelude: point at the warm start and set the tags ───────────────────────────
    # Warm start is the PUBLISHED v5, i.e. the same parent msft2 used, so msft2 and msft3-spec
    # are SIBLINGS with different objectives rather than a fine-tune of a fine-tune. The 11x
    # decay measured on this corpus says a child of msft2 would have almost nothing left to add.
    text = replace_once(
        text,
        "            '--grad_ckpt', '--ckpt', _WARM,\n",
        "            '--grad_ckpt', '--ckpt', _WARM,\n"
        "            '--tag', 'msft3spec',\n"
        "            '--spec_group', 'Medial Meniscus,Medial OA,Lateral OA,Effusion,Synovitis,Contusion',\n",
        "prelude-argv",
    )
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--head_only", action="store_true",
                    help="Tier 1: add --freeze_backbone and raise epochs for the head-only run")
    a = ap.parse_args()

    src = SRC / CODE
    if not src.is_file():
        raise SystemExit("!! missing %s" % src)

    text = msft2_patches(src.read_text(encoding="utf-8"))
    text = spec_patches(text)
    if a.head_only:
        text = replace_once(
            text,
            "            '--spec_group', 'Medial Meniscus,Medial OA,Lateral OA,Effusion,Synovitis,Contusion',\n",
            "            '--spec_group', 'Medial Meniscus,Medial OA,Lateral OA,Effusion,Synovitis,Contusion',\n"
            "            '--freeze_backbone', '--epochs', '8',\n",
            "prelude-head-only",
        )

    # ⚠️ NOT `SRC.parent / ...`: SRC.parent is `kernels/`, so the preview landed one level up and the
    # "next to the source" message (inherited from build_msft2_kernel.py) was simply false. A scratch
    # artifact also must not sit inside a directory that gets pushed, so it goes to work/.
    target = DST / CODE if a.build else ROOT / "work" / "_msft3_preview.py"
    if a.build:
        DST.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    py_compile.compile(str(target), doraise=True)

    probes = [
        ("SPECIALIST GROUP", "the group inventory print"),
        ("_spec_mask", "the masked loss"),
        ("_bce_none", "the per-label BCE"),
        ("_spec_auc > best", "group-aware epoch selection"),
        ("_spec_auc >= best", "group-aware save guard"),
        ("--spec_other_weight", "the other-weight flag"),
        ("--freeze_backbone", "the whole-backbone freeze flag"),
        ("optimizer groups", "the empty-group-tolerant optimizer"),
        ('"spec_auc": _spec_auc', "the recorded specialist score"),
        ("_HEAD_STATE", "msft2's head-carrying fix"),
        ("[zeroshot] VERDICT", "msft2's zero-shot gate"),
    ]
    missing = [why for probe, why in probes if probe not in text]
    if missing:
        raise SystemExit("!! patched payload is missing: %s" % missing)
    print("PATTERN CHECK OK: compiled, and all %d probes present" % len(probes))

    if not a.build:
        print("(--check only: wrote %s)" % target.name)
        return 0

    meta = json.loads((SRC / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta["id"] = KID
    meta["title"] = "RSNA knee msft3 spec"
    meta["code_file"] = CODE
    meta["enable_tpu"] = False
    meta["enable_gpu"] = True
    meta.pop("machine_shape", None)
    meta["is_private"] = True
    (DST / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("built %s  datasets=%d competition=%s"
          % (DST.relative_to(ROOT), len(meta.get("dataset_sources", [])),
             meta.get("competition_sources")))
    print("NOT PUSHED. The prereg gates this on the meniscus result; push is a separate decision.")
    if a.push:
        from kaggle import api
        api.kernels_push(str(DST))
        print("PUSHED %s" % KID)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
