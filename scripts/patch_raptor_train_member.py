# -*- coding: utf-8 -*-
"""
Patch the Raptor trainer (SHIPPED source, both kernel dirs) for the two-arm push:

  方向一 (微调臂): make it possible to START from a released >=0.90 checkpoint.

Three defects are fixed here, all of which fail SILENTLY rather than loudly:

  (1) load_raptor() unwrapped only the "student" container.  The released MaxSpan v5
      checkpoint nests its tensors under "model", so the dict comprehension matched
      NOTHING, load_state_dict({}, strict=False) accepted the empty dict, and the run
      proceeded from timm ImageNet weights while printing a success line.
      -> try every known container AND raise on zero tensors.
      (A168 recorded this as "fixed"; it is NOT fixed in either shipped file.  This is
       the same failure mode as before: the fix lived in a box copy, not the artefact.)

  (2) There was no way to initialise from a FULL model state (backbone + head).  The
      released checkpoint's 8 head tensors are named exactly like our
      RaptorClassifier's, so it is our state_dict.
      -> add --init_ckpt, which loads it and REFUSES if under half the tensors matched.

  (3) There was no way to score a checkpoint without training first.
      -> add --eval_only, which runs the existing evaluate() and exits.  This is the
         cheap decisive measurement: does this member add, before any GPU is spent.

Also drops machine_shape from the GPU metadata (A168 trap 3: an inherited
machine_shape makes enable_gpu:true still count as a TPU session).

Verified with compile(), not just ast.parse: a previous patcher inserted above
`from __future__ import annotations` and ast.parse did NOT catch it.
"""
import pathlib
import re
import hashlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
TARGETS = [
    ROOT / "kernels" / "kernel-raptor-train" / "knee-mri-training-the-twelve-finding-model.py",
    ROOT / "kernels" / "kernel-raptor-train-rule" / "knee-mri-training-the-twelve-finding-model.py",
]

# --------------------------------------------------------------------------- 1
OLD_LOADER = '''    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    st = ck["student"] if "student" in ck else ck
    bbst = {k[len("backbone."):]: v for k, v in st.items() if k.startswith("backbone.")}
    missing, unexpected = bb.load_state_dict(bbst, strict=False)
    ep = ck.get("epoch", "?")
    print(f"[raptor] loaded backbone from {os.path.basename(ckpt_path)} (ssl epoch {ep}) | "
          f"loaded {len(bbst)} tensors, missing {len(missing)}, unexpected {len(unexpected)}", flush=True)
    return f"{os.path.basename(ckpt_path)}@ep{ep}"'''

NEW_LOADER = '''    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Released checkpoints nest their tensors under a container key, and which key it is
    # varies ("model" for the MaxSpan/CoAtNet line, "student" for the SSL line,
    # "state_dict" elsewhere). Unwrapping only "student" failed SILENTLY: a foreign
    # container yielded an EMPTY dict, load_state_dict(strict=False) accepted it, and the
    # run continued from timm weights while printing a success line.
    st = ck
    for _container in ("model", "student", "state_dict", "net", "backbone", "ema"):
        if isinstance(st, dict) and isinstance(st.get(_container), dict):
            st = st[_container]
            break
    if not isinstance(st, dict):
        raise RuntimeError(f"load_raptor({os.path.basename(ckpt_path)}): no state dict found")
    bbst = {k[len("backbone."):]: v for k, v in st.items()
            if isinstance(k, str) and k.startswith("backbone.")}
    if not bbst:
        raise RuntimeError(
            f"load_raptor({os.path.basename(ckpt_path)}): extracted 0 backbone tensors -- "
            f"refusing to train from timm weights by accident. top-level keys="
            f"{list(ck)[:8]} nested keys={list(st)[:8]}")
    missing, unexpected = bb.load_state_dict(bbst, strict=False)
    ep = ck.get("epoch", "?")
    print(f"[raptor] loaded backbone from {os.path.basename(ckpt_path)} (ssl epoch {ep}) | "
          f"loaded {len(bbst)} tensors, missing {len(missing)}, unexpected {len(unexpected)}", flush=True)
    return f"{os.path.basename(ckpt_path)}@ep{ep}"'''

# --------------------------------------------------------------------------- 2
ANCHOR_ARGS = '''    ap.add_argument("--ckpt", default="timm",'''
NEW_ARGS = '''    ap.add_argument("--init_ckpt", default=None,
                    help="full model state_dict (backbone+head) to START from, e.g. the "
                         "released MaxSpan v5 CoAtNet (gold_auc 0.9214).")
    ap.add_argument("--eval_only", action="store_true",
                    help="score --init_ckpt on the gold split and exit; no training.")
    ap.add_argument("--ckpt", default="timm",'''

# --------------------------------------------------------------------------- 3
ANCHOR_MODEL = '''    model = RaptorClassifier(bb, F_dim=F_dim).to(dev)
'''
NEW_MODEL = '''    model = RaptorClassifier(bb, F_dim=F_dim).to(dev)
    if a.init_ckpt:
        # The released checkpoint's head tensors are named exactly like ours, so its
        # "model" dict IS our state_dict. Load it, and SAY how much landed -- an
        # under-matching load re-initialises silently and still looks like a success.
        _ick = torch.load(a.init_ckpt, map_location="cpu", weights_only=False)
        _ist = _ick
        for _c in ("model", "student", "state_dict"):
            if isinstance(_ist, dict) and isinstance(_ist.get(_c), dict):
                _ist = _ist[_c]
                break
        _own = model.state_dict()
        _use = {k: v for k, v in _ist.items()
                if k in _own and hasattr(v, "shape") and tuple(v.shape) == tuple(_own[k].shape)}
        _n_head = sum(1 for k in _use if not k.startswith("backbone."))
        if len(_use) < 0.5 * len(_own):
            raise RuntimeError(
                f"--init_ckpt matched only {len(_use)}/{len(_own)} tensors from "
                f"{os.path.basename(a.init_ckpt)} -- refusing (this would re-initialise "
                f"silently). ckpt keys={list(_ist)[:6]}")
        model.load_state_dict(_use, strict=False)
        print(f"[init] {os.path.basename(a.init_ckpt)}: loaded {len(_use)}/{len(_own)} tensors "
              f"({_n_head} head) | reported gold_auc={_ick.get('gold_auc')} "
              f"| src={_ick.get('src')} res={_ick.get('res')}", flush=True)

'''

# --------------------------------------------------------------------------- 4
# evaluate() is defined at ~line 536 and RETURNS a 4-tuple (au, aucs, P, Y) at ~548.
# The --eval_only block must sit AFTER that definition, so it anchors on the line that
# starts the training bookkeeping, not on the loss line above evaluate().
ANCHOR_EVAL_ONLY = '''    best = 0.0; best_state = None; best_P = None; t0 = time.time(); hist = []
'''

EVAL_ONLY_BLOCK = '''
    if a.eval_only:
        # The cheap decisive measurement: score the loaded weights on the gold split and
        # exit. Costs minutes, and it answers "does this member add" before any GPU is
        # spent on a fine-tune.
        if not a.init_ckpt:
            raise RuntimeError("--eval_only needs --init_ckpt")
        _au, _aucs, _P, _Y = evaluate()
        print(f"[eval-only] tag={a.tag} arch={a.arch} res={a.res} k_eval={a.k_eval} "
              f"n_gold={len(gold_ids)} gold macro AUC = {float(_au):.4f}", flush=True)
        for _n in LAB:
            if _n in _aucs:
                print(f"[eval-only]   {_n:<24} {_aucs[_n]:.4f}", flush=True)
        np.savez(os.path.join(HERE, f"raptor_gold_{a.tag}.npz"),
                 pred=_P, truth=_Y, ids=np.array(gold_ids))
        json.dump({"tag": a.tag, "eval_only": True, "init_ckpt": a.init_ckpt,
                   "gold_auc": float(_au), "aucs": _aucs, "arch": a.arch, "res": a.res,
                   "k_eval": a.k_eval, "src": src, "n_gold": len(gold_ids)},
                  open(os.path.join(HERE, f"raptor_evalonly_{a.tag}.json"), "w"), indent=1)
        return
'''


def patch(path: pathlib.Path) -> str:
    src = path.read_text(encoding="utf-8")
    orig = src
    report = []

    if "extracted 0 backbone tensors" in src:
        report.append("loader already patched")
    else:
        if OLD_LOADER not in src:
            raise SystemExit(f"FATAL {path.name}: loader anchor not found")
        src = src.replace(OLD_LOADER, NEW_LOADER, 1)
        report.append("loader: unwrap model/student/state_dict + raise on 0 tensors")

    if "--init_ckpt" in src:
        report.append("args already present")
    else:
        if ANCHOR_ARGS not in src:
            raise SystemExit(f"FATAL {path.name}: argparse anchor not found")
        src = src.replace(ANCHOR_ARGS, NEW_ARGS, 1)
        report.append("args: --init_ckpt / --eval_only")

    if "--init_ckpt matched only" in src:
        report.append("init block already present")
    else:
        if ANCHOR_MODEL not in src:
            raise SystemExit(f"FATAL {path.name}: model anchor not found")
        src = src.replace(ANCHOR_MODEL, NEW_MODEL, 1)
        report.append("init_ckpt: full state load + under-match refusal")

    if "[eval-only] tag=" in src:
        report.append("eval_only block already present")
    else:
        if ANCHOR_EVAL_ONLY not in src:
            raise SystemExit(f"FATAL {path.name}: training-loop anchor not found")
        src = src.replace(ANCHOR_EVAL_ONLY, ANCHOR_EVAL_ONLY + EVAL_ONLY_BLOCK, 1)
        report.append("eval_only: run evaluate() and exit")

    if src == orig:
        return f"{path.name}: unchanged"
    path.write_text(src, encoding="utf-8")
    return f"{path.name}: " + "; ".join(report)


def main():
    for p in TARGETS:
        if not p.exists():
            print(f"MISSING {p}")
            continue
        print(patch(p))
        # compile(), not ast.parse: a previous patcher inserted above a
        # `from __future__ import annotations` and ast.parse did NOT catch it.
        text = p.read_text(encoding="utf-8")
        compile(text, str(p), "exec")
        print(f"    compile OK  sha256={hashlib.sha256(text.encode()).hexdigest()[:16]}  "
              f"bytes={len(text.encode())}")

    # metadata: machine_shape turns a GPU run into a TPU session (A168 trap 3)
    for d in ("kernel-raptor-train", "kernel-raptor-train-rule"):
        mp = ROOT / "kernels" / d / "kernel-metadata.json"
        if not mp.exists():
            continue
        t = mp.read_text(encoding="utf-8")
        if '"machine_shape"' in t:
            t = re.sub(r'\s*"machine_shape":\s*"[^"]*",?', "", t)
            # keep the JSON valid if machine_shape was the last key
            t = re.sub(r',(\s*})', r'\1', t)
            mp.write_text(t, encoding="utf-8")
            import json
            json.loads(t)   # must still parse
            print(f"{d}/kernel-metadata.json: machine_shape REMOVED (JSON still valid)")
        else:
            print(f"{d}/kernel-metadata.json: no machine_shape")


if __name__ == "__main__":
    main()
