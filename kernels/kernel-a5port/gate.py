"""The gate: does OUR port of the A5 model load THEIR weights strictly?

WHY THIS IS THE FIRST THING TO RUN.  Reproducing an architecture from someone else's inference code
is the class of work this project has already been burned by (A108: a config in a different schema
that no gate could see was wrong, so the only safe move was to stop).  Here the situation is
different in one decisive way: `m_f0.pt` is a TRAINED WEIGHT SET for exactly this architecture, so a
strict load is a layer-by-layer proof -- including `in_chans`, since that is baked into the
patch-embed weight shape.  A 16-channel input that our code fed as 3 channels would fail loudly here.

WHAT IT DOES
  1. reads `cfg` out of the checkpoint (the recipe, not a guess),
  2. builds the model the way the frontier's own loader does,
  3. `load_state_dict(..., strict=True)` and reports missing / unexpected keys,
  4. prints the receipt: tensor count, patch-embed weight shape (the in_chans proof),
  5. runs one synthetic forward pass so a wiring error surfaces now rather than after a training run.

  It trains nothing and spends no accelerator quota -- it is a CPU kernel on purpose.
"""
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch                                                            # noqa: E402
import torch.nn as nn                                                   # noqa: E402
import timm                                                             # noqa: E402

from a5_model import N_SLOT_TYPES, MASK_IDX, Net, ViTSlotToken          # noqa: E402

REPORT = {}


def find_ckpt(name="m_f0.pt"):
    for root, _dirs, files in os.walk("/kaggle/input"):
        if name in files:
            return os.path.join(root, name)
    raise FileNotFoundError("no %s under /kaggle/input" % name)


def build(cfg, nn_mod):
    """Verbatim from the frontier's loader (kernel-0943 cell 39, lines 365-368)."""
    stem = cfg["stem"]
    in_ch = 3 if stem == "compress" else cfg["n_slice"]
    kw = {"pretrained": False, "num_classes": 0, "in_chans": in_ch}
    if "vit_" in cfg["backbone"]:
        kw["img_size"] = cfg["img"]
    enc = timm.create_model(cfg["backbone"], **kw)
    if cfg["cond"] == "token":
        enc = ViTSlotToken(enc, N_SLOT_TYPES)
    return Net(enc, cfg["cond"], cfg.get("n_meta", 0), cfg["pool"],
               stem=stem, n_slice=cfg.get("n_slice", 16))


def main():
    path = find_ckpt()
    print("[gate] checkpoint:", path, flush=True)
    z = torch.load(path, map_location="cpu", weights_only=False)
    cfg = z["cfg"]
    REPORT["cfg"] = {k: v for k, v in cfg.items() if k != "labels"}
    print("[gate] cfg:", json.dumps(REPORT["cfg"], default=str), flush=True)
    print("[gate] fold:", z.get("fold"), " state_dict tensors:", len(z["state_dict"]), flush=True)

    model = build(cfg, nn)
    n_params = sum(p.numel() for p in model.parameters())
    pe = model.enc.vit.patch_embed.proj.weight if hasattr(model.enc, "vit") else None
    print("[gate] built: %.1fM params; patch_embed weight %s  <- in_chans is the 2nd dim"
          % (n_params / 1e6, tuple(pe.shape) if pe is not None else "n/a"), flush=True)
    REPORT["params"] = int(n_params)
    REPORT["patch_embed"] = list(pe.shape) if pe is not None else None

    result = {"strict_ok": False}
    try:
        model.load_state_dict(z["state_dict"], strict=True)
        result["strict_ok"] = True
        print("[gate] STRICT LOAD: OK -- our construction is layer-for-layer theirs", flush=True)
    except Exception as e:                                             # noqa: BLE001
        result["strict_error"] = "%s: %s" % (type(e).__name__, str(e)[:400])
        missing, unexpected = model.load_state_dict(z["state_dict"], strict=False)
        result["missing"] = list(missing)[:20]
        result["unexpected"] = list(unexpected)[:20]
        print("[gate] STRICT LOAD: FAILED -- %s" % result["strict_error"], flush=True)
        print("[gate]   missing   :", result["missing"], flush=True)
        print("[gate]   unexpected:", result["unexpected"], flush=True)
    REPORT["strict"] = result

    # one synthetic forward pass: the contract is im=(T,C,img,img), slot=(T,), sidx=(T,), B studies
    if result["strict_ok"]:
        try:
            model.eval()
            img, n_slot, B = int(cfg["img"]), N_SLOT_TYPES, 1
            im = torch.rand(n_slot, int(cfg["n_slice"]), img, img)
            slot = torch.arange(1, n_slot + 1)
            sidx = torch.zeros(n_slot, dtype=torch.long)
            smeta = torch.zeros(B, 0)
            with torch.no_grad():
                out = model(im, slot, smeta, sidx, B)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            print("[gate] forward OK: logits %s finite=%s"
                  % (tuple(logits.shape), bool(torch.isfinite(logits).all())), flush=True)
            REPORT["forward"] = {"shape": list(logits.shape),
                                 "finite": bool(torch.isfinite(logits).all())}
        except Exception as e:                                         # noqa: BLE001
            REPORT["forward_error"] = "%s: %s" % (type(e).__name__, str(e)[:400])
            print("[gate] forward FAILED: %s" % REPORT["forward_error"], flush=True)
            traceback.print_exc()

    with open("/kaggle/working/a5_gate.json", "w") as f:
        json.dump(REPORT, f, indent=2, default=str)
    print("[gate] wrote /kaggle/working/a5_gate.json", flush=True)


main()
