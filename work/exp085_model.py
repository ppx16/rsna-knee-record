"""Reconstruction of the exp085 model (t88take/rsna-knee-t88-weights) from its checkpoint alone.

THE PROBLEM THIS SOLVES
    `exp085` claims **LB 0.947 as a single model** with **gold58 macro-AUC 0.9062**, and its manifest
    carries a sha256 identity check against the submission that produced that number. Its weights are
    public (5 folds, 360 MB each) and **its code is not** -- `t88take/rsna-knee-code` returns 403 and
    none of the author's 27 public kernels is from this competition.

    So the model has to be reconstructed from the checkpoint. The checkpoint is unusually helpful: it
    is a `state_dict` PLUS a config dict, so the tensor names, shapes and every data-pipeline setting
    are read out rather than guessed (`work/inspect_ckpt.py`).

WHAT IS VERIFIABLE WITHOUT ANY DATA
    `load_state_dict(..., strict=True)` passing over all 245 tensors is a real proof that the module
    tree is right -- every parameter name and shape must match, including the odd ones
    (`slice_pool.depth_mlp.2`, `head.slot_prior`, the bare `Parameter`s with no `.weight` suffix).
    That is `--selftest`, and it needs only the .pt file.

⚠️ WHAT IS *NOT* PROVEN BY THAT, AND IS A HYPOTHESIS
    The ORDER of operations in the forward pass. The shapes constrain it -- `slice_pool` is a
    depth-conditioned attention pool over slices and `head.out` is `Linear(256,12)` -- but norm / add /
    softmax order and the role of `head.query` / `slot_emb` / `slot_prior` cannot be read off shapes.
    Three readings are implemented and switchable so they can be told apart by a NUMERIC test
    (`gold58 ~ 0.9062`), which is the only thing that can decide.

ARCHITECTURE, from the checkpoint's own config
    img 224 | backbone facebook/dinov2-base (patch 14 -> 16x16) | slice_feature_dim 1536
    n_slice_cached 12 | n_group 4 | slice_select band_groups | group_pick per_study
    slice_depth_range (0.2, 0.8) | eval_group_agg mean | crop_mm 130.0 | 90.1 M params

Usage:
    python work/exp085_model.py --ckpt work/newckpt/fold0.pt --selftest
    python work/exp085_model.py --ckpt work/newckpt/fold0.pt --forward orderA --random
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class Backbone(nn.Module):
    """Wrapper that reproduces the `backbone.net.*` prefix around a HF Dinov2Model."""

    def __init__(self, name: str, img: int):
        super().__init__()
        from transformers import Dinov2Config, Dinov2Model

        if "dinov2-base" not in name:
            raise SystemExit("!! only facebook/dinov2-base is implemented, got %r" % name)
        # ⚠️ `img` is the INPUT size (224), but the checkpoint's position_embeddings are
        # (1, **1370**, 768) = 1 + 37**2 -- so the model was built from HF's default Dinov2Config
        # with image_size **518** (518/14 = 37) and the 224 forward works by INTERPOLATION.
        # Building it at 224 gives (1, 257, 768) and the load fails with a size mismatch -- measured.
        cfg = Dinov2Config(
            image_size=518, patch_size=14, hidden_size=768, num_hidden_layers=12,
            num_attention_heads=12, intermediate_size=3072,
        )
        self.net = Dinov2Model(cfg)
        self.out_dim = 768
        self.input_size = img

    def forward(self, x):
        return self.net(x).last_hidden_state      # (B, 1+256, 768)


class SlicePool(nn.Module):
    """Depth-conditioned attention pool over the study's slices -> one vector per finding.

    `slice_attention_dim` 256, `slice_attention_queries` 1, `target_specific_slice_attention` True,
    and `slice_pool.query` is (12, 256) -- so there is ONE 256-d query per FINDING (12 labels), and
    the pool is target-specific, which is what the config flag says.
    """

    def __init__(self, feat=1536, attn=256, n_targets=12, n_slots=6):
        super().__init__()
        self.feat, self.attn, self.n_targets = feat, attn, n_targets
        self.norm = nn.LayerNorm(feat)
        self.depth_mlp = nn.Sequential(nn.Linear(1, attn), nn.SiLU(), nn.Linear(attn, feat))
        self.key = nn.Linear(feat, attn)
        self.value = nn.Linear(feat, feat)
        self.query = nn.Parameter(torch.zeros(n_targets, attn))

    def forward(self, x, depth):
        # x: (B,S,1536)  depth: (B,S,1) normalised depth in [0,1]
        x = self.norm(x) + self.depth_mlp(depth)
        k = self.key(x)                                    # (B,S,attn)
        v = self.value(x)                                  # (B,S,feat)
        q = self.query.unsqueeze(0)                        # (1,T,attn)
        a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.attn), dim=-1)   # (B,T,S)
        return a @ v, a                                    # (B,T,feat), attention


class Head(nn.Module):
    """`LayerNorm(1536) -> Linear(1536,256)` then a per-finding read-out.

    `head.out` is `Linear(256, 12)`: with 12 findings and 12 labels the likely reading is a
    **per-finding classifier**, `logit[:,f] = out.weight[f] . h[:,f] + out.bias[f]`, giving (B,12)
    directly. `head.query` (12,256), `head.slot_emb` (6,256) and `head.slot_prior` (12,6) then form a
    soft mixture over 6 slots before that read-out. Both are hypotheses; `--forward` selects.
    """

    def __init__(self, feat=1536, attn=256, n_targets=12, n_slots=6):
        super().__init__()
        self.feat, self.attn, self.n_targets, self.n_slots = feat, attn, n_targets, n_slots
        self.proj = nn.Sequential(nn.LayerNorm(feat), nn.Linear(feat, attn))
        self.out = nn.Linear(attn, n_targets)
        self.query = nn.Parameter(torch.zeros(n_targets, attn))
        self.slot_emb = nn.Parameter(torch.zeros(n_slots, attn))
        self.slot_prior = nn.Parameter(torch.zeros(n_targets, n_slots))

    def forward(self, pooled, order="A"):
        h = self.proj(pooled)                              # (B,T,attn)
        if order == "A":
            # per-finding classifier: logit_f = out.weight[f] . h_f + out.bias[f]
            w = self.out.weight                            # (T,attn)
            return (h * w.unsqueeze(0)).sum(-1) + self.out.bias
        if order == "B":
            # per-finding classifier after a soft mixture over slots
            logits_att = h @ self.slot_emb.t() / math.sqrt(self.attn) + self.slot_prior
            w = torch.softmax(logits_att, dim=-1)          # (B,T,slots)
            mixed = w @ self.slot_emb                      # (B,T,attn)
            return (mixed * self.out.weight.unsqueeze(0)).sum(-1) + self.out.bias
        if order == "C":
            # findings attend to slots by their own query, then a shared Linear(256,12)
            logits_att = (h * self.query.unsqueeze(0)).sum(-1, keepdim=True) \
                + h @ self.slot_emb.t() / math.sqrt(self.attn) + self.slot_prior
            w = torch.softmax(logits_att, dim=-1)
            mixed = w @ self.slot_emb
            return self.out(mixed).mean(dim=1) if mixed.dim() == 3 and mixed.shape[1] != 12 \
                else self.out(h).sum(dim=1)
        raise SystemExit("!! unknown forward order %r" % order)


class Exp085(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.mean = nn.Parameter(torch.zeros(1, 3, 1, 1), requires_grad=False)
        self.std = nn.Parameter(torch.ones(1, 3, 1, 1), requires_grad=False)
        self.backbone = Backbone(cfg.get("backbone_name", "facebook/dinov2-base"), int(cfg.get("img", 224)))
        feat = int(cfg.get("slice_feature_dim", 1536))
        attn = int(cfg.get("slice_attention_dim", 256))
        self.n_targets = 12
        self.n_slots = 6
        self.slice_pool = SlicePool(feat, attn, self.n_targets, self.n_slots)
        self.head = Head(feat, attn, self.n_targets, self.n_slots)

    def forward(self, imgs, depth, order="A"):
        """imgs (B,S,3,H,W) already normalised, depth (B,S,1) in [0,1]."""
        b, s = imgs.shape[:2]
        flat = imgs.flatten(0, 1)
        tok = self.backbone(flat)                       # (B*S, 1+256, 768)
        # slice_feature_dim is 1536 = 2 x 768: CLS concatenated with the mean patch token.
        feat = torch.cat([tok[:, 0], tok[:, 1:].mean(dim=1)], dim=-1).view(b, s, -1)
        pooled, attn = self.slice_pool(feat, depth)
        logits = self.head(pooled, order)
        return logits, attn


def load(path: str | Path, device="cpu"):
    obj = torch.load(path, map_location=device, weights_only=True)
    cfg = {k: v for k, v in obj.items() if k != "state_dict"}
    sd = obj["state_dict"]
    model = Exp085(cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return model, cfg, sd, list(missing), list(unexpected)


def selftest(path: str) -> int:
    model, cfg, sd, missing, unexpected = load(path)
    n_sd = len(sd)
    n_model = len(model.state_dict())
    print("checkpoint tensors      : %d" % n_sd)
    print("reconstructed parameters: %d" % n_model)
    print("MISSING (in model, not in ckpt) : %d  %s" % (len(missing), missing[:8]))
    print("UNEXPECTED (in ckpt, not in model): %d  %s" % (len(unexpected), unexpected[:8]))
    try:
        model.load_state_dict(sd, strict=True)
        print("strict=True load: PASS  <-- the module tree matches the checkpoint EXACTLY")
        ok = True
    except Exception as e:
        print("strict=True load: FAIL  %s" % str(e)[:400])
        ok = False
    bad = [k for k, v in sd.items() if k in model.state_dict()
           and tuple(model.state_dict()[k].shape) != tuple(v.shape)]
    print("shape mismatches on shared names: %d %s" % (len(bad), bad[:5]))
    return 0 if (ok and not bad) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--forward", default="A", choices=["A", "B", "C"])
    ap.add_argument("--random", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest(a.ckpt)
    if a.random:
        model, cfg, sd, missing, unexpected = load(a.ckpt)
        model.eval()
        b, s = 2, int(cfg.get("n_slice_cached", 12))
        img = int(cfg.get("img", 224))
        x = torch.randn(b, s, 3, img, img)
        x = (x - model.mean) / model.std
        d = torch.rand(b, s, 1) * 0.6 + 0.2
        with torch.no_grad():
            logits, attn = model(x, d, order=a.forward)
        print("forward order %s -> logits %s  range [%.3f, %.3f]  attn %s"
              % (a.forward, tuple(logits.shape), float(logits.min()), float(logits.max()), tuple(attn.shape)))
        print("attn rows sum to 1: %s" % bool(torch.allclose(attn.sum(-1), torch.ones_like(attn.sum(-1)), atol=1e-4)))
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
