# -*- coding: utf-8 -*-
"""Build the TPU port-diagnostic notebook (one run, all landmines).

Every cell is wrapped in try/except and prints [PASS]/[FAIL], so a single failure does
not stop the notebook — the whole run reports a full matrix and one TPU queueing answers
everything. Covers, in port-layer order:

  A  device/autocast/bfloat16 (what replaces the cuda GradScaler + autocast)
  D  the REAL Model (DINOv2 backbone + SlotHead: einsum/masked_fill/topk/interpolate)
     forward+backward on a realistic [study,slot,group,img,img] bag, plus the
     fp32-vs-bf16 numerics that the FINGERPRINT_TOL=2e-3 check is built to catch
  C  host-numpy -> XLA transfer timing + grid_sample (the augment warp)
  B  multi-core xmp.spawn + torch.distributed('xla') + xm.optimizer_step

torch_xla 2.x API throughout (xr.world_size / xr.global_ordinal; no xrt_world_size).
"""

# --- GBK CONSOLE GUARD (see work/patch_gbk_guard.py) -------------------------------------------------
# This console is cp936, which cannot encode the U+26A0 warning sign used below. An unguarded
# `print("... \u26a0 ...")` raises UnicodeEncodeError at that line and kills the script MID-REPORT, while
# everything printed earlier is already on screen -- so the run looks complete. Self-contained on purpose:
# it imports `sys` itself and depends on nothing else in this tree.
import sys as _sys

try:
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001  -- non-reconfigurable stream: leave it alone, prints still degrade
    pass
# ----------------------------------------------------------------------------------------------------
import json
from pathlib import Path

from kpaths import kdir          # kernels/ layout, see kpaths.py

DST = kdir("kernel-tpu-smoke") / "rsna-knee-tpu-smoke.ipynb"

CELLS = [
    # --- cell 1: XLA basics -------------------------------------------------- #
    '''import os, time, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from kpaths import kdir          # kernels/ layout, see kpaths.py

try:
    print("torch", torch.__version__, "| torch_xla", torch_xla.__version__)
    print("world_size", xr.world_size())
    dev = xm.xla_device()
    print("device", dev, "| type", dev.type)
    a = torch.randn(16, 16, device=dev)
    b = (a @ a.t()).sum()
    xm.mark_step()
    print("[PASS] XLA basics, tiny matmul sum =", round(float(b.cpu()), 3))
except Exception as e:
    print("[FAIL] XLA basics:", repr(e)[:300])
''',
    # --- cell 2: the REAL Model (backbone + SlotHead) fwd+bwd ---------------- #
    '''from transformers import AutoModel

try:
    def find_dinov2(variant="large"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if "config.json" in files and "dinov2" in root.lower() and variant in root.lower():
                return root
        raise RuntimeError("dinov2-large model source not found")

    bb = AutoModel.from_pretrained(find_dinov2("large"))
    dim = bb.config.hidden_size
    print(f"DINOv2-large: {len(bb.encoder.layer)} layers, dim {dim}")

    class SlotHead(nn.Module):
        def __init__(self, dim, n_slot, n_out, hidden=256, p=0.2):
            super().__init__()
            self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
            self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
            self.query = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
            self.drop = nn.Dropout(p)
            self.out = nn.Linear(hidden, n_out)
            self.hidden = hidden
        def forward(self, x, mask):
            h = self.proj(x) + self.slot_emb
            att = torch.einsum("bsh,oh->bos", h, self.query) / self.hidden ** 0.5
            att = att.masked_fill(mask.unsqueeze(1) < 0.5, -1e4).softmax(-1)
            ctx = self.drop(torch.einsum("bos,bsh->boh", att, h))
            return (ctx * self.out.weight.unsqueeze(0)).sum(-1) + self.out.bias

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = bb
            self.head = SlotHead(dim * 2, 6, 12)
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        def forward(self, imgs, mask, img_size=None):
            B, S = imgs.shape[:2]
            x = imgs.reshape(B * S, *imgs.shape[2:]).float().div_(255.0)
            if img_size is not None and img_size != x.shape[-1]:
                x = F.interpolate(x, size=(img_size, img_size), mode="bilinear",
                                  align_corners=False)
            x = (x - self.mean) / self.std
            out = self.backbone(pixel_values=x).last_hidden_state
            patch = out[:, 1:]
            feat = torch.cat([out[:, 0], patch.mean(1)], dim=1).reshape(B, S, -1)
            return self.head(feat, mask)

    model = Model().to(dev)
    model.train()
    imgs = torch.randint(0, 256, (2, 6, 3, 224, 224), dtype=torch.uint8, device=dev)
    mask = torch.ones(2, 6, device=dev); mask[1, -1] = 0.0
    out = model(imgs, mask, 224)
    print("[PASS] full Model forward, logits", tuple(out.shape))
    out.mean().backward()
    xm.mark_step()
    print("[PASS] full Model backward")
    out2 = model(imgs, mask, 112)   # forces the F.interpolate branch
    xm.mark_step()
    print("[PASS] interpolate branch, logits", tuple(out2.shape))
except Exception as e:
    print("[FAIL] full Model fwd/bwd:", repr(e)[:400])
''',
    # --- cell 3: bfloat16 autocast + fp32-vs-bf16 numerics (fingerprint) ----- #
    '''try:
    with torch.no_grad():
        out_fp32 = model(imgs, mask, 224).float().cpu().numpy()
        try:
            with torch.autocast("xla", dtype=torch.bfloat16):
                out_bf16 = model(imgs, mask, 224).float().cpu().numpy()
            print("[PASS] autocast('xla', bf16) works")
            d = float(np.abs(out_fp32 - out_bf16).max())
            print(f"      fp32 vs bf16 max abs diff = {d:.4g}  (FINGERPRINT_TOL = 2e-3)")
            print("      ->" + (" fingerprint WILL break across device" if d > 2e-3
                               else " fingerprint within tol"))
        except Exception as e:
            print("[FAIL] autocast('xla', bf16):", repr(e)[:200])
            # fallback: run on bfloat16 model weights
            mbf = Model().to(torch.bfloat16).to(dev)
            out_bf16 = mbf(imgs.to(torch.bfloat16), mask, 224).float().cpu().numpy()
            d = float(np.abs(out_fp32 - out_bf16).max())
            print(f"      fallback bf16 weights diff = {d:.4g}")
except Exception as e:
    print("[FAIL] numerics:", repr(e)[:300])
''',
    # --- cell 4: data transfer + grid_sample timing (layer C) ---------------- #
    '''try:
    host = np.random.randint(0, 255, (8, 6, 3, 224, 224), dtype=np.uint8)  # realistic batch
    t0 = time.time()
    for _ in range(5):
        t = torch.from_numpy(np.ascontiguousarray(host)).to(dev)
        xm.mark_step()
    dt_sync = (time.time() - t0) / 5
    t0 = time.time()
    for _ in range(5):
        t = torch.from_numpy(np.ascontiguousarray(host)).to(dev)
    xm.mark_step()
    dt_async = (time.time() - t0) / 5
    print(f"[PASS] host->xla (8,6,3,224,224) uint8: sync {dt_sync*1000:.1f} ms, "
          f"async {dt_async*1000:.1f} ms")

    x = torch.randn(8 * 6, 3, 224, 224, device=dev)
    theta = torch.zeros(8 * 6, 2, 3, device=dev)
    theta[:, 0, 0] = theta[:, 1, 1] = 1.0
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    y = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    xm.mark_step()
    print("[PASS] grid_sample (augment warp) OK")
except Exception as e:
    print("[FAIL] layer C:", repr(e)[:300])
''',
    # --- cell 5: multi-core spawn + dist + optimizer_step (layer B) ---------- #
    '''try:
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.xla_backend
    import torch.distributed as dist

    def _core(idx):
        d = xm.xla_device()
        dist.init_process_group('xla', init_method='xla://')
        m = nn.Linear(4, 4).to(d)
        xm.broadcast_master_param(m)
        opt = torch.optim.SGD(m.parameters(), lr=0.01)
        for _ in range(3):
            xx = torch.randn(8, 4, device=d)
            loss = m(xx).square().mean()
            loss.backward()
            xm.optimizer_step(opt)
        xm.mark_step()
        if idx == 0:
            print(f"[PASS] spawn+dist+optimizer_step, world={dist.get_world_size()}")

    xmp.spawn(_core, start_method="fork")
    print("[PASS] xmp.spawn returned")
except Exception as e:
    print("[FAIL] layer B spawn:", repr(e)[:300])

print("=== DIAGNOSTIC COMPLETE ===")
''',
]

nb = {
    "cells": [
        {"cell_type": "code", "execution_count": None, "metadata": {},
         "outputs": [], "source": c}
        for c in CELLS
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

DST.parent.mkdir(parents=True, exist_ok=True)
DST.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"wrote {DST} ({len(nb['cells'])} cells)")
