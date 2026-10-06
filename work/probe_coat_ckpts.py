"""Inspect the newly-surfaced CoAtNet checkpoints BEFORE believing any of them is wireable.

Why this shape of check: wiring a foreign checkpoint into `ARMS` requires it to be the SAME family our
loader handles -- `coatnet_rmlp_2_rw_384`, backbone tail `backbone.norm.*`, and the 8-tensor head
`clsW (12,1024) / clsb / norm.{weight,bias} / att.0.* / att.3.*`. That is exactly what the `raptor-knee-arms`
investigation established, and the `hunter_pool` case showed what happens otherwise: a genuinely
different model with NO documented geometry, which cannot be wired responsibly at all.

So: download the small ones and read their state dicts. The head shapes answer the question in one line.
"""
import pathlib
import zipfile

import torch
from kaggle.api.kaggle_api_extended import KaggleApi

OUT = pathlib.Path("work/coat_probe")
OUT.mkdir(parents=True, exist_ok=True)

REFS = [
    "kawaharataishi/rsna-knee-coatnet-distilled-checkpoint",
    "myominhtet/rsna-knee-coat5fold",
    "haideptry/rsna-knee-coatnet-25d-masked-checkpoint",
    "tannguyenngoc/rsna-knee-coatnet-a100-best",
]

api = KaggleApi()
api.authenticate()

HEAD = ("clsW", "clsb", "norm.weight", "norm.bias", "att.0.weight", "att.0.bias",
        "att.3.weight", "att.3.bias")

for ref in REFS:
    d = OUT / ref.split("/", 1)[1]
    d.mkdir(parents=True, exist_ok=True)
    print("=" * 96)
    print(ref)
    if not any(d.rglob("*.pt")):
        try:
            api.dataset_download_files(ref, path=str(d), unzip=True, quiet=True)
        except Exception as e:  # noqa: BLE001
            print("   download FAILED: %s: %s" % (type(e).__name__, str(e)[:100]))
            continue
    pts = sorted(d.rglob("*.pt"))
    print("   %d checkpoint(s)" % len(pts))
    for p in pts[:3]:
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
        except Exception as e:  # noqa: BLE001
            print("   %-34s LOAD FAILED %s" % (p.name, str(e)[:70]))
            continue
        top = list(ck)[:10] if isinstance(ck, dict) else type(ck).__name__
        print("   -- %s (%.1f MB) top=%s" % (p.name, p.stat().st_size / 1e6, top))
        if not isinstance(ck, dict):
            continue
        for k in ("arch", "res", "gold_auc", "src", "lab", "epoch"):
            if k in ck and not torch.is_tensor(ck[k]):
                print("        %-9s %s" % (k + ":", str(ck[k])[:90]))
        sd = ck.get("model") if isinstance(ck.get("model"), dict) else ck
        if not isinstance(sd, dict):
            continue
        shapes = {k: tuple(v.shape) for k, v in sd.items() if hasattr(v, "shape")}
        print("        tensors %d" % len(shapes))
        present = {h: shapes.get(h) for h in HEAD if h in shapes}
        missing = [h for h in HEAD if h not in shapes]
        for h, s in present.items():
            print("        HEAD %-14s %s" % (h, s))
        if missing:
            print("        HEAD missing: %s" % missing)
        tails = [k for k in shapes if k.endswith(("backbone.norm.weight", "backbone.head.norm.weight",
                                                  "backbone.norm.bias"))]
        print("        backbone tail: %s" % [(k, shapes[k]) for k in tails])
        verdict = ("SAME FAMILY (wireable as a CoAt member)" if not missing
                   and any(k.endswith("backbone.norm.weight") for k in shapes)
                   else "DIFFERENT / unclear -- would need its own loader and geometry")
        print("        VERDICT: %s" % verdict)
