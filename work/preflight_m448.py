"""Local preflight for the public BTKR M448 member, BEFORE spending any GPU.

The two failure modes that would waste a kernel run:
  1. the checkpoint's SHA-256 does not match what the receipt pins;
  2. timm does not know the backbone name `convnext_small.dinov3_lvd1689m`
     (the script builds with pretrained=False, so no download is needed -- but the
      architecture must exist in whatever timm version the Kaggle image ships).
Also confirms the checkpoint is the fold-0 primary and its cfg matches the script's
expectations, without constructing the whole model.
"""
import hashlib
import pathlib
import sys

X = pathlib.Path(r"D:\kaggle\rsna-knee\work\cnx\x")
CKPT = X / "cnx_montage448_f0.pt"
EXPECTED_SHA = "3b7f50c7bd16d353b8d79bf6089530a376009b6c3480e1bd97b778aa1f4ecab8"

print("--- 1. checkpoint SHA-256 ---")
h = hashlib.sha256()
with CKPT.open("rb") as f:
    for chunk in iter(lambda: f.read(8 << 20), b""):
        h.update(chunk)
got = h.hexdigest()
print(f"  expected {EXPECTED_SHA}")
print(f"  actual   {got}")
print("  MATCH" if got == EXPECTED_SHA else "  *** MISMATCH ***")

print("\n--- 2. torch / timm availability ---")
try:
    import torch
    print(f"  torch {torch.__version__}")
except Exception as e:
    print(f"  torch MISSING: {e}")
    sys.exit(1)

try:
    import timm
    print(f"  timm {timm.__version__}")
except Exception as e:
    print(f"  timm MISSING: {e}")
    timm = None

print("\n--- 3. timm knows the backbone? ---")
BACKBONE = "convnext_small.dinov3_lvd1689m"
if timm is not None:
    try:
        names = set(timm.list_models(pretrained=False))
        print(f"  exact '{BACKBONE}' in list_models: {BACKBONE in names}")
        near = sorted(n for n in names if "convnext_small" in n and "dinov3" in n)
        print(f"  convnext_small *dinov3* variants: {near}")
        m = timm.create_model(BACKBONE, pretrained=False, num_classes=0, in_chans=3)
        print(f"  create_model OK -> {type(m).__name__}, num_features={m.num_features}")
        del m
    except Exception as e:
        print(f"  create_model FAILED: {type(e).__name__}: {e}")

print("\n--- 4. checkpoint payload (metadata only) ---")
payload = torch.load(CKPT, map_location="cpu", weights_only=False)
print(f"  top-level keys: {sorted(payload.keys())}")
for k in ("selection_role", "submission_eligible", "fold", "epoch"):
    print(f"  {k} = {payload.get(k)!r}")
cfg = payload.get("cfg", {})
print(f"  cfg keys: {len(cfg)}")
expected = {
    "backbone": BACKBONE, "pool": "hier_slot_mil", "cond": "post",
    "norm": "imagenet", "triplet": "montage4", "depth_remask": True,
    "montage_position_tokens": True, "montage_budget_profile": "anatomy_axfs",
    "img": 448, "meta": "none", "n_meta": 0, "site_mode": "none", "stem": "native",
}
mismatch = {k: (cfg.get(k), v) for k, v in expected.items() if cfg.get(k) != v}
print(f"  cfg mismatch vs script expectations: {mismatch if mismatch else 'NONE'}")

state = payload.get("state_dict")
print(f"  state_dict tensors: {len(state) if state else 0}")
if state:
    keys = list(state.keys())
    print(f"  first 5 keys: {keys[:5]}")
    print(f"  last 3 keys: {keys[-3:]}")
