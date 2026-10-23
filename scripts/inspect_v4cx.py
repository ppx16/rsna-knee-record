import torch, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ck = torch.load(r"work\v4cx\convnext_small.in12k__latL__v4cx0_best.pt",
                map_location="cpu", weights_only=False)
print("top-level type:", type(ck).__name__)
if isinstance(ck, dict):
    print("top-level keys:", list(ck)[:12])
    for w in ("model","student","state_dict"):
        if isinstance(ck.get(w), dict):
            print("  -> unwrapping %r" % w); ck = ck[w]; break
print("epoch:", ck.get("epoch","?") if isinstance(ck,dict) else "?")
bb = {k:v for k,v in ck.items() if k.startswith("backbone.")}
hd = {k:v for k,v in ck.items() if not k.startswith("backbone.") and torch.is_tensor(v)}
print("\nbackbone tensors: %d   head tensors: %d" % (len(bb), len(hd)))
print("\n---- HEAD (non-backbone) ----")
for k in sorted(hd):
    print("   %-28s %s" % (k, tuple(hd[k].shape)))
print("\n---- a few backbone keys ----")
for k in sorted(bb)[:3] + sorted(bb)[-3:]:
    print("   %-58s %s" % (k, tuple(bb[k].shape)))
