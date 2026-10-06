"""Do the two wireable checkpoints DOCUMENT their geometry? Without that they are the meniscus trap again.

Same architecture family means our loader can LOAD them. It does not mean we know which slot layout, span,
crop, resolution and window count they were trained on -- and `raptor-knee-finespacing` is the proof that
the same architecture can require a different slot vector. Wiring a guessed geometry produces plausible
wrong predictions and no gate in this repository can see it.

Sources of truth, in order: `model_info.json` inside the dataset; the dataset description; the checkpoint's
own keys.
"""
import json
import pathlib
import re

import torch
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

print("=== dataset descriptions ===")
for ref in ("kawaharataishi/rsna-knee-coatnet-distilled-checkpoint",
            "tannguyenngoc/rsna-knee-coatnet-a100-best"):
    slug = ref.split("/", 1)[1]
    try:
        got = api.dataset_list(search=slug) or []
        me = [d for d in got if getattr(d, "ref", "") == ref] or got
        d = me[0] if me else None
    except Exception as e:  # noqa: BLE001
        print("  %s: list failed %s" % (ref, str(e)[:70]))
        d = None
    print("-" * 88)
    print(ref)
    if d is not None:
        print("   title   :", getattr(d, "title", "?"))
        print("   subtitle:", getattr(d, "subtitle", "?"))
        desc = str(getattr(d, "description", "") or "")
        desc = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", desc)
        desc = re.sub(r"<[^>]+>", " ", desc)
        print("   description (%d chars):" % len(desc))
        print("   " + re.sub(r"\s+", " ", desc)[:1400] or "   (empty)")

print()
print("=== model_info.json, if present ===")
mi = sorted(pathlib.Path("work/coat_probe").rglob("model_info.json"))
for p in mi:
    print("-" * 88)
    print(p)
    try:
        print(json.dumps(json.loads(p.read_text(encoding="utf-8")), indent=2)[:2200])
    except Exception as e:  # noqa: BLE001
        print("   parse failed:", str(e)[:90])

print()
print("=== the checkpoints' own metadata keys (do they name a geometry?) ===")
for p in sorted(pathlib.Path("work/coat_probe").rglob("*.pt")):
    try:
        ck = torch.load(p, map_location="cpu", weights_only=False)
    except Exception:
        continue
    if not isinstance(ck, dict):
        continue
    scalar = {k: v for k, v in ck.items()
              if not torch.is_tensor(v) and not isinstance(v, dict) and k != "model"}
    print("-" * 88)
    print("%s" % p.relative_to("work/coat_probe"))
    for k, v in scalar.items():
        print("   %-22s %s" % (k + ":", str(v)[:150]))
    argish = [k for k in ck if re.search(r"arg|cfg|config|slot|span|crop|img|k_eval|norm", str(k), re.I)]
    for k in argish:
        v = ck[k]
        print("   %-22s %s" % (k + ":", str(v)[:220]))
