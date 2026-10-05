"""Inspect candidate competition models from the Kaggle MODELS hub.

WHY THIS ORDER. Adding a Raptor view is cheap because `ARMS` + `load_model` already exist and the
checkpoint carries its own `arch`. A FOREIGN architecture is a different thing: it needs a new loader
and a faithful reproduction of *that author's* preprocessing geometry (the same trap the meniscus arm
flagged). So the cheap test is a model whose **family we can already load**:

  * `ruicompany/rsna-knee-a5-rad`            -> the A5 / DINOv2 lineage we already run
  * `yunusgmsoy/rsna-knee-radhead-resnet50`  -> a RadImageNet head, same family as our _RAD_E11/E13

The others (`convnextv2-pool`, `pillar0/atlas-336-2ep`) may be better models but are a project each.

⚠️ Model refs need `{owner}/{model-slug}/{framework}/{instance}`; passing the short ref raises a
ValueError that says exactly this, which is how the format was learned.
"""
import inspect

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

CANDIDATES = [
    ("ruicompany/rsna-knee-a5-rad", "pytorch"),
    ("yunusgmsoy/rsna-knee-radhead-resnet50", "5-fold-yunus-v5"),
    ("yaolinghan/rsna-knee-pool-models", "convnextv2-pool"),
    ("fabotelli/pillar0-knee-finetune", "atlas-336-2ep"),
    ("beraterolelk/rsna-knee-25d-attention-backbone", "default"),
]

print("signature:", inspect.signature(api.model_instance_files))
print()

for base, inst in CANDIDATES:
    print("=" * 94)
    print("%s  ::  %s" % (base, inst))
    ok = False
    for fw in ("PyTorch", "pytorch", "TensorFlow", "tensorflow", "JAX", "jax", "ONNX", "onnx"):
        ref = "%s/%s/%s" % (base, fw, inst)
        try:
            fl = api.model_instance_files(ref)
        except Exception as e:  # noqa: BLE001
            msg = str(e)[:60]
            if "must be specified in the form" in msg or "Forbidden" in msg or "Not Found" in msg:
                continue
            print("   %-12s -> %s: %s" % (fw, type(e).__name__, msg))
            continue
        files = getattr(fl, "files", None) or fl or []
        print("   framework=%s  %d file(s)" % (fw, len(files)))
        for f in files[:14]:
            nm = getattr(f, "name", "") or getattr(f, "fileName", "") or ""
            sz = getattr(f, "totalBytes", 0) or getattr(f, "size", 0) or 0
            try:
                sz = "%.1f MB" % (float(sz) / 1e6)
            except Exception:
                sz = "?"
            print("        %-11s %s" % (sz, nm))
        if len(files) > 14:
            print("        ... %d more" % (len(files) - 14))
        ok = True
        break
    if not ok:
        print("   could not list files under any framework spelling")
