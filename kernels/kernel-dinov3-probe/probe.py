"""Does this image have a DINOv3 backbone we can actually load, offline or online?

WHY A SEPARATE ZERO-QUOTA PROBE
    scripts/arm_parity.py showed our arms (0.8769) already beat the frontier's own DINOv2
    seed-members (0.8676-0.8697), so A106's "training line closed, we are 0.04 behind" is a
    statement about A5 -- the one DINOv3 model in the stack.  That makes "can we train a DINOv3
    arm?" the next real question, and it is worth GPU/TPU hours only if the backbone is reachable.

    A failed training run costs queue time and quota and reports nothing useful; this costs a
    couple of CPU minutes and answers it beforehand.  The same discipline as scripts/preflight.

WHAT IT REPORTS
    * timm / torch / torch_xla versions in the actual image (a local check proves nothing --
      this repo has no timm installed locally at all).
    * every DINOv3 model name timm knows.
    * whether the smallest one can be CREATED with pretrained weights, and what num_features it
      reports -- the trainer's build_backbone() uses num_features to size the head, and the
      RaptorClassifier hardcodes F_dim only by default, so a mismatch matters.
    * whether dynamic_img_size is accepted, because the trainer sets it for ViT-family backbones
      and feeds 336px crops.

    Output is written to stdout and to a small json so it can be read from the kernel output.
"""
import json
import os
import traceback

REPORT = {}


def main():
    import torch
    REPORT["torch"] = torch.__version__
    print("[probe] torch", torch.__version__, flush=True)
    try:
        import torch_xla
        REPORT["torch_xla"] = getattr(torch_xla, "__version__", "?")
    except Exception as e:                                   # CPU sessions usually lack it
        REPORT["torch_xla"] = "absent: %s" % type(e).__name__
    print("[probe] torch_xla", REPORT["torch_xla"], flush=True)

    import timm
    REPORT["timm"] = timm.__version__
    print("[probe] timm", timm.__version__, flush=True)

    names = list(timm.list_models("*dinov3*"))
    REPORT["dinov3_names"] = names
    print("[probe] dinov3 models timm knows: %d" % len(names), flush=True)
    for n in names:
        print("        ", n, flush=True)

    # the families the trainer would use at 336px
    for n in ("vit_small_patch16_dinov3", "vit_small_patch16_dinov3.lvd1689m",
              "vit_base_patch16_dinov3", "vit_small_patch16_224"):
        try:
            m = timm.create_model(n, pretrained=False, num_classes=0, in_chans=3,
                                  global_pool="token", dynamic_img_size=True)
            REPORT.setdefault("buildable", {})[n] = int(m.num_features)
            print("[probe] create_model OK  %-38s num_features=%d" % (n, m.num_features), flush=True)
        except Exception as e:
            print("[probe] create_model FAIL %-38s %s: %s" % (n, type(e).__name__, e), flush=True)

    # can we actually FETCH pretrained weights?  internet is on for this kernel by design.
    fetch = None
    for n in names:
        if "small" in n:
            fetch = n
            break
    fetch = fetch or (names[0] if names else None)
    if fetch:
        try:
            m = timm.create_model(fetch, pretrained=True, num_classes=0, in_chans=3,
                                  global_pool="token", dynamic_img_size=True)
            m.eval()
            with torch.no_grad():
                y = m(torch.zeros(1, 3, 336, 336))
            REPORT["pretrained_fetch"] = {"name": fetch, "ok": True, "out_shape": list(y.shape)}
            print("[probe] PRETRAINED FETCH OK: %s -> %s" % (fetch, tuple(y.shape)), flush=True)
        except Exception as e:
            REPORT["pretrained_fetch"] = {"name": fetch, "ok": False,
                                          "err": "%s: %s" % (type(e).__name__, e)}
            print("[probe] PRETRAINED FETCH FAIL: %s" % e, flush=True)
            traceback.print_exc()

    REPORT["HF_HOME"] = os.environ.get("HF_HOME", "")
    out = "/kaggle/working/dinov3_probe.json"
    with open(out, "w") as f:
        json.dump(REPORT, f, indent=2)
    print("[probe] wrote", out, flush=True)


main()
