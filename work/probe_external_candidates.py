"""Settle the questions the external-data sweep could not: weights vs DATA, and licences.

The sweep could only read search-result TITLES (`web_fetch` is denied for non-public IPs here), so it
had to mark every size and licence "unverified" and left three questions for "a human with browser
access". The Kaggle API is not affected by that restriction -- it has worked all session.

Decisive distinction: files ending .pt/.pth/.safetensors are WEIGHTS (a pre-trained model, explicitly
allowed by the rules) and carry NO new patients. Image-like extensions *are* data.

⚠️ TWO API TRAPS, both hit while writing this:
  * `dataset_metadata(dataset, path)` is for reading a LOCAL metadata file, not a remote dataset --
    calling it with one argument raises TypeError, which the first version reported as a failure.
  * the lister is `dataset_list_files`, not `dataset_files_list`.
"""
from kaggle.api.kaggle_api_extended import KaggleApi

CANDIDATES = [
    "dreaddevelopment/raptor-knee-arms",
    "dreaddevelopment/raptor-knee-arms-x",
    "dreaddevelopment/raptor-knee-cnn336",
    "dreaddevelopment/raptor-knee-widedense",
    "dreaddevelopment/raptor-knee-fullspan",
    "prvsiyan/orthodiffusion-official-knee-mri-foundation",
    "prvsiyan/rsna-knee-pulse-v6-checkpoints-public-20260813",
]

WEIGHT_EXT = (".pt", ".pth", ".safetensors", ".ckpt", ".bin", ".onnx", ".h5", ".pb", ".tflite")
DATA_EXT = (".npy", ".npz", ".dcm", ".dicom", ".png", ".jpg", ".jpeg", ".mha", ".nii", ".nii.gz")

api = KaggleApi()
api.authenticate()


def mb(n):
    try:
        n = float(n or 0)
    except Exception:
        return "?"
    return "%.1f MB" % (n / 1e6) if n < 1e9 else "%.2f GB" % (n / 1e9)


for ref in CANDIDATES:
    owner, slug = ref.split("/", 1)
    print("=" * 96)
    print(ref)
    try:
        got = api.dataset_list(search=slug) or []
        me = [d for d in got if getattr(d, "ref", "") == ref]
        d = (me or got or [None])[0]
    except Exception as e:  # noqa: BLE001
        print("   list FAILED: %s: %s" % (type(e).__name__, str(e)[:110]))
        d = None
    if d is not None:
        for f in ("title", "subtitle", "licenseName", "totalBytes", "usabilityRating", "lastUpdated"):
            v = getattr(d, f, None)
            if v is not None:
                print("   %-16s %s" % (f + ":", v))
    try:
        fl = api.dataset_list_files(ref)
        files = getattr(fl, "files", None) or fl or []
    except Exception as e:  # noqa: BLE001
        print("   files FAILED: %s: %s" % (type(e).__name__, str(e)[:110]))
        continue
    w = dd = other = 0
    print("   %d file(s)" % len(files))
    for f in files:
        nm = getattr(f, "name", "") or getattr(f, "fileName", "") or ""
        sz = getattr(f, "totalBytes", 0) or getattr(f, "size", 0) or 0
        low = nm.lower()
        if low.endswith(WEIGHT_EXT):
            kind, w = "WEIGHT", w + 1
        elif low.endswith(DATA_EXT):
            kind, dd = "DATA?", dd + 1
        else:
            kind, other = "other", other + 1
        print("      %-8s %-11s %s" % (kind, mb(sz), nm))
    print("   => weights=%d data=%d other=%d  VERDICT: %s"
          % (w, dd, other,
             "WEIGHTS ONLY - no new patients" if w and not dd else
             ("DATA PRESENT - inspect" if dd else "unclear")))
