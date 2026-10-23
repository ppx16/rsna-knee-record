"""Fork rabari9999/rsna-knee-sota-dual-engine-stack-lb-0-951 (claim: 0.951).

Why this one, after blend-gold turned out to be built on a PRIVATE dataset (C119):
  * its inputs are ALL public and already attached elsewhere --
      goodpjw2008/rsna-knee-2-5d-convnext-reader  (3 fold reader)
      nartaa/rsna-knee-publication-swa-weights-20261007  (BOTH checkpoints:
          ..._r384_swa.pt = 0.949  AND  ..._c08_r224_swa.pt = 0.945)
    and its metadata has NO empty slug and NO kernel_sources;
  * the change versus Apex is MECHANISTIC, not another public-board knob: it adds
    nartaa's 224-crop CoAtNet as an orthogonal SCALE, fused at a fixed 0.7/0.3:
        coat = rank(0.7 * rank(c384) + 0.3 * rank(c224))
    then applies the same twelve per-target reader weights Apex uses;
  * it fails loudly on every leg ("No fallback on purpose"), so it cannot silently
    degrade into a fewer-leg submission.

The notebook is re-serialized with ensure_ascii=True (see A180: kernels_push reads
code_file as GBK) and verified cell-by-cell round-trip.

Run:  python work/make_triple_kernel.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "scratch" / "r951"
SRC_NB = SRC_DIR / "rsna-knee-sota-dual-engine-stack-lb-0-951.ipynb"
OUT_DIR = ROOT / "kernels" / "kernel-triplereader-0951"
OUT_NB = OUT_DIR / "rsna-knee-triple-reader-0951.ipynb"

DOCKER = ("gcr.io/kaggle-private-byod/python@sha256:"
          "2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f")

METADATA = {
    # NOTE: the id must be the SLUG OF THE TITLE. Kaggle derives the real slug from the
    # title and only WARNS if the id does not resolve to it (A183) -- the first push went
    # out as ".../rsna-knee-triple-reader-0951" while this file said ".../triplereader-0951",
    # and `kernels status` on the latter answers "Cannot access kernel".
    "id": "ppxl16/rsna-knee-triple-reader-0951",
    "title": "RSNA knee triple reader 0951",
    "code_file": "rsna-knee-triple-reader-0951.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "keywords": ["gpu"],
    "dataset_sources": [
        "goodpjw2008/rsna-knee-2-5d-convnext-reader",
        "nartaa/rsna-knee-publication-swa-weights-20261007",
    ],
    "kernel_sources": [],
    "competition_sources": ["rsna-knee-abnormality-detection"],
    "model_sources": [],
    "docker_image": DOCKER,
    "machine_shape": "NvidiaTeslaT4",
}

# Markers that prove we forked the notebook we think we did.
REQUIRED_MARKERS = [
    "W_C224, W_OWN = 0.3, 0.0",
    "TARGET_WEIGHTS",
    "No fallback on purpose",
    "_sota_0949",
]


def main() -> int:
    if not SRC_NB.is_file():
        print(f"REFUSING: missing {SRC_NB}")
        return 2
    up = json.loads((SRC_DIR / "kernel-metadata.json").read_text(encoding="utf-8"))
    if up.get("code_file") != SRC_NB.name:
        print(f"REFUSING: upstream code_file {up.get('code_file')!r} != {SRC_NB.name!r}")
        return 2
    bad = [d for d in up.get("dataset_sources", []) if not (isinstance(d, str) and d.strip())]
    if bad:
        print(f"REFUSING: upstream has an empty dataset slug {bad!r} -- an invisible PRIVATE dataset")
        print("          (that is exactly what killed blend-gold; see FACTS C119)")
        return 2

    nb = json.loads(SRC_NB.read_text(encoding="utf-8"))
    before = SRC_NB.read_bytes()
    text = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
    missing = [m for m in REQUIRED_MARKERS if m not in text]
    if missing:
        print(f"REFUSING: notebook is missing expected markers {missing}")
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_NB.write_text(json.dumps(nb, ensure_ascii=True), encoding="utf-8")
    (OUT_DIR / "kernel-metadata.json").write_text(
        json.dumps(METADATA, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    raw = OUT_NB.read_bytes()
    print(f"upstream {len(before)} bytes sha256 {hashlib.sha256(before).hexdigest()}")
    print(f"ours     {len(raw)} bytes sha256 {hashlib.sha256(raw).hexdigest()}")
    print(f"  cells {len(nb['cells'])}  ascii-only {all(c < 128 for c in raw)}")
    try:
        raw.decode("gbk")
        print("  GBK-decodable: YES")
    except UnicodeDecodeError as exc:
        print(f"  GBK-decodable: NO -- {exc}")
        return 1
    back = json.loads(raw.decode("utf-8"))
    for i, (a, b) in enumerate(zip(nb["cells"], back["cells"])):
        if a != b:
            print(f"REFUSING: cell {i} changed during re-serialization")
            return 1
    print(f"  round-trip: all {len(back['cells'])} cells identical in content: YES")
    print("  markers present:", REQUIRED_MARKERS)
    print("  metadata id:", METADATA["id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
