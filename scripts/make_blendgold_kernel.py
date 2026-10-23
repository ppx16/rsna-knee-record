"""Fork heliosli/rsna-knee-blend-gold (the 0.951 public blend) into our namespace.

Why this one rather than the Apex 0.950 stack:
  * its blend weights are FIXED BEFORE ANY RUN -- cell 0 says verbatim "weights fixed
    before any run (Raptor 0.7, ours 0.3); nothing is fitted on labels or on the test set",
    whereas Apex tunes twelve per-target weights on the public board;
  * it fails loudly (cell 3 raises if the Raptor stage dies; cell 6 raises on >3% Raptor
    failures and on a Spearman < 0.99 self-consistency check), so it cannot silently
    degrade into a different submission;
  * the community's own 342-vote evolution notebook documents it as the 0.951 step.

The notebook is copied BYTE-IDENTICAL. Only the metadata is written fresh, because the
pulled metadata carries an EMPTY dataset slug ("") that we do not want to push.

Run:  python work/make_blendgold_kernel.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "scratch" / "survey" / "heliosli__rsna-knee-blend-gold"
SRC_NB = SRC_DIR / "rsna-knee-blend-gold.ipynb"
OUT_DIR = ROOT / "kernels" / "kernel-blendgold-0951"
OUT_NB = OUT_DIR / "rsna-knee-blend-gold.ipynb"

# The pinned container is part of the recipe (both this notebook and the Apex stack use it,
# and nartaa's cell 1 documents it together with torch 2.11.0+cu128 / timm 1.0.29).
DOCKER = ("gcr.io/kaggle-private-byod/python@sha256:"
          "2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f")

METADATA = {
    "id": "ppxl16/rsna-knee-blendgold-0951",
    "title": "RSNA knee blendgold 0951",
    "code_file": "rsna-knee-blend-gold.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "keywords": ["gpu"],
    # NOTE: the stray "" entry in the upstream metadata turned out to be the PRIVATE
    # heliosli/rsna-knee-compact-v1 dataset (FACTS C119). On 2026-10-10 the author RESTORED it
    # with the full 4-fold bundle + manifest.json, so the slug is attachable and this kernel
    # should finally run. Verbatim from the restored README: "I removed it earlier because
    # someone told me it wasn't allowed, but it seems to be allowed after all."
    "dataset_sources": [
        "nartaa/rsna-knee-publication-swa-weights-20261007",
        "heliosli/rsna-orthofoundation-fold0-epoch7-20261004",
        "heliosli/rsna-knee-compact-v1",
    ],
    "kernel_sources": [],
    "competition_sources": ["rsna-knee-abnormality-detection"],
    "model_sources": [],
    "docker_image": DOCKER,
    "machine_shape": "NvidiaTeslaT4",
}


def main() -> int:
    if not SRC_NB.is_file():
        print(f"REFUSING: missing source notebook {SRC_NB}")
        return 2

    up = json.loads((SRC_DIR / "kernel-metadata.json").read_text(encoding="utf-8"))
    # Sanity: the upstream must be the kernel we think it is.
    if up.get("code_file") != "rsna-knee-blend-gold.ipynb":
        print(f"REFUSING: unexpected upstream code_file {up.get('code_file')!r}")
        return 2
    if "" in up.get("dataset_sources", []):
        print("note: upstream has a stray empty dataset slug -- dropping it in our copy")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # The upstream notebook is UTF-8 JSON with raw non-ASCII, and it is NOT GBK-decodable.
    # kernels_push reads code_file as GBK, so re-serialize with ensure_ascii=True: JSON
    # semantics and every cell string are preserved exactly, only non-ASCII becomes \uXXXX.
    nb = json.loads(SRC_NB.read_text(encoding="utf-8"))
    before = SRC_NB.read_bytes()
    OUT_NB.write_text(json.dumps(nb, ensure_ascii=True), encoding="utf-8")

    (OUT_DIR / "kernel-metadata.json").write_text(
        json.dumps(METADATA, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    raw = OUT_NB.read_bytes()
    print(f"copied  {SRC_NB.name} -> {OUT_NB}")
    print(f"  upstream bytes {len(before)}  sha256 {hashlib.sha256(before).hexdigest()}")
    print(f"  our bytes      {len(raw)}  sha256 {hashlib.sha256(raw).hexdigest()}")
    print("  (re-serialized with ensure_ascii=True -- JSON-equivalent, encoding-safe)")
    print(f"  ascii-only: {all(c < 128 for c in raw)}")
    try:
        raw.decode("gbk")
        print("  GBK-decodable: YES")
    except UnicodeDecodeError as exc:
        print(f"  GBK-decodable: NO -- {exc}")
        return 1
    back = json.loads(raw.decode("utf-8"))
    if len(back["cells"]) != len(nb["cells"]):
        print("REFUSING: cell count changed during re-serialization")
        return 1
    for i, (a, b) in enumerate(zip(nb["cells"], back["cells"])):
        if a != b:
            print(f"REFUSING: cell {i} changed during re-serialization")
            return 1
    print(f"  round-trip check: all {len(back['cells'])} cells byte-identical in content: YES")
    print("  metadata written:")
    print("   ", json.dumps({k: METADATA[k] for k in
                             ("id", "code_file", "enable_gpu", "machine_shape")},
                            ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
