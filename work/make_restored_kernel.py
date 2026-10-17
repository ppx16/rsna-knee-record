"""Fork heliosli/rsna-knee-blend-gold-dataset-restored -- the reproducible 0.951 recipe.

Why this one, and why now:
  * it is the author's CURRENT artifact (161 votes), and it is the one whose second leg is the
    restored S6 compact bundle: cell 0 says verbatim "Stage 2 runs our compact convnext_small
    model S6 (4 fold models, arm E_lrprior, bundle 209a32c8b50a6b1c) unchanged", and
    "weights fixed before any run (Raptor 0.7, ours 0.3); nothing is fitted on labels or on the
    test set";
  * ALL of its dataset slugs are present and public -- verified, not assumed (the sibling
    "one-click-955" kernel, by contrast, carries an EMPTY slug, i.e. an invisible private
    dataset, so it is NOT reproducible: FACTS C119/C132);
  * it is the Work-7 recipe that the community's own 342-vote evolution notebook documents as
    the step above the 0.950 Apex stack.

The notebook is re-serialized with ensure_ascii=True (A180: kernels_push reads code_file as GBK)
and verified cell-by-cell on round-trip.

Run:  python work/make_restored_kernel.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "scratch" / "hrest"
SRC_NB = SRC_DIR / "rsna-knee-blend-gold-dataset-restored.ipynb"
OUT_DIR = ROOT / "kernels" / "kernel-blendgold-restored"
OUT_NB = OUT_DIR / "rsna-knee-blendgold-restored.ipynb"

DOCKER = ("gcr.io/kaggle-private-byod/python@sha256:"
          "2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f")

METADATA = {
    "id": "ppxl16/rsna-knee-blendgold-restored",
    "title": "RSNA knee blendgold restored",
    "code_file": "rsna-knee-blendgold-restored.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "keywords": ["gpu"],
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

REQUIRED_MARKERS = [
    "weights fixed before any run",
    "W_RAPTOR, W_OURS = 0.7, 0.3",
    "No fallback on purpose" if False else "raise RuntimeError",
    "209a32c8b50a6b1c",
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
        print(f"REFUSING: upstream carries an EMPTY dataset slug {bad!r} == an invisible private "
              f"dataset (FACTS C119/C132). Do not fork an unreproducible kernel.")
        return 2

    nb = json.loads(SRC_NB.read_text(encoding="utf-8"))
    before = SRC_NB.read_bytes()
    text = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
    missing = [m for m in REQUIRED_MARKERS if m not in text]
    if missing:
        print(f"REFUSING: markers missing: {missing}")
        return 2
    if "TARGET_WEIGHTS" in text or '"Lateral OA": 0.50' in text:
        print("REFUSING: this notebook contains per-target FITTED weights -- that is the 0.954 "
              "variant we already measured at 0.950, not the a-priori-fixed 0.951 recipe")
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_NB.write_text(json.dumps(nb, ensure_ascii=True), encoding="utf-8")
    (OUT_DIR / "kernel-metadata.json").write_text(
        json.dumps(METADATA, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    raw = OUT_NB.read_bytes()
    print(f"upstream {len(before)} B sha256 {hashlib.sha256(before).hexdigest()}")
    print(f"ours     {len(raw)} B sha256 {hashlib.sha256(raw).hexdigest()}")
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
            print(f"REFUSING: cell {i} changed on round-trip")
            return 1
    print(f"  round-trip: all {len(back['cells'])} cells identical: YES")
    print("  markers:", REQUIRED_MARKERS)
    print("  datasets:", METADATA["dataset_sources"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
