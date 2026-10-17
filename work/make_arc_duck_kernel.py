"""Fork fabioolival/arc-agi-3-v21-taaf-27b -- the reproducible Tufa Labs duck harness.

Why this one (ARC-AGI-3 long-range workstream P1/P2, PLAN-2026-10-11-longrange.md):
  * it is the ONLY reproducible member of the strong "duck harness" family. The canonical
    Tufa Labs notebooks carry EMPTY dataset slugs (invisible private datasets) --
    `jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner` has two of them and is
    therefore NOT reproducible (same trap as FACTS C119/C132).
  * its two dependencies are public and verified present:
        keithtyser/taaf-duck-qwen38-serving-v1
        driessmit1/arc3-vllm-h100-wheelhouse-v3
  * it needs no internet (the model ships in the dataset).
  * ARC scoring is LIVE per run -- our own two runs of one recipe gave 24.53 and 23.67 -- and
    the platform keeps our best, so a submission is upside-only.

HONEST CAVEAT, quoted from its own cell 0: the author states this readable version has NOT
reproduced their milestone-winning score ("unfortunately, we haven't had the same lucky result
with this one"). It is a bet, not a guarantee -- which is exactly why it is worth one daily
submission rather than a rebuild.

Run:  python work/make_arc_duck_kernel.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # rsna-knee/
WS = ROOT.parent                                     # D:\kaggle
SRC_DIR = WS / "arc-agi-3-ledger" / "work" / "probe_arc-agi-3-v21-taaf-27b"
SRC_NB = SRC_DIR / "arc-agi-3-v21-taaf-27b.ipynb"
OUT_DIR = ROOT / "kernels" / "kernel-arc-duck"
OUT_NB = OUT_DIR / "arc-duck-taaf.ipynb"

METADATA = {
    "id": "ppxl16/arc-duck-taaf",
    "title": "ARC duck taaf",
    "code_file": "arc-duck-taaf.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "keywords": ["gpu"],
    "dataset_sources": [
        "keithtyser/taaf-duck-qwen38-serving-v1",
        "driessmit1/arc3-vllm-h100-wheelhouse-v3",
    ],
    "kernel_sources": [],
    "competition_sources": ["arc-prize-2026-arc-agi-3"],
    "model_sources": [],
    # cell 0 says a copy must "manually select the proper GPU (RTX Pro 6000)"; we set it in
    # metadata instead, which is the only way a scripted push can express it.
    "machine_shape": "NvidiaRtxPro6000",
}

REQUIRED = ["Tufa Labs ARC3 submission", "DATASET_BUNDLE_MARKER", "taaf-duck-qwen38-serving-v1",
            "Customization hook"]


def build_metadata(up: dict) -> dict:
    """Inherit the ENTIRE upstream metadata and override only what we must.

    Written after a real failure (2026-10-11): the first version of this builder wrote a
    hand-copied metadata dict and silently DROPPED two fields the kernel genuinely needs --
    `model_sources` (the Qwen3.8 27B mount) and `docker_image` (a different image from the
    RSNA one). The run died with `FileNotFoundError: Kaggle model root is absent:
    /kaggle/input/models`, i.e. exactly the FACTS A170/A183 family: a receipt that describes
    what we intended rather than what the kernel requires.

    So: never re-type an environment. Copy the upstream dict and change only identity fields.
    """
    OVERRIDE = {
        "id": "ppxl16/arc-duck-taaf",
        "title": "ARC duck taaf",
        "code_file": "arc-duck-taaf.ipynb",
        "is_private": True,
    }
    meta = dict(up)
    meta.update(OVERRIDE)
    meta.pop("id_no", None)          # stale upstream id; Kaggle reassigns
    # Everything else -- dataset_sources, model_sources, kernel_sources, competition_sources,
    # docker_image, machine_shape, enable_* -- is inherited verbatim, and we ASSERT that.
    KEEP = ("dataset_sources", "model_sources", "kernel_sources", "competition_sources",
            "docker_image", "machine_shape", "enable_gpu", "enable_internet")
    for k in KEEP:
        if up.get(k) != meta.get(k):
            raise SystemExit(f"REFUSING: field {k!r} was not inherited verbatim")
        if k in ("dataset_sources", "model_sources") and not meta.get(k):
            raise SystemExit(f"REFUSING: inherited {k!r} is empty -- the kernel needs it")
    return meta


def main() -> int:
    if not SRC_NB.is_file():
        print(f"REFUSING: missing {SRC_NB}")
        return 2
    up = json.loads((SRC_DIR / "kernel-metadata.json").read_text(encoding="utf-8"))
    bad = [d for d in up.get("dataset_sources", []) if not (isinstance(d, str) and d.strip())]
    if bad:
        print(f"REFUSING: upstream has empty dataset slugs ({len(bad)}) == invisible private "
              f"datasets (FACTS C119). Do not fork an unreproducible kernel.")
        return 2

    nb = json.loads(SRC_NB.read_text(encoding="utf-8"))
    text = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
    missing = [m for m in REQUIRED if m not in text]
    if missing:
        print(f"REFUSING: missing markers {missing}")
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_NB.write_text(json.dumps(nb, ensure_ascii=True), encoding="utf-8")
    meta = build_metadata(up)          # inherit the environment verbatim, override only identity
    (OUT_DIR / "kernel-metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("  inherited:", {k: meta.get(k) for k in
                          ("model_sources", "docker_image", "machine_shape", "enable_gpu")})

    raw = OUT_NB.read_bytes()
    print(f"upstream {len(SRC_NB.read_bytes())} B  ->  ours {len(raw)} B "
          f"sha256 {hashlib.sha256(raw).hexdigest()}")
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
    print("  datasets:", METADATA["dataset_sources"])
    print("  machine_shape:", METADATA["machine_shape"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
