"""Build the "fitted 0.954" variant on top of OUR OWN verified triple-reader.

Decision context (2026-10-10, user's call): we now HOLD bronze (rank 516, cut 566, margin +50,
FACTS C129). The upstream author `rabari9999` published a variant whose title claims LB 0.954 at
rank 127 (silver), and later proof `...lb-0-954-rank-127-silver`. A full diff proves it differs
from the 0.951 triple-reader ONLY in the twelve TARGET_WEIGHTS (plus an unused numpy import);
cells 1-9 are byte-identical.

So this builder starts from OUR triple-reader notebook -- the one we built, verified and scored
0.950 with -- and changes ONLY that dict. That keeps the diff auditable and the lineage single,
instead of forking a third-party notebook wholesale.

HONEST LABEL: the upstream comment on these numbers reads
    "# CLINICAL SPECIALIST WEIGHTING MATRIX (TARGET: >= 0.954 LB)"
i.e. they were iterated against the public leaderboard. We do NOT know whether the twelve numbers
were chosen on the LB or on a clean holdout, and that is exactly what decides whether 0.954
survives on the private split. This variant is therefore an EXPERIMENT, not a replacement: it must
not be ticked unless its measured behaviour justifies the swap.

Run:  python work/make_0954_kernel.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_NB = ROOT / "kernels" / "kernel-triplereader-0951" / "rsna-knee-triple-reader-0951.ipynb"
OUT_DIR = ROOT / "kernels" / "kernel-fitted-0954"
OUT_NB = OUT_DIR / "rsna-knee-fitted-0954.ipynb"

DOCKER = ("gcr.io/kaggle-private-byod/python@sha256:"
          "2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f")

METADATA = {
    "id": "ppxl16/rsna-knee-fitted-0954",
    "title": "RSNA knee fitted 0954",
    "code_file": "rsna-knee-fitted-0954.ipynb",
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

OLD = (
    'TARGET_WEIGHTS = {"Baker\'s": 0.35, "Contusion": 0.30, "Medial OA": 0.25, "ACL": 0.25, '
    '"Medial Meniscus": 0.20,\n'
    '                  "MCL": 0.20, "Lateral Meniscus": 0.08, "Fracture": 0.08, "Lateral OA": 0.08, '
    '"PF OA": 0.08,\n'
    '                  "Effusion": 0.04, "Synovitis": 0.03}'
)

NEW = (
    '# [FITTED-0954] Twelve per-target reader weights taken verbatim from the upstream\n'
    '# "0.954" variant. Its own comment reads "TARGET: >= 0.954 LB", i.e. these numbers were\n'
    '# iterated against the public leaderboard. Everything else in this notebook is our own\n'
    '# verified 0.950 triple-reader, unchanged.\n'
    'TARGET_WEIGHTS = {\n'
    '    "Lateral OA": 0.50,\n'
    '    "Baker\'s": 0.42,\n'
    '    "Contusion": 0.36,\n'
    '    "ACL": 0.30,\n'
    '    "Medial OA": 0.28,\n'
    '    "MCL": 0.25,\n'
    '    "Medial Meniscus": 0.22,\n'
    '    "Lateral Meniscus": 0.08,\n'
    '    "Fracture": 0.08,\n'
    '    "PF OA": 0.08,\n'
    '    "Effusion": 0.04,\n'
    '    "Synovitis": 0.03,\n'
    '}'
)

REQUIRED_MARKERS = ["FITTED-0954", '"Lateral OA": 0.50', "W_C224, W_OWN = 0.3, 0.0",
                    "No fallback on purpose"]


def main() -> int:
    if not SRC_NB.is_file():
        print(f"REFUSING: missing source {SRC_NB} (build the triple-reader first)")
        return 2
    nb = json.loads(SRC_NB.read_text(encoding="utf-8"))
    if len(nb["cells"]) != 11:
        print(f"REFUSING: expected 11 cells, got {len(nb['cells'])}")
        return 2

    last = nb["cells"][-1]
    src = "".join(last["source"])
    n = src.count(OLD)
    if n != 1:
        print(f"REFUSING: the old TARGET_WEIGHTS literal appears {n} times, expected exactly 1")
        # help the caller see what the cell actually holds
        for i, line in enumerate(src.splitlines()):
            if "TARGET_WEIGHTS" in line:
                print(f"   line {i}: {line[:140]}")
        return 2

    src = src.replace(OLD, NEW)
    last["source"] = src.splitlines(keepends=True)
    last["outputs"] = []
    last["execution_count"] = None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_NB.write_text(json.dumps(nb, ensure_ascii=True), encoding="utf-8")
    (OUT_DIR / "kernel-metadata.json").write_text(
        json.dumps(METADATA, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    raw = OUT_NB.read_bytes()
    text = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
    missing = [m for m in REQUIRED_MARKERS if m not in text]
    if missing:
        print(f"REFUSING: markers missing after the edit: {missing}")
        return 2
    if OLD in text:
        print("REFUSING: the old weights are still present")
        return 2

    print(f"source : {SRC_NB.parent.name}/{SRC_NB.name}")
    print(f"output : {OUT_NB}  {len(raw)} bytes  sha256 {hashlib.sha256(raw).hexdigest()}")
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
    print("  metadata id:", METADATA["id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
