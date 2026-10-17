"""Build the "pure 0.949" hedge variant of the Apex kernel.

Why: the Apex fusion applies twelve per-target rank weights that were tuned on the
public board (its own comments cite per-finding deltas). That is one more fitted knob
on the public split, and the author already concedes gold58 was reused for selection.
The raw nartaa checkpoint scores 0.949 on its own with no such knobs, so it is the
lower-variance asset for the private split.

This variant keeps the upstream notebook byte-identical except for the LAST cell,
which is replaced by a self-verifying no-op: it asserts that submission.csv is still
exactly the pure checkpoint output.

Run:  python work/make_pure_variant.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "scratch" / "apex" / "apex-nartaa-rank-blend-0-950-lb.ipynb"
OUT_DIR = ROOT / "kernels" / "kernel-apex-pure0949"
OUT_NB = OUT_DIR / "apex-pure0949.ipynb"

VERIFY_CELL = '''# --- pure-0.949 variant: the Apex per-target rank fusion is intentionally SKIPPED ---
# The upstream notebook's SOTA cell already wrote BOTH:
#   /kaggle/working/submission.csv    <- the pure checkpoint result
#   /kaggle/working/_sota_0949.csv    <- the same thing, as a backup
# The Apex fusion step (twelve per-target weights tuned on the public board) is the
# only thing this variant drops, so the deliverable carries no public-board-fitted knob.
import os
import pandas as pd

_pub = "/kaggle/working/submission.csv"
_sota = "/kaggle/working/_sota_0949.csv"

for _p in (_pub, _sota):
    assert os.path.exists(_p), f"missing expected artefact: {_p}"

_a = pd.read_csv(_pub, dtype={"StudyInstanceUID": str})
_b = pd.read_csv(_sota, dtype={"StudyInstanceUID": str})
assert list(_a.columns) == list(_b.columns), "column mismatch between pub and sota backup"
assert _a["StudyInstanceUID"].tolist() == _b["StudyInstanceUID"].tolist(), "row order mismatch"
assert _a.drop(columns=["StudyInstanceUID"]).equals(
    _b.drop(columns=["StudyInstanceUID"])
), "submission.csv was modified -- fusion was NOT supposed to run"
assert _a.notna().all().all(), "NaN in submission"
assert ( _a.drop(columns=["StudyInstanceUID"]).nunique() > 1).all(), "constant column in submission"

print(f"[pure0949] OK {len(_a)} rows x {len(_a.columns) - 1} findings; "
      f"no public-board-fitted fusion applied")
print(f"[pure0949] file size {os.path.getsize(_pub)} bytes")
'''

METADATA = {
    "id": "ppxl16/rsna-knee-apex-pure0949",
    "title": "RSNA knee apex pure0949",
    "code_file": "apex-pure0949.ipynb",
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
    "docker_image": "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f",
    "machine_shape": "NvidiaTeslaT4",
}


def main() -> int:
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    cells = nb["cells"]
    last = cells[-1]
    if last["cell_type"] != "code":
        print(f"REFUSING: last cell is {last['cell_type']}, expected code")
        return 2
    old = "".join(last["source"])
    if "TARGET_WEIGHTS" not in old:
        print("REFUSING: last cell does not look like the Apex fusion cell")
        return 2

    last["source"] = VERIFY_CELL.splitlines(keepends=True)
    last["outputs"] = []
    last["execution_count"] = None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_NB.write_text(json.dumps(nb, ensure_ascii=True, indent=1), encoding="utf-8")
    (OUT_DIR / "kernel-metadata.json").write_text(
        json.dumps(METADATA, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    raw = OUT_NB.read_bytes()
    print(f"wrote {OUT_NB}  ({len(raw)} bytes)")
    print(f"  cells: {len(cells)}  last cell len {len(VERIFY_CELL)}")
    print(f"  ascii-only: {all(c < 128 for c in raw)}")
    try:
        raw.decode("gbk")
        print("  GBK-decodable: YES")
    except UnicodeDecodeError as exc:
        print(f"  GBK-decodable: NO -- {exc}")
        return 1
    json.loads(raw.decode("utf-8"))
    print("  valid ipynb JSON: YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
