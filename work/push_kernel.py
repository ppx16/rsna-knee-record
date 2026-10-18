"""Push a kernel directory to Kaggle, verifying the push actually happened.

`kaggle kernels push` returns rc=0 even when it fails (A28/A32), so the ONLY
reliable verdict is the literal string "successfully pushed" on stdout.

Usage:  python work/push_kernel.py kernels/kernel-apex-repro
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Guards against the "pusher built by string replacement pushed the wrong kernel" bug:
# the directory basename and the metadata id must be declared together, and the
# metadata on disk must agree.
EXPECTED = {
    "kernel-0958-s6-flat": "ppxl16/rsna-knee-0958-s6-flat",
    "kernel-0958-s6-s6w60": "ppxl16/rsna-knee-0958-s6-s6w60",
    "kernel-0958-s6-s6w40": "ppxl16/rsna-knee-0958-s6-s6w40",
    "kernel-0958-r9all-w10": "ppxl16/rsna-knee-0958-r9all-w10",
    "kernel-0958-blend-a50": "ppxl16/rsna-knee-0958-blend-a50",
    "kernel-0958-blend-a65": "ppxl16/rsna-knee-0958-blend-a65",
    "kernel-0958-band32-w08": "ppxl16/rsna-knee-0958-band32-w08",
    "kernel-0958-band32-w15": "ppxl16/rsna-knee-0958-band32-w15",
    "kernel-0958-blend": "ppxl16/rsna-knee-0958-blend",
    "kernel-09573-band32-w04": "ppxl16/rsna-knee-09573-band32-w04",
    "kernel-09573-band32-w15": "ppxl16/rsna-knee-09573-band32-w15",
    "kernel-09576-r7": "ppxl16/rsna-knee-09576-r7",
    "kernel-09576-verbatim": "ppxl16/rsna-knee-09576-verbatim",
    "kernel-09573-legs2-w08": "ppxl16/rsna-knee-09573-legs2-w08",
    "kernel-09573-r7ds-a80": "ppxl16/rsna-knee-09573-r7ds-a80",
    "kernel-09573-band32-w08": "ppxl16/rsna-knee-09573-band32-w08",
    "kernel-09573-r9all-w10": "ppxl16/rsna-knee-09573-r9all-w10",
    "kernel-09573-r7ds": "ppxl16/rsna-knee-09573-r7ds",
    "kernel-09573-r7ds-a50": "ppxl16/rsna-knee-09573-r7ds-a50",
    "kernel-09573-r7ds-a65": "ppxl16/rsna-knee-09573-r7ds-a65",
    "kernel-09573-one-click": "ppxl16/rsna-knee-09573-one-click",
    "kernel-r7-c224-ours50": "ppxl16/rsna-knee-r7-c224-ours50",
    "kernel-r7-c224-ours30": "ppxl16/rsna-knee-r7-c224-ours30",
    "kernel-r7-c224-band32-w15": "ppxl16/rsna-knee-r7-c224-band32-w15",
    "kernel-r7-c224-band32-w30": "ppxl16/rsna-knee-r7-c224-band32-w30",
    "kernel-r7-c224-band32-w08": "ppxl16/rsna-knee-r7-c224-band32-w08",
    "kernel-r7-c224-r9all-w45": "ppxl16/rsna-knee-r7-c224-r9all-w45",
    "kernel-r7-c224-r9all-w25": "ppxl16/rsna-knee-r7-c224-r9all-w25",
    "kernel-r7-c224-r9all-w10": "ppxl16/rsna-knee-r7-c224-r9all-w10",
    "kernel-r7-c224-w60": "ppxl16/rsna-knee-r7-c224-w60",
    "kernel-r7-c224-w40": "ppxl16/rsna-knee-r7-c224-w40",
    "kernel-r7-c224-w70": "ppxl16/rsna-knee-r7-c224-w70",
    "kernel-r7-c224w02": "ppxl16/rsna-knee-r7-c224-w02",
    "kernel-r7-c224w05": "ppxl16/rsna-knee-r7-c224-w05",
    "kernel-r7-c224leg": "ppxl16/rsna-knee-r7-c224leg",
    "kernel-r7-r224arm": "ppxl16/rsna-knee-r7-r224arm",
    "kernel-r7-onclick": "ppxl16/rsna-knee-r7-onclick",
    "kernel-raptor-s6-v1ds": "ppxl16/rsna-knee-raptor-s6-v1ds",
    "kernel-raptor-s6-fork": "ppxl16/rsna-knee-raptor-s6-fork",
    "kernel-arc-duck18-ve": "ppxl16/arc-duck18-ve",
    "kernel-arc-duck18-vd": "ppxl16/arc-duck18-vd",
    "kernel-arc-duck18-vb": "ppxl16/arc-duck18-vb",
    "kernel-arc-pth-order": "ppxl16/arc-pth-order",
    "kernel-arc-pth-probe": "ppxl16/arc-pth-probe",
    "kernel-arc-duck18-va": "ppxl16/arc-duck18-va",
    "kernel-arc-vllm-probe": "ppxl16/arc-vllm-probe",
    "kernel-arc-duck": "ppxl16/arc-duck-taaf",
    "kernel-blendgold-restored": "ppxl16/rsna-knee-blendgold-restored",
    "kernel-fitted-0954": "ppxl16/rsna-knee-fitted-0954",
    "kernel-triplereader-0951": "ppxl16/rsna-knee-triple-reader-0951",
    "kernel-blendgold-0951": "ppxl16/rsna-knee-blendgold-0951",
    "kernel-apex-repro": "ppxl16/rsna-knee-apex-0949",
    "kernel-apex-pure0949": "ppxl16/rsna-knee-apex-pure0949",
}


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: push_kernel.py <kernel-dir>")
        return 2
    d = Path(argv[1])
    if not d.is_absolute():
        d = ROOT / d
    d = d.resolve()
    name = d.name

    if name not in EXPECTED:
        print(f"REFUSING: {name!r} is not in the EXPECTED allow-list {sorted(EXPECTED)}")
        return 2
    want_id = EXPECTED[name]

    meta_path = d / "kernel-metadata.json"
    if not meta_path.is_file():
        print(f"REFUSING: missing {meta_path}")
        return 2
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    got_id = meta.get("id")
    if got_id != want_id:
        print(f"REFUSING: {meta_path} has id={got_id!r}, expected {want_id!r}")
        return 2

    code_file = d / meta["code_file"]
    if not code_file.is_file():
        print(f"REFUSING: code_file missing: {code_file}")
        return 2

    # kernels_push reads the code_file as GBK; make sure that cannot explode.
    try:
        code_file.read_bytes().decode("gbk")
    except UnicodeDecodeError as exc:
        print(f"REFUSING: {code_file.name} is not GBK-decodable: {exc}")
        return 2

    print(f"dir   : {d}")
    print(f"id    : {got_id}")
    print(f"code  : {meta['code_file']} ({code_file.stat().st_size} bytes)")
    print(f"gpu   : {meta.get('enable_gpu')}  tpu: {meta.get('enable_tpu')}  "
          f"shape: {meta.get('machine_shape')}")
    print(f"data  : {meta.get('dataset_sources')}")
    print("-" * 70)

    r = subprocess.run(
        ["kaggle", "kernels", "push", "-p", str(d)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800,
    )
    out = (r.stdout or "") + (r.stderr or "")
    print(out.strip()[:2000])

    m = re.search(r"Kernel version (\d+) successfully pushed", out)
    if not m:
        m = re.search(r"successfully pushed", out)
    if not m:
        print(f"\nPUSH FAILED (rc={r.returncode}) -- no 'successfully pushed' in output")
        return 1

    ver = m.group(1) if m.lastindex else "?"

    # A183: Kaggle derives the REAL slug from the TITLE and only warns when the id in
    # metadata does not resolve to it. Never report the id we *asked for* -- report the
    # one Kaggle actually used, or our bookkeeping will name a kernel that does not exist.
    actual = got_id
    ref = re.search(r"kaggle\.com/code/([A-Za-z0-9_\-]+/[A-Za-z0-9_\-]+)", out)
    if ref:
        actual = ref.group(1)
    if actual != got_id:
        print(f"\n!! SLUG MISMATCH: metadata id {got_id!r} but Kaggle created/updated {actual!r}")
        print("   The id must be the slug of the title. Fix kernel-metadata.json, then reuse this id.")
    print(f"\nPUSH OK: {actual} version {ver}")
    print(f"record it:  python scripts/director.py --record-push {actual}:{ver}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
