"""Final pre-push verification of every kernel we intend to push.

Checks the things that have actually bitten us:
  * A168 trap (3): `machine_shape` OVERRIDES enable_gpu/enable_tpu, so an inherited
    TPU shape silently turns a "GPU" push into a TPU session. Refuse any Tpu* shape.
  * A180: kernels_push reads code_file as GBK; refuse anything not GBK-decodable.
  * stray empty dataset slugs (the pulled heliosli metadata had one).
  * the code_file actually is the notebook we think it is (marker strings).
  * enable_gpu True / enable_tpu False / competition attached.

Usage: python work/verify_kernels.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# dir name -> (kernel id, marker string that must appear in the notebook)
EXPECTED = {
    "kernel-blendgold-restored": ("ppxl16/rsna-knee-blendgold-restored", "weights fixed before any run"),
    "kernel-fitted-0954": ("ppxl16/rsna-knee-fitted-0954", '"Lateral OA": 0.50'),
    "kernel-blendgold-0951": ("ppxl16/rsna-knee-blendgold-0951", "W_RAPTOR, W_OURS = 0.7, 0.3"),
    "kernel-triplereader-0951": ("ppxl16/rsna-knee-triple-reader-0951", "No fallback on purpose"),
    "kernel-apex-repro": ("ppxl16/rsna-knee-apex-0949", "TARGET_WEIGHTS"),
    "kernel-apex-pure0949": ("ppxl16/rsna-knee-apex-pure0949", "no public-board-fitted fusion applied"),
}
DOCKER = ("gcr.io/kaggle-private-byod/python@sha256:"
          "2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f")
COMP = "rsna-knee-abnormality-detection"

fails: list[str] = []
warns: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"    ok   {msg}")
    else:
        print(f"    FAIL {msg}")
        fails.append(msg)


def main() -> int:
    for name, (want_id, marker) in EXPECTED.items():
        d = ROOT / "kernels" / name
        print("=" * 88)
        print(f"### {name}")
        if not d.is_dir():
            check(False, f"{name}: directory missing")
            continue

        mp = d / "kernel-metadata.json"
        check(mp.is_file(), "metadata exists")
        if not mp.is_file():
            continue
        try:
            m = json.loads(mp.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            check(False, f"metadata parses ({exc})")
            continue
        check(True, "metadata parses as JSON")

        check(m.get("id") == want_id, f"id == {want_id} (got {m.get('id')!r})")
        check(m.get("enable_gpu") is True, "enable_gpu is True")
        check(m.get("enable_tpu") is False, "enable_tpu is False")

        shape = m.get("machine_shape", "")
        check("Tpu" not in shape and "TPU" not in shape,
              f"machine_shape is not a TPU shape (got {shape!r}) -- A168 trap 3")
        check(shape == "NvidiaTeslaT4", f"machine_shape pinned to NvidiaTeslaT4 (got {shape!r})")
        check(m.get("docker_image") == DOCKER, "docker_image matches the scored container")
        check(COMP in m.get("competition_sources", []), f"competition {COMP} attached")

        ds = m.get("dataset_sources", [])
        check(bool(ds) and all(isinstance(x, str) and x.strip() for x in ds),
              f"dataset_sources all non-empty ({len(ds)} entries)")
        for x in ds:
            if not (isinstance(x, str) and x.strip()):
                check(False, f"stray empty dataset slug in {name}")
        check(not m.get("kernel_sources"), "no kernel_sources (self-contained)")

        cf = d / m.get("code_file", "")
        check(cf.is_file(), f"code_file present: {m.get('code_file')}")
        if not cf.is_file():
            continue
        raw = cf.read_bytes()
        try:
            raw.decode("gbk")
            check(True, f"GBK-decodable ({len(raw)} bytes) -- A180")
        except UnicodeDecodeError as exc:
            check(False, f"GBK-decodable -- A180 ({exc})")

        try:
            nb = json.loads(raw.decode("utf-8"))
            ccells = nb.get("cells", [])
            check(len(ccells) > 0, f"valid ipynb with {len(ccells)} cells")
        except Exception as exc:  # noqa: BLE001
            check(False, f"valid ipynb JSON ({exc})")
            continue

        text = "\n".join("".join(c.get("source", [])) for c in ccells)
        check(marker in text, f"contains expected marker: {marker!r}")
        if "RUN_INFERENCE = True" in text:
            check(True, "RUN_INFERENCE = True present")
        # anything that would abort immediately
        if "raise SystemExit" in text and "RUN_INFERENCE" in text:
            warns.append(f"{name}: contains the static-notebook SystemExit guard (expected if RUN_INFERENCE=True)")

    print("=" * 88)
    for w in warns:
        print("  WARN", w)
    if fails:
        print(f"\n{len(fails)} CHECK(S) FAILED -- DO NOT PUSH")
        return 1
    print(f"\nALL CHECKS PASSED for {len(EXPECTED)} kernels -- safe to push")
    return 0


if __name__ == "__main__":
    sys.exit(main())
