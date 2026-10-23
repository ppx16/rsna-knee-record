"""Pull a set of candidate public kernels and print ONLY the headline claims.

Pulling costs no GPU quota. What matters for us:
  * does the kernel CLAIM a score, and is it a claim or a "Target:"?
  * which datasets does it depend on -- in particular, does it use the OAI-trained
    nartaa checkpoint, or something we could also use?
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "scratch" / "survey"

TARGETS = [
    "evgendvorkin/rsna-versia-7",
    "heliosli/rsna-knee-blend-gold",
    "romantamrazov/rsna-knee-dinosaur-v5",
    "sujanmajhisuzan/rsna-knee-tri-specialist-superstack",
    "nartaa/rsna-knee-0949-anatomical-mirror",
    "prvsiyan/the-bee-s-knees-final-rsna-push",
    "matterhorn3838/rsna-knee-v2-velciraptor-dinosaur-speed",
]


def headline(nb_path: Path) -> str:
    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return f"<unreadable: {exc}>"
    parts = []
    for c in nb["cells"][:3]:
        src = "".join(c["source"])
        if c["cell_type"] == "markdown" or src.lstrip().startswith("#"):
            parts.append(src)
        if sum(len(p) for p in parts) > 1200:
            break
    return "\n".join(parts)[:1200]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for slug in TARGETS:
        safe = slug.replace("/", "__")
        d = OUT / safe
        print("=" * 100)
        print("###", slug)
        if not (d / "kernel-metadata.json").exists():
            r = subprocess.run(
                ["kaggle", "kernels", "pull", slug, "-p", str(d), "-m"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900,
            )
            if r.returncode != 0:
                print("  PULL FAILED:", (r.stderr or "")[:200])
                continue
        try:
            meta = json.loads((d / "kernel-metadata.json").read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print("  metadata unreadable:", exc)
            continue

        print("  votes/file:", meta.get("code_file"))
        print("  datasets:")
        for ds in meta.get("dataset_sources", []):
            mark = "  <== OAI CHECKPOINT" if "nartaa" in ds else ""
            print("     ", ds, mark)
        print("  kernel_sources:", meta.get("kernel_sources", []))
        print("  --- headline ---")
        hp = d / str(meta.get("code_file", ""))
        text = headline(hp) if hp.is_file() else "<code_file missing>"
        for line in text.splitlines():
            if line.strip():
                print("   |", line.rstrip()[:150])
        # every distinct 0.9xx claim in the first 4 cells
        claims = set()
        try:
            nb = json.loads(hp.read_text(encoding="utf-8"))
            for c in nb["cells"][:4]:
                for m in re.finditer(r"0\.9[0-9]{2}", "".join(c["source"])):
                    claims.add(m.group(0))
        except Exception:  # noqa: BLE001
            pass
        print("  scores mentioned in first cells:", sorted(claims))


if __name__ == "__main__":
    main()
