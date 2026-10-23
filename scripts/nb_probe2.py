"""Compact probe of a pulled notebook: what code/weights does it need?"""
from __future__ import annotations

import json
import re
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PATS = [
    (r"/kaggle/input/[^\s\"')]{0,70}", "input paths"),
    (r"pip install[^\n]{0,90}", "installs"),
    (r"\b\w*\.(?:pt|pth|safetensors|ckpt)\b", "weight files"),
    (r"(?i)\b(dataset_sources|kernel_sources)\b[^\n]{0,110}", "attached sources"),
    (r"(?i)\b(0\.9[0-9]{2,3})\b", "score-ish literals"),
    (r"(?i)\b(dinov2|dinov3|d4|coatnet|raptor|maxspan|radimagenet)\b", "architectures"),
]


def main() -> int:
    for p in sys.argv[1:]:
        nb = json.loads(open(p, encoding="utf-8").read())
        txt = "\n".join("".join(c.get("source", [])) for c in nb.get("cells", []))
        print("=" * 84)
        print("%s | cells %d | chars %d" % (p.split("/")[-2], len(nb.get("cells", [])), len(txt)))
        for pat, label in PATS:
            seen: list[str] = []
            for m in re.findall(pat, txt):
                m = (m if isinstance(m, str) else " ".join(m)).strip()
                if m and m not in seen:
                    seen.append(m)
                if len(seen) >= 8:
                    break
            print("  -- %-18s %s" % (label, seen if seen else "(none)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
