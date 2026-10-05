"""Render a Kaggle kernel log (JSON-lines of {stream_name,time,data}) into plain readable text."""
import json
import pathlib
import sys

src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                   "work/klog_flat/rsna-knee-0943cfmsft2flat.log")
for line in src.read_text(encoding="utf-8", errors="replace").splitlines():
    line = line.lstrip(",").strip()
    if not line:
        continue
    try:
        rec = json.loads(line)
    except Exception:
        print(line)
        continue
    d = rec.get("data")
    if d is None:
        continue
    sys.stdout.write(d if d.endswith("\n") else d + "\n")
