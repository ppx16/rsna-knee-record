"""A/B the cfmsft2flat run against the cfmsft2 run, from their OWN logs.

What must be true for the arm to be what it claims:
  1. preset differs:            cfmsft2 = speedy   vs   cfmsft2flat = parent
  2. the per-finding outer weight table differs: cfmsft2 NON-uniform vs cfmsft2flat all 0.650
  3. both produce a submission, and the SHAs DIFFER (proving the one-axis edit reached the output)
  4. the msft2 member is present in both (otherwise the arm is not "cfmsft2 + flattening")
"""
import json
import pathlib
import re
import shutil
import time

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

CACHE = pathlib.Path("work/klog_ab")
CACHE.mkdir(parents=True, exist_ok=True)


def get_log(ref: str, slug: str) -> str:
    dst = CACHE / ("%s.txt" % slug)
    if dst.is_file() and dst.stat().st_size > 1000:
        return dst.read_text(encoding="utf-8", errors="replace")
    for attempt in range(4):
        try:
            api.kernels_output(ref, path=str(CACHE / slug), file_pattern=("%s.log" % slug),
                               force=True, quiet=True)
            break
        except Exception as e:  # noqa: BLE001
            print("   %s attempt %d: %s" % (slug, attempt, type(e).__name__))
            time.sleep(5)
    hits = sorted((CACHE / slug).rglob("*.log"))
    if not hits:
        return ""
    raw = hits[0].read_text(encoding="utf-8", errors="replace")
    out = []
    for ln in raw.splitlines():
        ln = ln.lstrip(",").strip()
        if not ln:
            continue
        try:
            d = json.loads(ln).get("data")
        except Exception:
            d = None
        if d:
            out.append(d)
    txt = "".join(out)
    dst.write_text(txt, encoding="utf-8")
    return txt


PAIRS = [("ppxl16/rsna-knee-0943cfmsft2", "rsna-knee-0943cfmsft2"),
         ("ppxl16/rsna-knee-0943cfmsft2flat", "rsna-knee-0943cfmsft2flat")]

texts = {}
for ref, slug in PAIRS:
    t = get_log(ref, slug)
    texts[slug] = t
    print("%-28s log chars: %d" % (slug, len(t)))

print()
for slug, t in texts.items():
    preset = re.search(r"\[config\] preset=(\w+)", t)
    flat = re.search(r"outer CoAtNet weight per finding: (flat [\d.]+)", t)
    sha = re.search(r"submission\.csv, \d+ studies, SHA256=([0-9a-f]{16})", t)
    members = re.search(r"weights package: (\d+) member\(s\)", t)
    gate = "DINOv2 fail-closed gate PASS" in t
    print("%-28s preset=%-7s | %-22s | members=%-3s gate=%-5s | sha=%s"
          % (slug, preset.group(1) if preset else "?",
             flat.group(1) if flat else "(none printed)",
             members.group(1) if members else "?",
             gate, sha.group(1) if sha else "?"))

print()
print("=== per-finding outer CoAtNet w, side by side ===")
rows = {}
for slug, t in texts.items():
    d = {}
    for m in re.finditer(r"^(\S[^\n]*?)\s+(\d\.\d{3})\s+(-?\d\.\d{3})\s+(\d+)$", t, re.M):
        name = m.group(1).strip()
        if name.lower() in ("finding",) or "outer" in name:
            continue
        d[name] = m.group(2)
    rows[slug] = d
allnames = sorted(set().union(*[set(d) for d in rows.values()])) if rows else []
for n in allnames:
    a = rows["rsna-knee-0943cfmsft2"].get(n, "-")
    b = rows["rsna-knee-0943cfmsft2flat"].get(n, "-")
    print("   %-20s cfmsft2=%-6s cfmsft2flat=%-6s %s" % (n, a, b, "DIFFERS" if a != b else "same"))
