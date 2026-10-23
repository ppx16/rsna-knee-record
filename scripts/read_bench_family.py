"""Read the `chenchinglong/rsna-kad-bench-*` family: someone else's systematic benchmark of the public variants.

This is the free sweep we never ran. Ten notebooks, all named `rsna-kad-bench-<variant>`, updated across
2026-10-05/06 -- i.e. one person comparing the public 0.942-class approaches against each other. If their
table says which variant wins, that is evidence we did not have to buy with GPU hours.

A "bench" notebook that only re-runs someone else's pipeline is worth exactly its comparison table, so
that is what this looks for.
"""

# --- GBK CONSOLE GUARD (see work/patch_gbk_guard.py) -------------------------------------------------
# This console is cp936, which cannot encode the U+26A0 warning sign used below. An unguarded
# `print("... \u26a0 ...")` raises UnicodeEncodeError at that line and kills the script MID-REPORT, while
# everything printed earlier is already on screen -- so the run looks complete. Self-contained on purpose:
# it imports `sys` itself and depends on nothing else in this tree.
import sys as _sys

try:
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001  -- non-reconfigurable stream: leave it alone, prints still degrade
    pass
# ----------------------------------------------------------------------------------------------------
import json
import pathlib
import re

from kaggle.api.kaggle_api_extended import KaggleApi

OUT = pathlib.Path("work/newpub_1006")
api = KaggleApi()
api.authenticate()

# discover the family rather than guessing slugs
try:
    ks = api.kernels_list(user="chenchinglong", sort_by="dateRun", page_size=40) or []
except Exception as e:  # noqa: BLE001
    print("list failed:", str(e)[:120])
    ks = []
fam = [k.ref for k in ks if "kad-bench" in getattr(k, "ref", "")]
print("found %d bench notebooks:" % len(fam))
for f in fam:
    print("   ", f)

for ref in fam:
    slug = ref.split("/", 1)[1]
    dst = OUT / slug
    dst.mkdir(parents=True, exist_ok=True)
    if not list(dst.rglob("*.ipynb")):
        try:
            api.kernels_pull(ref, path=str(dst), metadata=True, quiet=True)
        except Exception as e:  # noqa: BLE001
            print("  pull failed %s: %s" % (slug, str(e)[:70]))
            continue
    nb = json.loads(list(dst.rglob("*.ipynb"))[0].read_text(encoding="utf-8", errors="replace"))
    md = "\n".join(("".join(c.get("source", [])) if isinstance(c.get("source"), list)
                    else (c.get("source") or "")) for c in nb["cells"] if c.get("cell_type") == "markdown")
    code = "\n".join(("".join(c.get("source", [])) if isinstance(c.get("source"), list)
                      else (c.get("source") or "")) for c in nb["cells"] if c.get("cell_type") == "code")
    print("=" * 96)
    print("%s  md=%d code=%d" % (slug, len(md), len(code)))
    # any comparison table / score claims
    tab = re.findall(r"^\|.*\|$", md, re.M)[:14]
    for t in tab:
        print("   " + t[:150])
    for m in set(re.findall(r"0\.9[0-4]\d", md + code)):
        pass
    claims = sorted(set(re.findall(r"0\.9[0-4]\d", md)))
    print("   score literals in md:", claims)
    head = re.sub(r"\s+", " ", md)[:260]
    if head.strip():
        print("   HEAD: %s" % head)
