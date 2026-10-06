"""Read the rest of the 2026-10-06 public arrivals for a new lever.

The sweep found a batch of notebooks posted in the last 48 h. One of them (goodpjw2008's) already gave
us the reader leg. The rest are unread. A lever has to come from NEW evidence, so this is the first
place to look -- and the cheap test is always "what does it change, and is that change measurable",
never "what score does its title claim".
"""
import json
import pathlib
import re

from kaggle.api.kaggle_api_extended import KaggleApi

OUT = pathlib.Path("work/newpub_1006")
OUT.mkdir(parents=True, exist_ok=True)

REFS = [
    "romantamrazov/rsna-knee-dinosaur-v5",
    "matterhorn3838/rsna-knee-v2-velciraptor-dinosaur-speed",
    "qierroo/rsna-knee-public-blend-drop-repair-v1",
    "kiritohayate95/rsna-knee-weak-label",
    "tensorhydra/rsna-knee-d4blend-fork",
    "lnifold/rsna-knee-resnet-v4-physical-fixed",
]

api = KaggleApi()
api.authenticate()
for ref in REFS:
    slug = ref.split("/", 1)[1]
    dst = OUT / slug
    dst.mkdir(parents=True, exist_ok=True)
    try:
        api.kernels_pull(ref, path=str(dst), metadata=True, quiet=True)
    except Exception as e:  # noqa: BLE001
        print("PULL FAILED %-56s %s" % (ref, str(e)[:70]))
        continue

    nbs = sorted(dst.rglob("*.ipynb"))
    if not nbs:
        print("no notebook in %s" % ref)
        continue
    nb = json.loads(nbs[0].read_text(encoding="utf-8", errors="replace"))
    cells = nb.get("cells", [])
    md = "\n".join(("".join(c.get("source", [])) if isinstance(c.get("source"), list)
                    else (c.get("source") or "")) for c in cells if c.get("cell_type") == "markdown")
    code = "\n".join(("".join(c.get("source", [])) if isinstance(c.get("source"), list)
                      else (c.get("source") or "")) for c in cells if c.get("cell_type") == "code")
    print("=" * 100)
    print("%s   cells=%d  md=%d chars  code=%d chars" % (ref, len(cells), len(md), len(code)))
    # the author's own headline, if any
    head = re.sub(r"\s+", " ", md)[:420]
    print("   HEAD: %s" % head)
    # what does it change? look for the load-bearing terms
    for pat, label in [
        (r"OWN_W|own_w|weight\s*=\s*0\.\d+", "own weight"),
        (r"convnext\w*", "convnext"),
        (r"\bMIL\b|attention pool", "MIL"),
        (r"meniscus|MCL|LatMen", "label-specialist terms"),
        (r"drop.?repair|repair", "repair"),
        (r"rank.?mean|rank\(method", "rank blending"),
        (r"0\.94\d", "0.94x claims"),
        (r"/kaggle/input/datasets/[\w\-]+/[\w\-]+", "datasets"),
    ]:
        h = re.findall(pat, code + md)
        uniq = sorted(set(h))[:6]
        if h:
            print("   %-22s x%-4d %s" % (label, len(h), uniq))
    print()
