"""Read the `sameerk2004/rsna-knee-sil-d2-*` ablation family, plus the two new dataset candidates.

WHY THIS FAMILY. Its names are experiment labels -- `dino-v5`, `tony-dinov3`, `4arm-v55`, `mcl-latmen`,
`meniscus10` -- and two of them (`mcl-latmen`, `meniscus10`) are *our own* closed experiments. Somebody is
running a systematic ablation series in public, updated within the last two hours. That is where a
measured result we do not have would most plausibly live.

`sil-d2` is unexplained, so the markdown is read rather than guessed at.

Also lists files for the two new dataset candidates, because a title is not a listing.
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

from kaggle.api.kaggle_api_extended import KaggleApi

OUT = pathlib.Path("work/newpub_1006")
api = KaggleApi()
api.authenticate()

KERNELS = ["sameerk2004/rsna-knee-sil-d2-dino-v5",
           "sameerk2004/rsna-knee-sil-d2-4arm-v55",
           "sameerk2004/rsna-knee-sil-d2-mcl-latmen",
           "sameerk2004/rsna-knee-sil-d2-meniscus10"]

for ref in KERNELS:
    slug = ref.split("/", 1)[1]
    dst = OUT / slug
    dst.mkdir(parents=True, exist_ok=True)
    if not list(dst.rglob("*.ipynb")):
        try:
            api.kernels_pull(ref, path=str(dst), metadata=True, quiet=True)
        except Exception as e:  # noqa: BLE001
            print("PULL FAILED %-46s %s" % (slug, str(e)[:70]))
            continue
    nbs = list(dst.rglob("*.ipynb"))
    if not nbs:
        print("%s: no notebook" % slug)
        continue
    nb = json.loads(nbs[0].read_text(encoding="utf-8", errors="replace"))
    md = "\n".join(("".join(c.get("source", [])) if isinstance(c.get("source"), list)
                    else (c.get("source") or "")) for c in nb["cells"] if c.get("cell_type") == "markdown")
    code = "\n".join(("".join(c.get("source", [])) if isinstance(c.get("source"), list)
                      else (c.get("source") or "")) for c in nb["cells"] if c.get("cell_type") == "code")
    print("=" * 96)
    print("%s   cells=%d md=%d code=%d" % (slug, len(nb["cells"]), len(md), len(code)))
    import re
    txt = re.sub(r"<[^>]+>", " ", md)
    txt = re.sub(r"\s+", " ", txt).strip()
    print("   MD: %s" % (txt[:900] if txt else "(no markdown)"))
    for pat, label in ((r"sil[-_ ]?d2", "sil-d2"), (r"0\.9[0-4]\d", "score literals"),
                       (r"OWN_W|weight\s*=\s*0\.\d+", "own weight"), (r"meniscus|MCL", "label terms")):
        h = sorted(set(re.findall(pat, code + md, re.I)))[:6]
        if h:
            print("   %-14s %s" % (label, h))
    print()

print("=" * 96)
print("=== new dataset candidates: file listings (a title is not a listing) ===")
for ref in ("barun2104/rsna-knee-mri-processed-3d-volumes",
            "lixin73/rsna-knee-train-checkpoints",
            "alenic/rsna-knee-abnormality-detection-jpeg-224x224"):
    print("-" * 88)
    print(ref)
    try:
        got = api.dataset_list(search=ref.split("/", 1)[1]) or []
        me = [d for d in got if getattr(d, "ref", "") == ref] or got
        d = me[0] if me else None
        if d is not None:
            print("   title   :", str(getattr(d, "title", ""))[:80])
            print("   subtitle:", str(getattr(d, "subtitle", ""))[:110])
    except Exception as e:  # noqa: BLE001
        print("   list failed:", str(e)[:80])
    try:
        fl = api.dataset_list_files(ref)
        files = getattr(fl, "files", None) or fl or []
        print("   %d file(s)" % len(files))
        for f in files[:20]:
            print("      ", getattr(f, "name", "?"))
        if len(files) > 20:
            print("       ... %d more" % (len(files) - 20))
    except Exception as e:  # noqa: BLE001
        print("   files failed:", str(e)[:100])
