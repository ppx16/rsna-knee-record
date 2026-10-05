"""Find published knee checkpoints that are candidates for the msft2 recipe.

msft2 worked because its warm start came from the SAME trainer, so its head transferred
245/245 strict=True, and 4 epochs + SWA took it to 0.9069 gold58 -- the best member we own and
the only one that ever moved the rank. So the search target is: a published checkpoint that is
(a) strong, (b) NOT already mounted in our stack, and (c) ideally head-compatible.

This enumerates RSNA-knee datasets, lists their files, and flags anything that looks like model
weights by extension and size. It deliberately does NOT try to download weights.
"""
import json
import pathlib
import time

from kaggle.api.kaggle_api_extended import KaggleApi

WORK = pathlib.Path(r"D:\kaggle\rsna-knee\work")
api = KaggleApi()
api.authenticate()

QUERIES = ["rsna knee", "knee mri", "knee abnormality", "rsna knee weights",
           "rsna knee checkpoint", "knee raptor", "knee coatnet", "knee dinov2"]
WEIGHT_EXT = (".pt", ".pth", ".safetensors", ".ckpt", ".bin", ".npz")

seen, rows = set(), []
for q in QUERIES:
    try:
        ds = api.dataset_list(search=q)
    except Exception as e:
        print(f"search {q!r} failed: {type(e).__name__}: {str(e)[:120]}")
        continue
    for d in ds:
        ref = getattr(d, "ref", None)
        if not ref or ref in seen:
            continue
        title = (getattr(d, "title", "") or "").lower()
        if not any(k in (ref + " " + title).lower() for k in ("knee",)):
            continue
        seen.add(ref)
        rows.append({
            "ref": ref,
            "title": getattr(d, "title", ""),
            "bytes": getattr(d, "total_bytes", 0) or 0,
            "updated": str(getattr(d, "last_updated", "")),
            "votes": getattr(d, "vote_count", None),
            "downloads": getattr(d, "download_count", None),
        })
    time.sleep(0.3)

print(f"knee-named datasets found: {len(rows)}\n")
rows.sort(key=lambda r: -r["bytes"])
print(f"{'MB':>9} {'votes':>6} {'dloads':>7}  ref")
print("-" * 96)
for r in rows:
    print(f"{r['bytes']/1048576:>9.1f} {str(r['votes']):>6} {str(r['downloads']):>7}  {r['ref']}")

# which ones actually ship weights?
print("\n\n=== file listings for the 25 largest (looking for .pt/.safetensors) ===")
for r in rows[:25]:
    try:
        fs = api.dataset_list_files(r["ref"]).files
    except Exception as e:
        print(f"\n{r['ref']}: listing failed ({type(e).__name__})")
        continue
    w = [(f.name, f.total_bytes) for f in fs
         if str(f.name).lower().endswith(WEIGHT_EXT) and (f.total_bytes or 0) > 5_000_000]
    if w:
        print(f"\n{r['ref']}  ({len(w)} weight file(s))")
        for n, b in sorted(w, key=lambda x: -x[1])[:8]:
            print(f"     {b/1048576:>8.1f} MB  {n}")
    time.sleep(0.3)

(WORK / "knee_checkpoint_datasets.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\nwrote {WORK / 'knee_checkpoint_datasets.json'}")
