"""Does a Repair-v1 CoAt checkpoint exist publicly, or is "4-Way Consensus" aspirational everywhere?

The code configures a 4th CoAt member (`_run_repairv1_reader`, `RV1_MAN_SHA`, `_coat_repairv1.csv`) but
the notebooks that use it mount only mattiaangeli's THREE CoAt datasets, and they carry graceful-failure
events (`coat_repairv1_child_failed`, `coat_repairv1_fallback_studies`). That combination is exactly the
shape of a capability that is declared but not supplied -- the `F_dim=768` pattern.

So: search Kaggle for the checkpoint. If it is public, adding a 4th CoAt member is a real lever; if it is
not, then "4-Way Consensus" is a label nobody is actually running, and our 3-member stack is the truth.
"""
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

QUERIES = ["rsna knee repair", "rsna-knee-repair", "coatnet repairv1", "knee coatnet top3",
           "rsna knee coat", "rsna-knee-coat"]
seen = {}
for q in QUERIES:
    try:
        got = api.dataset_list(search=q) or []
    except Exception as e:  # noqa: BLE001
        print("search %-22r FAILED: %s" % (q, str(e)[:80]))
        continue
    print("search %-22r -> %d" % (q, len(got)))
    for d in got:
        ref = getattr(d, "ref", "?")
        seen.setdefault(ref, (q, str(getattr(d, "title", ""))[:70],
                              str(getattr(d, "subtitle", ""))[:90]))

print()
print("=== all distinct datasets found (%d) ===" % len(seen))
for ref, (q, title, sub) in sorted(seen.items()):
    flag = "  <== REPAIR" if "repair" in (ref + title + sub).lower() else ""
    print("   %-58s | %-46s | %s%s" % (ref[:58], title[:46], sub[:60], flag))

print()
print("=== explicit check: any dataset whose slug or title contains 'repair' ===")
hits = [r for r, (_, t, s) in seen.items() if "repair" in (r + t + s).lower()]
print("   ", hits or "NONE FOUND")

print()
print("=== what the notebooks actually mount for CoAt (from a bench fork) ===")
for ref in ("mattiaangeli/rsna-knee-coat-resgated-ep10-top3",
            "mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2",
            "mattiaangeli/rsna-knee-coatnet-global96-top3"):
    try:
        fl = api.dataset_list_files(ref)
        files = getattr(fl, "files", None) or fl or []
        print("   %-52s %d file(s): %s" % (ref, len(files),
              [getattr(f, "name", "?") for f in files][:4]))
    except Exception as e:  # noqa: BLE001
        print("   %-52s files FAILED %s" % (ref, str(e)[:60]))
