import numpy as np, pathlib, sys, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
cands = list(pathlib.Path("work/eff_assets").rglob("s_cache_e13_224_meta.npz"))
print("file:", cands)
if cands:
    z = np.load(cands[0], allow_pickle=True)
    print("keys:", list(z.keys()))
    for k in list(z.keys())[:20]:
        a = z[k]
        try: print("  %-28s shape=%s dtype=%s  %s" % (k, getattr(a,'shape',None), getattr(a,'dtype',None), str(a.ravel()[:6])[:90]))
        except Exception as e: print("  %-28s %s" % (k, e))
for p in pathlib.Path("work/eff_assets").rglob("s0b_audit.json"):
    print("\n== s0b_audit.json"); print(p.read_text(encoding="utf-8")[:900])
