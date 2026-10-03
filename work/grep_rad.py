import json, re, sys, pathlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
for d in ["kernel-0943cf","kernel-0943f","kernel-0943"]:
    p = pathlib.Path("kernels")/d
    nbs = list(p.glob("*.ipynb"))
    if not nbs: print(d, "-> no notebook"); continue
    nb = json.loads(nbs[0].read_text(encoding="utf-8", errors="replace"))
    print("== %s  (%s)" % (d, nbs[0].name))
    for i,c in enumerate(nb["cells"]):
        t = "".join(c.get("source",[]))
        for m in re.finditer(r'.*_RAD_EXCLUDE\s*=.*', t):
            print("   [%3d] %s" % (i, m.group(0).strip()[:110]))
