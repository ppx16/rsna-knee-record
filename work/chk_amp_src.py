import json, re, sys, pathlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
cands = [
  ("FRONTIER SOURCE", pathlib.Path("downloads/research/fresh0924/maverickss26_rsna_knee_restructured_version_3/rsna-knee-restructured-version-3.ipynb")),
  ("OUR kernel-0943   ", pathlib.Path("kernels/kernel-0943/rsna-knee-restructured-version-3.ipynb")),
  ("OUR 0943cfm       ", pathlib.Path("kernels/kernel-0943cfm/rsna-knee-restructured-version-3.ipynb")),
]
for label, p in cands:
    if not p.exists(): print("%s : MISSING %s" % (label, p)); continue
    nb = json.loads(p.read_text(encoding="utf-8", errors="replace"))
    hits = []
    for i, c in enumerate(nb["cells"]):
        t = "".join(c.get("source", []))
        for m in re.finditer(r".*AMP_PREF\s*=.*|.*torch\.bfloat16.*|.*is_bf16_supported.*", t):
            hits.append((i, m.group(0).strip()[:120]))
    print("== %s cells=%d  hits=%d" % (label, len(nb["cells"]), len(hits)))
    for i, h in hits[:8]: print("   [%3d] %s" % (i, h))
