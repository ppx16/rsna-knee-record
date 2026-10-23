import json, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
nb = json.load(open(r"kernels\kernel-0943s-2coat\rsna-knee-restructured-version-3.ipynb", encoding="utf-8"))
print("cells:", len(nb["cells"]))
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    first = next((l for l in src.split("\n") if l.strip()), "")
    if c["cell_type"] == "markdown":
        print("  [%3d] MD   %s" % (i, first[:96]))
    else:
        print("  [%3d] code %4d ch | %s" % (i, len(src), first[:88]))
