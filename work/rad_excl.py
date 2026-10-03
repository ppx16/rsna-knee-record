import json, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
nb = json.load(open(r"kernels\kernel-0943s-2coat\rsna-knee-restructured-version-3.ipynb", encoding="utf-8"))
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    for m in re.finditer(r".*(_RAD_EXCLUDE|RAD_EXCLUDE|rad_alpha|coatnet_w\b|_KE_LAB|per.label|PER_LABEL).*", src):
        t = m.group(0).strip()
        if len(t) > 8 and not t.startswith("#"):
            print("[%3d] %s" % (i, t[:150]))
