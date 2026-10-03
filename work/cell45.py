import json, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
nb = json.load(open(r"kernels\kernel-0943s-2coat\rsna-knee-restructured-version-3.ipynb", encoding="utf-8"))
src = "".join(nb["cells"][45]["source"])
print("cell45 chars:", len(src), "lines:", src.count("\n"))
pat = re.compile(r"^\s*(def |class |#\s*=+|#\s*-+|RA?PTOR|.*blend.*|.*alpha.*|.*gold.*|.*write_submission.*|.*priors?\s*=.*)$", re.I)
for i, l in enumerate(src.split("\n")):
    t = l.rstrip()
    if not t.strip(): continue
    if pat.match(t) or re.search(r"blend|gold|alpha|write_submission|oof|AUC|auc", t, re.I):
        print("%5d| %s" % (i, t[:118]))
