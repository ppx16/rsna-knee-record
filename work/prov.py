import json, sys, pathlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
def cells(p):
    nb = json.loads(pathlib.Path(p).read_text(encoding="utf-8", errors="replace"))
    return ["".join(c.get("source", [])) for c in nb["cells"]]
F = cells("downloads/research/fresh0924/maverickss26_rsna_knee_restructured_version_3/rsna-knee-restructured-version-3.ipynb")
A = cells("kernels/kernel-0943/rsna-knee-restructured-version-3.ipynb")
B = cells("kernels/kernel-0943cf/rsna-knee-restructured-version-3.ipynb")
C = cells("kernels/kernel-0943cfm/rsna-knee-restructured-version-3.ipynb")
print("cells: frontier=%d  0943=%d  0943cf=%d  0943cfm=%d" % (len(F),len(A),len(B),len(C)))
def report(name, X, Y):
    n = min(len(X), len(Y))
    diff = [i for i in range(n) if X[i] != Y[i]]
    print("\n== %s : %d differing cells of %d%s" % (name, len(diff), n, "" if len(X)==len(Y) else "  (+%d cell count)" % (len(Y)-len(X))))
    for i in diff[:8]:
        a, b = X[i], Y[i]
        # shortest distinguishing fragment
        import difflib
        sm = difflib.SequenceMatcher(None, a, b)
        frag = []
        for tag,i1,i2,j1,j2 in sm.get_opcodes():
            if tag != "equal":
                frag.append("%s: %r -> %r" % (tag, a[i1:i2][:70], b[j1:j2][:70]))
        print("   cell %-3d len %d->%d  %s" % (i, len(a), len(b), " | ".join(frag)[:190]))
report("frontier -> kernel-0943", F, A)
report("kernel-0943 -> kernel-0943cf", A, B)
report("kernel-0943cf -> kernel-0943cfm", B, C)
