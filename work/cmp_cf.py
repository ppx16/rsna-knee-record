import sys, subprocess, pathlib, hashlib, urllib.request, tempfile
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = pathlib.Path(".")
def outs(ref):
    d = pathlib.Path(tempfile.mkdtemp()); u = d/"u.txt"
    subprocess.run([sys.executable, "scripts/kernel_output_urls.py", "--ref", ref, "--out", str(u)], capture_output=True)
    m = {}
    for l in u.read_text(encoding="utf-8", errors="replace").splitlines():
        if "\t" in l:
            n, url = l.split("\t", 1); m[n] = url
    return m
A = outs("ppxl16/rsna-knee-0943cf"); B = outs("ppxl16/rsna-knee-0943cfm")
diff = sorted(set(A) ^ set(B))
print("files only in one side (%d):" % len(diff))
for n in diff[:12]: print("   %s" % n)
print("\n---- sha of key files ----")
for n in ["submission.csv","_pipeline_stage.csv","v558_rad_receipt.json","_coat_arm_receipt.json","_raptor.csv"]:
    row = []
    for tag, M in (("cf", A), ("cfm", B)):
        if n not in M: row.append("%s=absent" % tag); continue
        try:
            b = urllib.request.urlopen(M[n], timeout=40).read()
            row.append("%s=%s(%d)" % (tag, hashlib.sha256(b).hexdigest()[:12], len(b)))
        except Exception as e: row.append("%s=ERR" % tag)
    print("  %-28s %s" % (n, "  ".join(row)))
