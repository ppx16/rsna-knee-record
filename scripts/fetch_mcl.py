import sys, urllib.request, pathlib, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
lines = [l.rstrip("\n") for l in open("work/mcl_urls.txt", encoding="utf-8") if l.strip()]
pairs = [l.split("\t") for l in lines if "\t" in l]
want = {}
for name, url in pairs:
    if name.endswith(".log") and "/" not in name: want["log"] = (name, url)
    if name == "submission.csv": want["sub"] = (name, url)
out = pathlib.Path("work/mcl_out"); out.mkdir(parents=True, exist_ok=True)
for k,(n,u) in want.items():
    p = out / n
    try:
        urllib.request.urlretrieve(u, p)
        print("got %-44s %d bytes" % (n, p.stat().st_size))
    except Exception as e:
        print("FAIL %s: %s" % (n, str(e)[:120]))
# acceptance checks
lp = [p for p in out.glob("*.log")]
if lp:
    t = lp[0].read_text(encoding="utf-8", errors="replace")
    print("\n---- ACCEPTANCE ----")
    for pat in [r"DINOv2 fail-closed gate[^\n]*", r"CoAt family members[^\n]*", r"COMPLETE: [^\n]*submission\.csv[^\n]*",
                r"MCL[^\n]*", r"_RAD_EXCLUDE[^\n]*", r"FAILED[^\n]*", r"Traceback[^\n]*"]:
        for m in re.findall(pat, t)[:3]:
            print("  %s" % m.strip()[:160])
