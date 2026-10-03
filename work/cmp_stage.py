import sys, urllib.request, pathlib, hashlib, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
lines = [l.rstrip("\n") for l in open("work/eff_urls.txt", encoding="utf-8") if l.strip()]
urls = {n: u for n, u in (l.split("\t") for l in lines if "\t" in l)}
out = pathlib.Path("work/eff_out"); out.mkdir(parents=True, exist_ok=True)
for n in ["_pipeline_stage.csv","submission.csv","_coat_arm_receipt.json","raptor_input_summary.json","v558_rad_receipt.json"]:
    if n not in urls: print("(absent) %s" % n); continue
    p = out / n.replace("/", "__")
    for a in range(3):
        try: urllib.request.urlretrieve(urls[n], p); break
        except Exception as e:
            if a==2: print("FAIL %s %s" % (n, str(e)[:70]))
            time.sleep(3)
def sig(p):
    b = p.read_bytes(); return len(b), hashlib.sha256(b).hexdigest()[:16]
a = pathlib.Path("work/mcl_out/_pipeline_stage.csv"); b = out / "_pipeline_stage.csv"
print("\n---- _pipeline_stage.csv ----")
if a.exists() and b.exists():
    print("0943cfm   (no leg): %d bytes  sha %s" % sig(a))
    print("0943cfm2eff(leg) : %d bytes  sha %s" % sig(b))
    print("IDENTICAL -> leg was ABANDONED" if a.read_bytes()==b.read_bytes() else "DIFFERENT -> leg APPLIED")
else:
    print("missing: a=%s b=%s" % (a.exists(), b.exists()))
for f in ["submission.csv"]:
    p = out / f
    if p.exists(): print("\n%s: %d bytes sha %s" % (f, *sig(p)))
