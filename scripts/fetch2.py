import sys, urllib.request, pathlib, time, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
lines = [l.rstrip("\n") for l in open("work/mcl_urls.txt", encoding="utf-8") if l.strip()]
urls = {n: u for n, u in (l.split("\t") for l in lines if "\t" in l)}
out = pathlib.Path("work/mcl_out")
targets = ["_coat_arm_receipt.json","v558_rad_receipt.json","_pipeline_stage.csv",
           "diagnostics/asset_preflight.json","raptor_input_summary.json",
           "resgated_runtime.log","global96_runtime.log","_coat_arm.csv","_raptor.csv"]
for n in targets:
    if n not in urls: print("(absent) %s" % n); continue
    p = out / n.replace("/", "__")
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(urls[n], p); print("got %-40s %7d" % (n, p.stat().st_size)); break
        except Exception as e:
            if attempt == 2: print("FAIL %-40s %s" % (n, str(e)[:80]))
            time.sleep(3)
