import sys, pathlib, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
d = pathlib.Path("work/newds")
for n in ["comparacion_auc_oro.csv","config.json"]:
    p = d / n
    if p.exists(): print("==== %s ====\n%s" % (n, p.read_text(encoding="utf-8",errors="replace")[:700]))
for n in ["A_scratch_pred_dev.csv","B_resnet50_pred_dev.csv"]:
    p = d / n
    if p.exists():
        t = p.read_text(encoding="utf-8",errors="replace").splitlines()
        print("==== %s  (%d lines) ====" % (n, len(t)))
        print("  head:", t[0][:150])
        for l in t[1:3]: print("        ", l[:150])
print("==== build_summary.json ====")
p = d / "build_summary.json"
if p.exists(): print(p.read_text(encoding="utf-8",errors="replace")[:600])
