import sys, json, pathlib, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
out = pathlib.Path("work/mcl_out")
for n in ["_coat_arm_receipt.json","v558_rad_receipt.json","diagnostics__asset_preflight.json","raptor_input_summary.json"]:
    p = out / n
    if not p.exists(): continue
    print("==== %s" % n)
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        s = json.dumps(d, ensure_ascii=False)
        print("   " + s[:700])
    except Exception as e:
        print("   (not json) " + p.read_text(encoding="utf-8", errors="replace")[:400])
print("\n==== 子进程日志里的网关/成员证据 ====")
for n in ["resgated_runtime.log","global96_runtime.log"]:
    p = out / n
    if not p.exists(): continue
    t = p.read_text(encoding="utf-8", errors="replace")
    for pat in [r"gate[^\n]*", r"fail-closed[^\n]*", r"20/20[^\n]*", r"members[^\n]*", r"Error[^\n]*", r"Traceback[^\n]*"]:
        for m in re.findall(pat, t, re.I)[:3]:
            print("  [%s] %s" % (n[:12], m.strip()[:140]))
