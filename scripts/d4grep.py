import json, re, sys, pathlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
nb = json.loads(pathlib.Path("work/d4blend/rsna-knee-d4-blend.ipynb").read_text(encoding="utf-8", errors="replace"))
txt = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
print("cells:", len(nb["cells"]), " chars:", len(txt))
public = {"dreaddevelopment","mattiaangeli","marwanmath","antoinegg1","pilkwang","prvsiyan",
          "sofiaanjenje","metaresearch","arc-prize-2026","rsna-knee-abnormality-detection"}
paths = sorted({m.group(0) for m in re.finditer(r"/kaggle/input/[A-Za-z0-9_\-./]+", txt)})
print("\n---- /kaggle/input paths referenced ----")
for p in paths:
    seg = p.split("/")[3] if len(p.split("/")) > 3 else ""
    flag = "" if seg in public or seg in ("datasets","models","competitions") else "   <-- NOT a public source"
    print("   %-72s%s" % (p[:72], flag))
print("\n---- mentions of private-ish things ----")
for kw in ["private","Private","ours10","convnext","ConvNeXt","repair","Repair"]:
    n = len(re.findall(kw, txt))
    if n: print("   %-12s %d" % (kw, n))
