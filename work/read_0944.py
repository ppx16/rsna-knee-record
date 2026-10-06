"""Read the 0.944 notebook: what does it actually do, and is any of it reproducible by us?

284 KB of ipynb. Dump the markdown (the author's own explanation) first, then the structural facts:
datasets mounted, model families, blend composition, and any self-trained component.
"""
import json
import pathlib
import re

NB = pathlib.Path("work/newpub_1006/rsna-knee-stack-2-5d-convnext-mil-lb-0-944/"
                  "rsna-knee-stack-2-5d-convnext-mil-lb-0-944.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8", errors="replace"))
cells = nb["cells"]
print("cells: %d" % len(cells))

md = []
for c in cells:
    if c.get("cell_type") == "markdown":
        s = c.get("source")
        md.append("".join(s) if isinstance(s, list) else (s or ""))
text_md = "\n\n".join(md)
print("markdown chars: %d" % len(text_md))
print("=" * 100)
print(text_md[:7000])
print("=" * 100)

code = "".join(
    ("".join(c.get("source", [])) if isinstance(c.get("source"), list) else (c.get("source") or ""))
    for c in cells if c.get("cell_type") == "code")
print("\nCODE STRUCTURAL FACTS")
for pat, label in [
    (r"convnext\w*", "convnext mentions"),
    (r"\bMIL\b|attention.?pool|gated.?attention", "MIL / attention pooling"),
    (r"mattiaangeli|pilkwang|prvsiyan|dreaddevelopment|evgendvorkin|jiweiliu|tonylica|pjmathematician",
     "known authors referenced"),
    (r"/kaggle/input/[A-Za-z0-9_\-/]+", "input paths"),
    (r"0\.94\d\b", "0.94x literals"),
    (r"rank.?mean|rankdata|rank\(", "rank blending"),
]:
    hits = sorted(set(re.findall(pat, code)))[:14]
    print("   %-28s %d  %s" % (label, len(re.findall(pat, code)), hits))
