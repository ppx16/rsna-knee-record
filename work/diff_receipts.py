"""Did the MCL edit actually reach the graph?

The visible submission.csv is byte-identical to the parent's, and the two
`_coat_arm_receipt.json` files hash differently -- but `elapsed_seconds` differs too
(37.91 vs 38.66), and a TIMING change alone would change the hash. So "receipts differ"
is not evidence by itself.

This walks both receipts field by field and reports exactly WHICH fields differ, so the
timing confound is separated from a genuine content change. Per A80/A143 the receipts are
the primary signal -- but only if the differing field is content, not a stopwatch.
"""
import json
import pathlib

PARENT = pathlib.Path(r"D:\kaggle\rsna-knee\work\rcpt_parent\_coat_arm_receipt.json")
THIS = pathlib.Path(r"D:\kaggle\rsna-knee\work\cfmsft2m-out\_coat_arm_receipt.json")

TIMING_HINTS = ("elapsed", "seconds", "time", "duration", "perf", "rss", "memory", "alloc")


def flatten(o, prefix=""):
    out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(flatten(v, f"{prefix}{k}."))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            out.update(flatten(v, f"{prefix}{i}."))
    else:
        out[prefix[:-1]] = o
    return out


a = flatten(json.loads(PARENT.read_text(encoding="utf-8")))
b = flatten(json.loads(THIS.read_text(encoding="utf-8")))

print(f"fields: parent={len(a)}  this={len(b)}")
only_a = sorted(set(a) - set(b))
only_b = sorted(set(b) - set(a))
if only_a:
    print("only in parent:", only_a[:20])
if only_b:
    print("only in this  :", only_b[:20])

diffs = [(k, a[k], b[k]) for k in sorted(set(a) & set(b)) if a[k] != b[k]]
print(f"\ndiffering fields: {len(diffs)}\n")
content, timing = [], []
for k, va, vb in diffs:
    (timing if any(h in k.lower() for h in TIMING_HINTS) else content).append((k, va, vb))

print("--- TIMING / RESOURCE fields (expected to differ run to run) ---")
for k, va, vb in timing:
    print(f"  {k}\n      parent={str(va)[:110]}\n      this  ={str(vb)[:110]}")

print("\n--- CONTENT fields (a difference here means the edit reached the graph) ---")
if not content:
    print("  NONE.")
    print("  => every differing field is timing/resource, so the receipts differing does NOT")
    print("     show that the MCL edit applied. The identical submission.csv is then")
    print("     UNEXPLAINED but also unevidenced either way.")
else:
    for k, va, vb in content:
        print(f"  *** {k}\n      parent={str(va)[:160]}\n      this  ={str(vb)[:160]}")
