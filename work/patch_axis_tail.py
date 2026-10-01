"""Rewrite the reporting tail of the exp085 axis kernel for the VARIANTS grid."""
import json
import pathlib
import py_compile
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

p = pathlib.Path(__file__).resolve().parents[1] / "kernels" / "kernel-exp085-axis" / "exp085-axis.py"
s = p.read_text(encoding="utf-8")
i = s.find("    from sklearn.metrics import roc_auc_score")
if i < 0:
    sys.exit("!! marker not found")
head = s[:i]

TAIL = '''    from sklearn.metrics import roc_auc_score

    def auc_of(pr):
        per = [roc_auc_score(Y[:, j], pr[:, j]) for j in range(len(LAB))
               if len(np.unique(Y[:, j])) > 1]
        return float(np.mean(per)), per

    print("\\n" + "=" * 96)
    print("%-7s %s   (author exp085 gold58: 0.9062;  A167 member gate: solo >= 0.90)"
          % ("crop", "  ".join("%-18s" % v[0] for v in VARIANTS)))
    print("-" * 96)
    best = None
    for c in CROPS:
        cells = []
        for v in VARIANTS:
            a, _ = auc_of(np.asarray(P[(c, v[0])]))
            cells.append("%-18.4f" % a)
            if best is None or a > best[0]:
                best = (a, c, v[0])
        print("%-7d %s" % (c, "  ".join(cells)), flush=True)
    print("-" * 96)
    print("BEST: crop=%d variant=%s  macro-AUC=%.4f" % (best[1], best[2], best[0]))
    print("base MUST reproduce run 3's grp_mean_local at crop 272 = 0.8191.")
    print("A big jump on `mirror` means the corpus depth axis is the other way round.")
    print("A big jump on `pct_clip` means the intensity/window convention is the gap.")
    print("No jump on any of them means exp085 needs the author's own preprocessing.")
    json.dump({"variants": [v[0] for v in VARIANTS], "slices": idx, "mirror": MIRROR,
               "results": [{"crop": c, "variant": v[0],
                            "macro_auc": auc_of(np.asarray(P[(c, v[0])]))[0],
                            "per_label": auc_of(np.asarray(P[(c, v[0])]))[1]}
                           for c in CROPS for v in VARIANTS]},
              open("/kaggle/working/exp085_axis.json", "w"), indent=1)
    print("wrote /kaggle/working/exp085_axis.json")


if __name__ == "__main__":
    main()
'''

out = head + TAIL
p.write_text(out, encoding="utf-8")
py_compile.compile(str(p), doraise=True)
print("rewrote tail: syntax OK, %d bytes" % len(out))
assert "MODES" not in out, "stale MODES reference remains"
assert "GRP = lambda" not in out, "stale GRP helper remains"
print("no stale MODES / GRP references")
