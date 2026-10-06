"""Is the 58-study GOLD truth baked into `report_labels_v5.csv` -- the table every member trains on?

`report_labels_v5` scoring AUC = 1.0000 against the gold binarisation is either (a) a tautology, because
the gold labels ARE in the table, or (b) an extraordinary coincidence. This distinguishes them by
comparing the VALUES elementwise rather than the ranking.

If it is (a), two things follow and both are serious:
  * `gold-58` is NOT a clean holdout for anything trained on v5, so every gold-58 number in this repo is
    optimistic;
  * it would explain the systematic gap we have been attributing to "instrument difference" --
    ConvNeXt gold-58 0.9003 vs 870-OOF 0.8493, and msft2 gold-58 0.9069 vs OOF 0.8847.
"""
import glob

import numpy as np
import pandas as pd

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

z = np.load(sorted(glob.glob("downloads/**/raptor_gold_*.npz", recursive=True))[0], allow_pickle=True)
truth = np.asarray(z["truth"], dtype=float)
ids = [str(x) for x in z["ids"]]
print("gold truth: %s   distinct values: %s" % (truth.shape, sorted(set(np.round(truth.ravel(), 4)))[:12]))

v5 = pd.read_csv("downloads/labels/rsna-knee-llm-labels-4-source-merged/report_labels_v5.csv")
v5["StudyInstanceUID"] = v5["StudyInstanceUID"].astype(str)
m = pd.DataFrame({"StudyInstanceUID": ids}).merge(v5, on="StudyInstanceUID", how="left")
V = m[LAB].to_numpy(dtype=float)
print("v5 values on the gold studies: distinct: %s" % sorted(set(np.round(V.ravel(), 4)))[:12])

print("\nexact equality, elementwise: %s  (%d/%d cells)"
      % (np.array_equal(truth, V), int((truth == V).sum()), truth.size))
close = np.isclose(truth, V, atol=1e-6)
print("equal within 1e-6        : %s  (%d/%d)" % (close.all(), int(close.sum()), truth.size))

Tb, Vb = (truth > 0.5).astype(int), (V > 0.5).astype(int)
print("binarised agreement      : %d/%d cells" % (int((Tb == Vb).sum()), truth.size))

print("\nper-label: gold positives | v5 positives | agreement")
for j, L in enumerate(LAB):
    print("   %-20s %3d | %3d | %s" % (L, int(Tb[:, j].sum()), int(Vb[:, j].sum()),
                                       "same" if np.array_equal(Tb[:, j], Vb[:, j]) else "DIFFERS"))
