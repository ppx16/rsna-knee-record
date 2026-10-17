# -*- coding: utf-8 -*-
"""Self-test: auc_soft must equal sklearn's roc_auc_score exactly on binary labels."""
import sys
import pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from largen_instrument import auc_soft  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

rng = np.random.default_rng(0)
worst = 0.0
for trial in range(200):
    n = int(rng.integers(20, 400))
    y = (rng.random(n) < rng.uniform(0.1, 0.9)).astype(float)
    if y.min() == y.max():
        continue
    # inject ties on purpose: discrete scores
    p = np.round(rng.random(n), int(rng.integers(1, 4)))
    a = auc_soft(y, p)
    b = roc_auc_score(y, p)
    worst = max(worst, abs(a - b))
print(f"binary equivalence over 200 trials: worst |auc_soft - roc_auc_score| = {worst:.2e}")
assert worst < 1e-12, "SOFT-AUC DOES NOT MATCH SKLEARN ON BINARY -- do not trust the soft column"
print("OK")
