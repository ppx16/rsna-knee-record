# ==== six-ensembling-ideas-that-failed-0-873-to-0-919.ipynb : 20 cells ====

# ---- cell 0 [markdown] ----
# Six ensembling ideas that failed, and the two that worked (0.873 → 0.919)

Public notebooks tend to show the thing that worked. This one shows the eight things I tried, because
six of them failed and the failures were more instructive than the successes — three of them looked
like clear wins on validation before turning out to be noise.

**Where the numbers come from.** Six knee-MRI models, each trained separately and each scored on the
leaderboard or on a held-out set, then blended:

| member | architecture | notes |
|---|---|---|
| v2 | ResNet-34, sagittal fluid-sensitive only, mean-pooled slices | LB 0.873 |
| v3 | ResNet-34, three planes, per-label plane-mixture head | LB 0.874 |
| v4 | ConvNeXt-T, gated slice attention, EMA weights | LB 0.883 |
| v5 | **RadImageNet** ResNet-50 (medically pretrained) | weakest member |
| v6 / v7 | DINOv2-small, five series slots, geometric slice ordering | two folds |

| submission | public LB |
|---|---|
| best single model (v4) | 0.883 |
| 3-member rank mean | 0.898 |
| 4-member (+RadImageNet) | 0.907 |
| 5-member | 0.913 |
| **6-member + per-target weights** | **0.919** |

Everything below is reproducible here: the six members' validation predictions are attached as a
dataset, so you can rerun every experiment on CPU in seconds without my model weights.

# ---- cell 1 [code] ----
import itertools
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

import glob
path = glob.glob('/kaggle/input/**/member_val_preds.npz', recursive=True)[0]
z = np.load(path, allow_pickle=True)
LABELS = [str(x) for x in z['labels']]
Y = z['Y'].astype(float)                       # radiologist labels, 58 annotated studies
names = [k for k in ['V2', 'V3', 'V4', 'V5', 'V6', 'V7'] if k in z.files]
P = {k: z[k] for k in names}

def rank_norm(a):
    return np.stack([rankdata(a[:, j]) / len(a) for j in range(a.shape[1])], 1)

R = {k: rank_norm(v) for k, v in P.items()}
Rs = [R[k] for k in names]

def macro(pred, idx=None):
    idx = slice(None) if idx is None else idx
    out = []
    for j in range(len(LABELS)):
        y = (Y[idx, j] >= 0.5).astype(int)
        if 0 < y.sum() < len(y):
            out.append(roc_auc_score(y, pred[idx, j]))
    return float(np.mean(out)) if out else np.nan

print(f'{len(Y)} annotated studies, {len(names)} members')
for k in names:
    print(f'  {k}: {macro(R[k]):.4f}')
print(f'\nequal rank mean: {macro(sum(Rs)/len(Rs)):.4f}')

# ---- cell 2 [markdown] ----
## ✅ What worked #1 — rank mean rather than probability mean

The metric is ROC AUC, which reads only the *order* of scores. Averaging raw probabilities lets whichever
member happens to output the widest numeric range dominate the sum — a property the metric never asked
about. Rank-normalising each member per label first puts everyone on the same footing.

# ---- cell 3 [code] ----
prob_mean = np.mean([P[k] for k in names], axis=0)
rank_mean = sum(Rs) / len(Rs)
print(f'probability mean : {macro(prob_mean):.4f}')
print(f'rank mean        : {macro(rank_mean):.4f}')

# ---- cell 4 [markdown] ----
## ✅ What worked #2 — per-target complement weights

Each member is stronger on some findings than others: the sagittal-only model reads ACL best, the
medically-pretrained one is the only member that reads Synovitis above 0.72. Giving every member double
weight on its own specialty labels (declared in advance from training logs, then normalised per label)
was worth about +0.007 on validation and +0.004 on the leaderboard.

The important discipline: the specialty lists were written down **before** scoring the blend, not searched
for afterwards. The next section shows what happens when you search instead.

# ---- cell 5 [code] ----
SPECIALTY = {
    'V2': ['ACL'],
    'V3': ["Baker's", 'Lateral Meniscus', 'Lateral OA'],
    'V4': ['Effusion'],
    'V5': ['Synovitis', 'Fracture', 'Medial OA'],
    'V6': ['Medial Meniscus', 'Contusion', 'Effusion'],
    'V7': ['MCL', 'Contusion', 'Medial OA'],
}

def complement_weights(subset=None):
    subset = subset or names
    W = np.ones((len(subset), len(LABELS)))
    for i, k in enumerate(subset):
        for lab in SPECIALTY[k]:
            W[i, LABELS.index(lab)] += 1.0
    return W / W.sum(0, keepdims=True)

def blend(subset, W):
    return sum(R[k] * W[i][None, :] for i, k in enumerate(subset))

comp = blend(names, complement_weights())
print(f'equal weights      : {macro(rank_mean):.4f}')
print(f'complement weights : {macro(comp):.4f}')

# ---- cell 6 [markdown] ----
## ❌ Failure 1 — per-label member dropping

If each member has weak labels, why not set its weight to **zero** there instead of merely down-weighting?
On validation this looked like the biggest win of the whole project.

# ---- cell 7 [code] ----
per = {k: {c: (roc_auc_score((Y[:, j] >= .5).astype(int), R[k][:, j])
                if 0 < (Y[:, j] >= .5).sum() < len(Y) else 0.5)
           for j, c in enumerate(LABELS)} for k in names}

def drop_weights(drop_n, per_table):
    W = np.ones((len(names), len(LABELS)))
    for j, c in enumerate(LABELS):
        for k in sorted(names, key=lambda k: per_table[k][c])[:drop_n]:
            W[names.index(k), j] = 0.0
    for i, k in enumerate(names):
        for lab in SPECIALTY[k]:
            if W[i, LABELS.index(lab)] > 0:
                W[i, LABELS.index(lab)] += 1.0
    s = W.sum(0, keepdims=True); s[s == 0] = 1
    return W / s

for d in (1, 2):
    print(f'drop worst {d} per label: {macro(blend(names, drop_weights(d, per))):.4f}   <- looks great')
print(f'complement baseline    : {macro(comp):.4f}')

# ---- cell 8 [markdown] ----
**+0.011 over the baseline.** That would have been the single largest gain of the project.

It is entirely an illusion. The member being dropped is chosen using the same 58 studies the blend is then
scored on. Refit the choice on one half and score on the other, and the gain disappears:

# ---- cell 9 [code] ----
rng = np.random.RandomState(0)
n = len(Y)
res = {'complement': [], 'drop1': [], 'drop2': []}
for _ in range(40):
    p = rng.permutation(n); h = n // 2
    for a, b in ((p[:h], p[h:]), (p[h:], p[:h])):
        per_a = {k: {c: (roc_auc_score((Y[a, j] >= .5).astype(int), R[k][a, j])
                         if 0 < (Y[a, j] >= .5).sum() < len(a) else 0.5)
                     for j, c in enumerate(LABELS)} for k in names}
        res['complement'].append(macro(comp, b))
        res['drop1'].append(macro(blend(names, drop_weights(1, per_a)), b))
        res['drop2'].append(macro(blend(names, drop_weights(2, per_a)), b))

print('honest out-of-fold (choice fitted on the other half):')
for k, v in res.items():
    print(f'  {k:12} {np.nanmean(v):.4f} +/- {np.nanstd(v):.4f}')

# ---- cell 10 [markdown] ----
In-sample **+0.011**, out-of-fold **+0.0003**. The entire gain was the selection itself.

# ---- cell 11 [markdown] ----
## ❌ Failure 2 — searching for the best subset of members

Same trap, one level up. With six members there are 57 subsets of size 3 to 6; the best of them scores
noticeably above the full set. A properly nested test — pick the subset on one half, score it on the
untouched half — shows that *selecting at all* costs you accuracy.

# ---- cell 12 [code] ----
subsets = [list(s) for r in range(3, len(names) + 1) for s in itertools.combinations(names, r)]
cand = {'+'.join(s): blend(s, complement_weights(s)) for s in subsets}

best_name = max(cand, key=lambda k: macro(cand[k]))
print(f'best subset in-sample : {best_name}  {macro(cand[best_name]):.4f}')
print(f'all six               : {macro(comp):.4f}')

sel, fixed = [], []
rng = np.random.RandomState(1)
for _ in range(60):
    p = rng.permutation(n); h = n // 2
    for a, b in ((p[:h], p[h:]), (p[h:], p[:h])):
        pick = max(cand.values(), key=lambda P_: macro(P_, a))     # choose on half a
        sel.append(macro(pick, b))                                  # score on half b
        fixed.append(macro(comp, b))
print(f'\nnested: select-a-subset {np.nanmean(sel):.4f} | keep all six {np.nanmean(fixed):.4f}')
print(f'selecting costs {np.nanmean(sel)-np.nanmean(fixed):+.4f}')

# ---- cell 13 [markdown] ----
## ❌ Failures 3–6, briefly

**Test-time augmentation.** Three crops per member, averaged. It genuinely helps each member individually
(+0.003 to +0.005 each) — and moves the blend by **-0.001**. Rank-mean ensembling is already averaging
away that variance, so TTA re-buys something you own, at three times the inference cost. Worth keeping for
a single-model submission; not for an ensemble.

**Borrowing a correlated label.** Synovitis is the worst label for every model here (0.62-0.73) and
correlates about 0.5 with Effusion, which is one of the best. Mixing a fraction of the Effusion rank into
the Synovitis prediction sounded free. Every mixing weight made it worse, monotonically.

# ---- cell 14 [code] ----
si, ei = LABELS.index('Synovitis'), LABELS.index('Effusion')
for w in (0.0, 0.1, 0.2, 0.3, 0.4):
    q = comp.copy()
    q[:, si] = (1 - w) * comp[:, si] + w * comp[:, ei]
    syn = roc_auc_score((Y[:, si] >= .5).astype(int), q[:, si])
    print(f'  borrow {w:.1f}: macro {macro(q):.4f}  synovitis {syn:.3f}')

# ---- cell 15 [markdown] ----
**Power-weighted ranks.** Sharpening or flattening the rank distribution before averaging (rank\*\*p).
p = 1 was best; every other exponent was worse.

**A seventh member.** I trained a third DINOv2 fold. It was the *strongest single member* I had, and
adding it moved the blend from 0.8987 to 0.8980 — nothing. Three folds of one architecture are near
duplicates of each other: blending only the DINOv2 folds together scores 0.879, barely above the best one
alone at 0.875. Meanwhile the **weakest** member in the whole set, RadImageNet at 0.843, was worth +0.009
when it joined, because a medically pretrained encoder fails differently from ImageNet ones.

That is the single most useful thing I learned: **an ensemble is fed by disagreement, not by member quality.**

# ---- cell 16 [code] ----
pairs = [(a, b) for a, b in itertools.combinations(names, 2)]
corr = [(np.mean([np.corrcoef(R[a][:, j], R[b][:, j])[0, 1] for j in range(len(LABELS))]), a, b)
        for a, b in pairs]
corr.sort()
print('most different pairs (mean per-label rank correlation):')
for c, a, b in corr[:4]:
    print(f'  {a}-{b}: {c:.3f}')
print('most similar pairs:')
for c, a, b in corr[-4:]:
    print(f'  {a}-{b}: {c:.3f}')

# ---- cell 17 [markdown] ----
## Why validation lies here, in one plot

Fifty-eight annotated studies, 9 to 35 positives per label. Bootstrapping a *fixed* set of predictions
shows how much a score can move by luck alone — and the interval is wider than the gap between most of
the things I was comparing.

# ---- cell 18 [code] ----
rng = np.random.RandomState(0)
boot = []
for _ in range(2000):
    idx = rng.randint(0, n, n)
    boot.append(macro(comp, idx))
lo, hi = np.percentile(boot, [2.5, 97.5])

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(boot, bins=50, color='#4477AA')
ax.axvline(lo, color='#CC3311', ls='--'); ax.axvline(hi, color='#CC3311', ls='--')
ax.set_title(f'Same predictions, resampled studies — 95% CI width {hi-lo:.3f}')
ax.set_xlabel('macro AUC'); plt.tight_layout(); plt.show()
print(f'95% CI: {lo:.3f} to {hi:.3f}  (width {hi-lo:.3f})')
print('for comparison, the whole 0.873 -> 0.919 climb is 0.046')

# ---- cell 19 [markdown] ----
## What I would tell myself at the start

1. **Blend by rank, not probability**, whenever the metric only reads order.
2. **Add members that fail differently**, not members that score well. My weakest model was my most
   valuable addition; my strongest addition was worthless.
3. **Write weighting rules down before you score them.** Anything chosen by looking at a small validation
   set will look good on that set and nowhere else.
4. **Nested-validate any selection step.** Two of my three "best ideas" reversed sign under a proper test.
5. **Bootstrap the validation metric once, at the start.** If the confidence interval is wider than the
   effects you plan to chase, that set cannot referee your experiments — and mine could not.

Corrections and counter-results welcome, especially from anyone who has made a learned stacker work on a
validation set this small. I could not, and I would like to know what I was missing.