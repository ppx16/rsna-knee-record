"""Binarise the soft labels before roc_auc_score, then restart the run.

sklearn's roc_auc_score needs a BINARY y_true and refuses continuous labels -- 'continuous format is not
supported'. The label table here is probabilities (0.25, 0.975, ...), so every evaluation crashed at the first
checkpoint while training itself was fine.

The convention is stated rather than implied, because the gate compares against msft2's 0.8847 measured on the
same instrument and a different binarisation would silently move the bar: labels are binarised at > 0.5.
"""
import pathlib

p = pathlib.Path("/root/arm3d_cuda.py")
t = p.read_text(encoding="utf-8")

OLD = """        col = y[:, j]
        m = np.isfinite(col)
        if m.sum() < 10 or len(np.unique(col[m])) < 2:
            continue
        aucs.append(roc_auc_score(col[m], p[m, j]))"""
NEW = """        col = y[:, j]
        m = np.isfinite(col)
        if m.sum() < 10:
            continue
        # \u26a0\ufe0f BINARISE. The table carries SOFT probabilities; roc_auc_score refuses continuous y_true
        # ('continuous format is not supported'). Convention, stated so the bar cannot drift silently:
        # a study counts as positive for a finding when its label exceeds 0.5.
        b = (col[m] > 0.5).astype(np.int8)
        if b.min() == b.max():
            continue
        aucs.append(roc_auc_score(b, p[m, j]))"""
assert OLD in t, "macro_auc body not found"
t = t.replace(OLD, NEW, 1)
p.write_text(t, encoding="utf-8")
print("macro_auc now binarises at >0.5")
