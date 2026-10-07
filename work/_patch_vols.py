"""Replace the one stale `vols.shape` reference the two-part loader patch left behind."""
import pathlib
import re

p = pathlib.Path("/root/arm3d_cuda.py")
t = p.read_text(encoding="utf-8")
old = "    log('[3darm] vols %s | ids %d | labels %d | folds %d' % (vols.shape, len(ids), len(lab), len(folds)))"
new = ("    log('[3darm] studies %d across %d part(s) | labels %d | folds %d'\n"
       "        % (len(ids), len(PARTS), len(lab), len(folds)))")
assert old in t, "stale vols.shape line not found"
t = t.replace(old, new, 1)
left = [l.strip() for l in t.splitlines() if re.search(r"\bvols\.shape|\bvols\[pos|build_x\(vols", l)]
assert not left, "other stale references remain: %r" % left[:3]
p.write_text(t, encoding="utf-8")
print("patched OK; stale references now:", left)
