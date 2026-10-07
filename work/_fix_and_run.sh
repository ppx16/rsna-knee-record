#!/bin/bash
# Fix the one stale reference the two-part loader patch left behind, then run.
#
# The patch replaced `vols = np.load(...)` with a PARTS list but left a log line reading `vols.shape`, and my
# assertion only looked for `vols[pos` and `build_x(vols`, so it passed. The corpus itself loaded correctly --
# both parts, 3200 + 1207 -- which is the part that mattered.
set -eu
sed -i 's/\r$//' /root/fix_and_run.sh 2>/dev/null || true
cd /root

python3 - <<'PY' 2>/dev/null || /root/miniconda3/bin/python - <<'PY'
import pathlib, re
p = pathlib.Path('/root/arm3d_cuda.py')
t = p.read_text(encoding='utf-8')
old = "    log('[3darm] vols %s | ids %d | labels %d | folds %d' % (vols.shape, len(ids), len(lab), len(folds)))"
new = ("    log('[3darm] studies %d across %d part(s) | labels %d | folds %d'\n"
       "        % (len(ids), len(PARTS), len(lab), len(folds)))")
assert old in t, 'stale vols.shape line not found'
t = t.replace(old, new, 1)
left = [l.strip() for l in t.splitlines() if re.search(r'\bvols\.shape|\bvols\[pos|build_x\(vols', l)]
assert not left, 'other stale references remain: %r' % left[:3]
p.write_text(t, encoding='utf-8')
print('patched; stale references now:', left)
PY

echo "=== start training ==="
nohup /root/miniconda3/bin/python /root/arm3d_cuda.py --steps 2000 --bs 8 --depth 32 --hw 128 --width 32 \
      --eval-every 250 > /root/arm3d.log 2>&1 &
echo "pid $!"
sleep 110
echo "=== first 110s ==="
tail -22 /root/arm3d.log
