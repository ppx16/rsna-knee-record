"""Extract the code + markdown content of a notebook into a compact readable .py-ish digest."""
import json, sys, os, re

def cell_src(c):
    s = c.get('source')
    if isinstance(s, list):
        return ''.join(s)
    return s or ''

def digest(path, out):
    nb = json.load(open(path, encoding='utf-8'))
    cells = nb.get('cells', [])
    lines = []
    lines.append('# ==== %s : %d cells ====' % (os.path.basename(path), len(cells)))
    for i, c in enumerate(cells):
        t = c.get('cell_type')
        src = cell_src(c).rstrip()
        if not src.strip():
            continue
        lines.append('\n# ---- cell %d [%s] ----' % (i, t))
        lines.append(src)
    txt = '\n'.join(lines)
    open(out, 'w', encoding='utf-8').write(txt)
    return len(cells), len(txt.splitlines())

if __name__ == '__main__':
    import glob
    root = r'D:\kaggle\rsna-knee\downloads\research\nb'
    for d in sorted(os.listdir(root)):
        dd = os.path.join(root, d)
        for f in os.listdir(dd):
            if f.endswith('.ipynb'):
                src = os.path.join(dd, f)
                out = os.path.join(dd, 'DIGEST.py')
                nc, nl = digest(src, out)
                print('%-70s cells=%-4d lines=%d' % (d, nc, nl))
