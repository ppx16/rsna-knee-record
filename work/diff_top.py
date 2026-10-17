import json, glob, difflib, re

A = glob.glob('scratch/top/blend-gold-dataset-restored/*.ipynb')[0]      # 0.954
B = glob.glob('scratch/top/blend-mine-in-private-lb-luck-to-all-upvoters/*.ipynb')[0]  # 0.956
C = glob.glob('kernels/kernel-blendgold-restored/*.ipynb')[0]           # ours 0.954


def cells(p):
    nb = json.load(open(p, encoding='utf-8'))
    return [(c.get('cell_type'), ''.join(c.get('source', []))) for c in nb.get('cells', [])]


def norm(s):
    # collapse the giant inlined python-source string literals so the diff shows structure
    return '\n'.join(l for l in s.splitlines())


for name, p in (('A=gold-restored 0.954', A), ('B=blend-mine-in 0.956', B), ('C=ours 0.954', C)):
    cs = cells(p)
    code = [s for t, s in cs if t == 'code']
    print('%-24s cells=%d code=%d  total_code_chars=%d' % (name, len(cs), len(code), sum(map(len, code))))
    print('    biggest code cell chars:', sorted((len(s) for s in code), reverse=True)[:4])

print()
print('#' * 90)
print('# structural diff A(0.954) vs B(0.956), cell by cell (code only)')
print('#' * 90)
ca = [s for t, s in cells(A) if t == 'code']
cb = [s for t, s in cells(B) if t == 'code']
for i, (x, y) in enumerate(zip(ca, cb)):
    if x == y:
        continue
    print('\n===== code cell %d differs  (A %d chars / B %d chars)' % (i, len(x), len(y)))
    d = list(difflib.unified_diff(x.splitlines(), y.splitlines(), 'A', 'B', lineterm='', n=2))
    for l in d[:80]:
        print('   ', l[:200])

print()
print('#' * 90)
print('# the blend / weight / deadline lines in each')
print('#' * 90)
for name, p in (('A=0.954', A), ('B=0.956', B), ('C=ours', C)):
    print('\n---', name)
    txt = '\n'.join(s for t, s in cells(p) if t == 'code')
    for kw in ('W_RAPTOR', 'W_OURS', 'DEADLINE_HOURS', 'max-fallback-fraction', '_weight_name',
               'raptor_per_class', 'ARMS', 'weight_parts'):
        for l in txt.splitlines():
            if kw in l:
                print('   [%-22s] %s' % (kw, l.strip()[:170]))
                break
