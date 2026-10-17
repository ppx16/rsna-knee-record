import json, glob, sys, os

path = sys.argv[1]
out = sys.argv[2]
nb = json.load(open(path, encoding='utf-8'))
lines = []
for i, c in enumerate(nb.get('cells', [])):
    if c.get('cell_type') != 'code':
        continue
    for o in c.get('outputs', []) or []:
        t = o.get('output_type')
        if t == 'stream':
            lines.append('--- cell %d stream ---' % i)
            lines.append(''.join(o.get('text', [])))
        elif t in ('execute_result', 'display_data'):
            d = o.get('data', {})
            if 'text/plain' in d:
                lines.append('--- cell %d result ---' % i)
                lines.append(''.join(d['text/plain']))
        elif t == 'error':
            lines.append('--- cell %d ERROR ---' % i)
            lines.append('\n'.join(o.get('traceback', [])))
open(out, 'w', encoding='utf-8').write('\n'.join(lines))
print('wrote', out, os.path.getsize(out))
