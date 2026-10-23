import json, sys, io, os

def dump(path, out):
    nb = json.load(open(path, encoding='utf-8'))
    lines = []
    for i, c in enumerate(nb.get('cells', [])):
        st = c.get('cell_type')
        src = ''.join(c.get('source', []))
        lines.append('\n' + '=' * 78)
        lines.append('### CELL %d  [%s]' % (i, st))
        lines.append('=' * 78)
        lines.append(src)
    open(out, 'w', encoding='utf-8').write('\n'.join(lines))
    print('wrote', out, os.path.getsize(out), 'bytes')

if __name__ == '__main__':
    dump(sys.argv[1], sys.argv[2])
