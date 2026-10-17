import glob, json, unicodedata

p = glob.glob('scratch/top/blend-mine-in-private-lb-luck-to-all-upvoters/*.ipynb')[0]
b = open(p, 'rb').read()
print('bytes', len(b))
try:
    b.decode('gbk')
    print('GBK: OK, no changes needed')
except UnicodeDecodeError as e:
    print('GBK: FAILS ->', e)

# which characters cannot be encoded to GBK (scan the decoded utf-8 text)
t = b.decode('utf-8')
bad = {}
for ch in t:
    try:
        ch.encode('gbk')
    except UnicodeEncodeError:
        bad[ch] = bad.get(ch, 0) + 1
print('distinct non-GBK chars:', len(bad))
for ch, n in sorted(bad.items(), key=lambda kv: -kv[1]):
    try:
        name = unicodedata.name(ch)
    except ValueError:
        name = '?'
    print('   %r  U+%04X  x%-5d %s' % (ch, ord(ch), n, name))

# where do they occur? markdown only, or inside code too?
nb = json.loads(t)
for i, c in enumerate(nb.get('cells', [])):
    src = ''.join(c.get('source', []))
    hits = [ch for ch in src if ch in bad]
    if hits:
        print('cell %d [%s]: %d non-GBK chars %s' % (i, c.get('cell_type'), len(hits),
                                                     sorted(set(hits))))
