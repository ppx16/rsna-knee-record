import glob, json, unicodedata, collections

p = glob.glob('scratch/top/blend-mine-in-private-lb-luck-to-all-upvoters/*.ipynb')[0]
t = open(p, encoding='utf-8').read()

cnt = collections.Counter()
for ch in t:
    if ord(ch) > 127:
        cnt[ch] += 1

print('whole file: distinct non-ASCII:', len(cnt), 'total:', sum(cnt.values()))
for ch, n in cnt.most_common():
    try:
        name = unicodedata.name(ch)
    except ValueError:
        name = '?'
    idx = [i for i, c in enumerate(t) if c == ch][:3]
    print('  %-6r U+%04X x%-4d %-32s first offsets %s' % (ch, ord(ch), n, name, idx))
    for i in idx[:1]:
        print('      context: ...%s...' % t[max(0, i - 80):i + 80].replace('\n', '\\n'))

# also show what is around byte position 2481
b = open(p, 'rb').read()
print()
print('bytes around 2481:', b[2420:2540])
print('text  around 2481:', repr(t[2380:2520]))
