"""One-shot: retarget the msft2 builder's gate from the metadata number to our measured reload.

Kept as a file rather than a `python -c` because the strings contain `%` and both quote kinds, which
PowerShell eats -- the third time this session that a shell one-liner has mangled a payload.
"""
import pathlib

p = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "build_msft2_kernel.py"
t = p.read_text(encoding="utf-8")

SUBS = [
    ('bar = "%.4f" % PARENT_BAR', 'bar = "%.4f" % GATE_BAR'),
    ("def patch(text: str) -> str:", 'def patch(text: str, mode: str = "gate") -> str:'),
    ('text = patch(src.read_text(encoding="utf-8"))',
     'text = patch(src.read_text(encoding="utf-8"), a.mode)'),
]

for old, new in SUBS:
    n = t.count(old)
    print("found %d : %s" % (n, old[:64]))
    if n == 1:
        t = t.replace(old, new, 1)
    else:
        raise SystemExit("expected exactly 1 match for %r" % old[:64])

p.write_text(t, encoding="utf-8")
print("rewrote", p.name)
