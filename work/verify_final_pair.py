"""Verify the final pair is still present, selectable, and uniquely identifiable on the Submissions page.

The tick is a BROWSER action -- there is no `select` verb, `ApiSubmission` has no `selected` field, and
legacy `/api/v1` 401s (all three verified 2026-10-03 on kaggle 2.1.2 and the 2.2.2 wheel). So the ONLY
discriminator the user has is the description text shown on that page, which makes "is this description
unique among our submissions?" a load-bearing question rather than a cosmetic one.

Re-checked now because several submissions have been added since the sheet was written.
"""
import re
import subprocess

KG = r"D:\kaggle\venv-kaggle\Scripts\kaggle.exe"
COMP = "rsna-knee-abnormality-detection"
PAIR = {"56814081": "PRIMARY  0943cfmsft2",
        "56613303": "SECOND   0943cflat"}

r = subprocess.run([KG, "competitions", "submissions", "-c", COMP],
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
txt = (r.stdout or "") + (r.stderr or "")
if "SSLError" in txt or "Traceback" in txt:
    raise SystemExit("TRANSPORT ERROR -- retry rather than concluding anything")

rows = []
for ln in txt.splitlines():
    ln = ln.strip()
    m = re.match(r"^(\d{8})\s+(\S+)\s+(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\S*\s+(.*)$", ln)
    if not m:
        continue
    ref, fname, date, rest = m.groups()
    sm = re.search(r"SubmissionStatus\.(\w+)\s+([\d.]+|)", rest)
    desc = rest[:rest.find("SubmissionStatus")].strip() if "SubmissionStatus" in rest else rest
    rows.append((ref, date, (sm.group(2) or "-") if sm else "?", sm.group(1) if sm else "?", desc, fname))

print("total submissions visible: %d" % len(rows))
by_ref = {r[0]: r for r in rows}

print()
print("=== the pair ===")
ok = True
for ref, label in PAIR.items():
    row = by_ref.get(ref)
    if row is None:
        print("   %s  ref %s  *** NOT PRESENT IN THE SUBMISSIONS LIST ***" % (label, ref))
        ok = False
        continue
    date, sc, st, desc, fname = row[1], row[2], row[3], row[4], row[5]
    print("   %s" % label)
    print("      ref=%s  date=%s  public=%s  status=%s  file=%s" % (ref, date, sc, st, fname))
    print("      description: %s" % desc[:150])
    if st != "COMPLETE":
        print("      *** status is %s, not COMPLETE ***" % st)
        ok = False

print()
print("=== uniqueness: does any OTHER submission share the pair's description prefix? ===")
for ref, label in PAIR.items():
    row = by_ref.get(ref)
    if row is None:
        continue
    pref = row[4][:34].lower()
    clashes = [r[0] for r in rows if r[0] != ref and r[4][:34].lower() == pref]
    print("   %-24s prefix %-36r clashes: %s" % (label, pref, clashes or "NONE"))

print()
print("=== every 0.943+ submission still on the list (so we do not tick the auto-default) ===")
for ref, date, sc, st, desc, fname in rows:
    try:
        v = float(sc)
    except Exception:
        continue
    if v >= 0.942:
        mark = "  <== IN OUR PAIR" if ref in PAIR else ""
        print("   %-9s %-7s %s%s" % (ref, sc, desc[:78], mark))

print()
print("VERDICT:", "pair present and COMPLETE" if ok else "*** PROBLEM -- see above ***")
