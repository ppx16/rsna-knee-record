"""Fill in the two PENDING ledger rows with their measured public scores."""
import csv
import pathlib

LED = pathlib.Path("state/lb_ground_truth.csv")
rows = list(csv.reader(LED.open(encoding="utf-8", newline="")))

SCORES = {"56845973": "0.940", "56848247": "0.942"}
n = 0
for r in rows:
    if len(r) >= 2 and r[1] in SCORES:
        r[3] = SCORES[r[1]]
        r[4] = "COMPLETE"
        n += 1
        print("updated ref %s -> %s COMPLETE" % (r[1], r[3]))

with LED.open("w", encoding="utf-8", newline="") as fh:
    csv.writer(fh).writerows(rows)
print("rows updated:", n)
print()
print("=== last 6 ledger rows ===")
for r in rows[-6:]:
    print("  %s | %s | %s | %s" % (r[0][:19], r[1], r[3] or "-", r[2][:70]))
