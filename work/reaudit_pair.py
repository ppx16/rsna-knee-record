"""Re-derive the final pair from RECORDED snapshots, not from the prose in the sheets.

Why redo it: the `0.943` block's bounds have moved a lot (376..1389 on 10-05, 485..1483 on 10-06), and
every in-block percentile quoted in the selection sheets was computed against whatever bounds existed at
the time. This puts every archival and measured move into ONE coordinate system -- the current block --
so the comparison is finally like-for-like.

The pair's value is the BETTER of its two arms, so the question is not "which two score highest" but
"which two maximise E[max(a, b)]". That needs each arm's effect and its sign uncertainty, not its score.
"""
import csv
import glob
import json
import pathlib

STATE = pathlib.Path("state")

print("=== the block, over time (from state/lb_snapshot_*.json) ===")
rows = []
for f in sorted(STATE.glob("lb_snapshot_*.json")):
    d = json.loads(f.read_text(encoding="utf-8"))
    b = d.get("block", {})
    rows.append((d["at"], d["n_teams"], d["our_rank"], d.get("margin"), b.get("lo"), b.get("hi"),
                 b.get("n"), d.get("in_block_pct")))
print("%-21s %6s %5s %5s %6s %6s %6s %8s" % ("at", "teams", "rank", "marg", "lo", "hi", "n", "in-blk%"))
for r in rows:
    print("%-21s %6s %5s %5s %6s %6s %6s %8s" % tuple("" if x is None else x for x in r))

print()
print("=== submissions and the snapshots that bracket them ===")
subs = []
for f in sorted(STATE.glob("submit_*.json")):
    d = json.loads(f.read_text(encoding="utf-8"))
    ref = d.get("ref")
    arm = d.get("arm")
    # find the ledger score
    score = ""
    for r in csv.DictReader((STATE / "lb_ground_truth.csv").open(encoding="utf-8")):
        if str(r.get("ref")) == str(ref):
            score = r.get("score")
            break
    subs.append((d["at"], ref, arm, d.get("before_snapshot", {}).get("in_block_pct"), score))
print("%-21s %-10s %-22s %9s %7s" % ("submitted_at", "ref", "arm", "before%", "score"))
for s in subs:
    print("%-21s %-10s %-22s %9s %7s" % tuple("" if x is None else x for x in s))

print()
print("=== measured in-block moves (from the docs, restated) ===")
for arm, ref, before, after, note in [
    ("0943cfmsft2", "56814081", 38.89, 13.66, "the only measured gain"),
    ("0943cfmsft2m", "56826372", 13.67, 14.31, "slightly worse -> not our best"),
    ("0943cf448", "56798408", 39.50, 38.90, "no change"),
]:
    print("   %-14s %-9s %6.2f%% -> %6.2f%%   delta %+6.2f   %s" % (arm, ref, before, after, after - before, note))

print()
print("=== current block and where our measured arms sit in it ===")
cur = json.loads(sorted(STATE.glob("lb_snapshot_*.json"))[-1].read_text(encoding="utf-8"))
lo, hi = cur["block"]["lo"], cur["block"]["hi"]
print("   current: %s  lo=%d hi=%d n=%d  our rank=%d margin=%d"
      % (cur["at"], lo, hi, cur["block"]["n"], cur["our_rank"], cur["margin"]))
print("   our in-block percentile now: %.3f%%" % cur["in_block_pct"])
