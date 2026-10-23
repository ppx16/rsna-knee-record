"""Self-check of the durable state files. Run before trusting any handover note.

Validates:
  * state/final_pair.json: schema, explicit boolean `oai` per candidate, tick refs exist,
    tick refs have boolean oai, and the never-two-OAI invariant holds;
  * state/FACTS.md: unique entry ids, no duplicates, reports the max per section;
  * state/rsna_submissions.jsonl: parses, reports the rows still missing a score.

Written because the alternative is ad-hoc inline python, which keeps tripping over
PowerShell quoting (FACTS A185).
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    bad = 0

    # ---- final_pair.json -----------------------------------------------------------------
    print("=" * 80)
    print("state/final_pair.json")
    fp = ROOT / "state" / "final_pair.json"
    try:
        d = json.loads(fp.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL: {exc}")
        return 1
    cands = d.get("candidates", {})
    tick = d.get("tick", [])
    print(f"  ok: parses; {len(cands)} candidates; tick={tick}")
    if len(tick) > 2:
        print(f"  FAIL: {len(tick)} ticks but the rules allow at most 2")
        bad += 1
    flags = []
    for ref in tick:
        c = cands.get(ref)
        if c is None:
            print(f"  FAIL: ticked ref {ref} is not in candidates")
            bad += 1
            continue
        if not isinstance(c.get("oai"), bool):
            print(f"  FAIL: {ref} has no BOOLEAN 'oai' (got {c.get('oai')!r}) -- see FACTS A184")
            bad += 1
        flags.append(c.get("oai"))
    print(f"  ticked oai flags: {flags}")
    if len(tick) == 2 and all(f is True for f in flags):
        print("  FAIL: BOTH ticks are OAI -- that discards the only criterion the pair exists for")
        bad += 1
    for ref, c in cands.items():
        for k in ("label", "bloodline", "why"):
            if not c.get(k):
                print(f"  WARN: {ref} missing {k!r}")
    print(f"  candidates: " + ", ".join(f"{r}(oai={c.get('oai')},score={c.get('score')})"
                                       for r, c in cands.items()))

    # ---- FACTS.md ------------------------------------------------------------------------
    print("=" * 80)
    print("state/FACTS.md")
    lines = (ROOT / "state" / "FACTS.md").read_text(encoding="utf-8").splitlines()
    ids = [m.group(1) + m.group(2) for ln in lines
           if (m := re.match(r"^\|\s*([ACD])(\d+)\s*\|", ln))]
    print(f"  {len(lines)} lines, {len(ids)} entry rows")
    for p in "ACD":
        n = sorted({int(i[1:]) for i in ids if i.startswith(p)})
        print(f"    {p}: unique={len(n)} max={n[-1] if n else '-'}")
    dup = [k for k, v in Counter(ids).items() if v > 1]
    if dup:
        print(f"  FAIL: duplicate ids {dup}")
        bad += 1
    else:
        print("  ok: no duplicate ids")

    # ---- submissions ledger --------------------------------------------------------------
    print("=" * 80)
    print("state/rsna_submissions.jsonl")
    led = ROOT / "state" / "rsna_submissions.jsonl"
    rows, broken = [], 0
    if led.is_file():
        for ln in led.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                broken += 1
    print(f"  {len(rows)} rows, {broken} unparsable")
    if broken:
        bad += 1
    pending = [r for r in rows if not r.get("score")]
    print(f"  {len(pending)} row(s) without a score:")
    for r in pending[-8:]:
        print(f"    {r.get('ref')}  {r.get('date')}  {str(r.get('desc'))[:70]}")
    scored = [r for r in rows if r.get("score")]
    if scored:
        best = max(scored, key=lambda r: r["score"])
        print(f"  best scored submission: {best['ref']} at {best['score']}")

    print("=" * 80)
    print("STATE CHECK:", "OK" if bad == 0 else f"{bad} problem(s)")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
