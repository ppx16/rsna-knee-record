"""Verify the pair we intend to tick at the deadline.

The tick is a BROWSER action -- there is no `select` verb and `ApiSubmission` has no `selected`
field (verified 2026-10-03 on kaggle 2.1.2 and 2.2.2), so the only discriminator on that page is
the DESCRIPTION TEXT. That makes "is this description unique among our submissions?" a load-bearing
question rather than a cosmetic one.

Changed 2026-10-10 (three fixes, all of the "the receipt must describe what actually happened"
family):
  1. the pair is no longer hardcoded -- it is read from state/final_pair.json, which
     FINAL-TICK.md also reads, so the runbook and this verifier cannot drift apart;
  2. the kaggle executable is resolved instead of pinned to one venv path;
  3. it now ENFORCES the one rule that must not be broken: never tick two OAI-bloodline
     submissions, because that throws away the only reason the pair exists.

Usage: python work/verify_final_pair.py [--pair ref1,ref2] [--json]
"""

# --- GBK CONSOLE GUARD ------------------------------------------------------------------------------
# This console is cp936, which cannot encode the U+26A0 warning sign used below. An unguarded print
# raises UnicodeEncodeError at that line and kills the script MID-REPORT, while everything printed
# earlier is already on screen -- so the run looks complete.
import sys as _sys

try:
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass
# ----------------------------------------------------------------------------------------------------
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "state" / "final_pair.json"
COMP = "rsna-knee-abnormality-detection"


def kaggle_exe() -> str:
    """Env override -> the venv we used to use -> whatever is on PATH."""
    env = os.environ.get("KAGGLE_EXE")
    if env and Path(env).is_file():
        return env
    venv = ROOT.parent / "venv-kaggle" / "Scripts" / "kaggle.exe"
    if venv.is_file():
        return str(venv)
    found = shutil.which("kaggle")
    if found:
        return found
    raise SystemExit("cannot locate a kaggle executable (set KAGGLE_EXE)")


def load_cfg() -> dict:
    if not CFG_PATH.is_file():
        raise SystemExit(f"missing {CFG_PATH} -- it is the single source of truth for the pair")
    return json.loads(CFG_PATH.read_text(encoding="utf-8"))


def fetch_rows() -> list[tuple]:
    kg = kaggle_exe()
    last = ""
    for _ in range(4):
        r = subprocess.run([kg, "competitions", "submissions", "-c", COMP],
                           capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=600)
        txt = (r.stdout or "") + (r.stderr or "")
        if txt.strip() and "SSLError" not in txt and "Max retries" not in txt and "Traceback" not in txt:
            break
        last = txt[:200]
    else:
        # A transport failure must never be reported as "not present" (FACTS A173/A174/A181).
        raise SystemExit(f"TRANSPORT ERROR -- retry rather than concluding anything\n{last}")

    rows = []
    for ln in txt.splitlines():
        ln = ln.strip()
        m = re.match(r"^(\d{8})\s+(\S+)\s+(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\S*\s+(.*)$", ln)
        if not m:
            continue
        ref, fname, date, rest = m.groups()
        # optional score group: a PENDING row has no trailing score, and requiring one makes the
        # row report '?', which reads like a fault instead of "still being scored".
        sm = re.search(r"SubmissionStatus\.(\w+)(?:\s+([\d.]+))?", rest)
        desc = rest[:rest.find("SubmissionStatus")].strip() if "SubmissionStatus" in rest else rest
        rows.append((ref, date, (sm.group(2) or "-") if sm else "?", sm.group(1) if sm else "?", desc, fname))
    return rows


def main(argv: list[str]) -> int:
    cfg = load_cfg()
    cands = cfg["candidates"]
    tick = list(cfg["tick"])
    if "--pair" in argv:
        tick = argv[argv.index("--pair") + 1].split(",")
    if len(tick) > 2:
        print(f"!! {len(tick)} refs requested but the rules allow at most 2 final submissions")
        return 1

    rows = fetch_rows()
    by_ref = {r[0]: r for r in rows}
    print(f"total submissions visible: {len(rows)}")
    print(f"pair under test: {tick}   (from {CFG_PATH.name})")

    ok = True
    print("\n=== the pair ===")
    oai_flags = []
    for ref in tick:
        meta = cands.get(ref, {})
        row = by_ref.get(ref)
        if row is None:
            print(f"   ref {ref}  *** NOT PRESENT IN THE SUBMISSIONS LIST ***")
            ok = False
            continue
        date, sc, st, desc, fname = row[1:6]
        if not isinstance(meta.get("oai"), bool):
            print(f"   ref {ref}  *** final_pair.json has no BOOLEAN 'oai' flag for this ref ***")
            ok = False
        oai_flags.append(bool(meta.get("oai")))
        print(f"   {meta.get('label', '?'):14s} ref={ref}  date={date}  public={sc}  status={st}"
              f"  oai={meta.get('oai')}")
        print(f"      bloodline: {meta.get('bloodline', '?')}")
        print(f"      description: {desc[:150]}")
        if meta.get("match") and meta["match"].lower() not in desc.lower():
            print(f"      *** description does not contain the expected text {meta['match']!r} ***")
            ok = False
        tok = meta.get("unique_token")
        if tok and tok not in desc:
            print(f"      *** the unique token {tok!r} is MISSING -- this is probably the WRONG submission ***")
            ok = False
        if st != "COMPLETE":
            print(f"      *** status is {st}, not COMPLETE ***")
            ok = False

    # ---- the invariant that justifies the pair's existence -------------------------------
    # Uses the EXPLICIT boolean. An earlier version substring-matched the label, and
    # "ours-nonOAI".upper() contains "OAI", so it accused our non-OAI pair of being OAI.
    print("\n=== bloodline check (the one rule that must not be broken) ===")
    n_oai = sum(1 for f in oai_flags if f)
    print(f"   oai flags: {oai_flags}  -> {n_oai} of {len(oai_flags)} are OAI")
    if len(tick) == 2 and n_oai == 2:
        print("   *** BOTH ticks are OAI ***")
        print("   *** This discards the only criterion the pair exists for (not co-failing on private). ***")
        print(f"   *** {cfg['_decision_rules']['never']} ***")
        ok = False
    elif n_oai == 1:
        print("   exactly one OAI tick -> the other one survives any adverse OAI ruling. OK.")
    elif len(tick) == 2:
        print("   no OAI ticks -> immune to the ruling, but also forgoes the +0.006 upside. Allowed.")

    print("\n=== uniqueness: does any OTHER submission share a pair member's description prefix? ===")
    for ref in tick:
        row = by_ref.get(ref)
        if row is None:
            continue
        pref = row[4][:34].lower()
        clashes = [r[0] for r in rows if r[0] != ref and r[4][:34].lower() == pref]
        declared = cands.get(ref, {}).get("clashes_with") or []
        extra = [c for c in clashes if c not in declared]
        print(f"   {ref}: prefix {pref!r} clashes={clashes or 'NONE'}"
              f"{'  <-- UNDECLARED CLASH' if extra else ''}")
        if extra:
            print(f"      *** undeclared clash with {extra} -- update state/final_pair.json ***")
            ok = False

    print("\n=== do-not-tick list ===")
    for ref, why in cfg.get("do_not_tick", {}).items():
        print(f"   {ref}: {why}")

    print("\n=== every 0.942+ submission still on the list (so we do not tick the auto-default) ===")
    for ref, date, sc, st, desc, fname in rows:
        try:
            v = float(sc)
        except Exception:  # noqa: BLE001
            continue
        if v >= 0.942:
            mark = "  <== IN OUR PAIR" if ref in tick else ""
            print(f"   {ref:9s} {sc:7s} {st:10s} {desc[:70]}{mark}")

    print()
    print("VERDICT:", "pair present, COMPLETE, and legal" if ok else "*** PROBLEM -- see above ***")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(_sys.argv))
