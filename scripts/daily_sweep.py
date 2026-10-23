# -*- coding: utf-8 -*-
"""
The daily ritual, in one command -- user's standing instruction:
"每天都要更新信息全盘搜索".

FOUR STEPS, IN ORDER
  1. leaderboard snapshot  -> our rank / margin vs the bronze line, appended to rank_track
  2. submission ledger     -> refresh, append new rows, and re-check whether the final PAIR is
                              still co-failing (same substrate) or made of two different ones
  3. public sweep          -> newest public kernels AND datasets, flagging anything whose TITLE
                              claims a score above our best. Titles lie: "0946" was a verbatim
                              copy of a 0.943 notebook, and "0.957" was a LOCAL diagnostic score
                              while the same file says "as submitted (0.941 public)". So every
                              claim gets listed for verification, never believed.
  4. hand-off reminder     -> the C101/C102 pipeline for anything found: read the mechanism out
                              of the actual code -> write a SPEC -> move ONE variable -> measure
                              locally (870 instrument, +/-0.004) -> submit only if >= +0.005

Transport-failure aware: a Kaggle SSL blip must never be reported as a result (FACTS A173).
"""
import json
import pathlib
import re
import subprocess
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = pathlib.Path(__file__).resolve().parent.parent
COMP = "rsna-knee-abnormality-detection"
BEST = 0.944          # our best displayed public score; anything claiming more gets listed


def cli(*a, cwd=None):
    last = ""
    for attempt in range(3):
        try:
            p = subprocess.run(["kaggle", *a], capture_output=True, text=True,
                               encoding="utf-8", errors="replace",
                               cwd=str(cwd or ROOT))
            o = (p.stdout or "") + (p.stderr or "")
            if o.strip():
                return o
            last = o
        except Exception as e:                      # noqa: BLE001
            last = repr(e)
        time.sleep(15 * (attempt + 1))
    return "NETFAIL " + last


def step(n, title):
    print()
    print("=" * 92)
    print(f"STEP {n} -- {title}")
    print("=" * 92)


def run_script(rel):
    """Run a sibling script; return (rc, combined_output).

    A step that dies must NEVER look like a step that had nothing to say (A173/A174,
    and A176 where an SSL outage was reported as "STEP 1 printed nothing").
    """
    p = subprocess.run([sys.executable, str(ROOT / rel)], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", cwd=str(ROOT))
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def report_failure(label, rc, out):
    print(f"  !! {label} FAILED (rc={rc}) -- THIS IS A FAILURE, NOT A RESULT.")
    tail = [l.rstrip() for l in out.splitlines() if l.strip()][-4:]
    if tail:
        print("     last output was:")
        for l in tail:
            print("       " + l.strip()[:150])
    else:
        print("     (no output at all -- suspect a transport/SSL failure, retry later)")


def summary_line(out):
    """Parse the numbers out of lb_snapshot's prose instead of grep-ing keywords."""
    got = {
        "us": re.search(r"^\s*rank\s+(\d+)\s+score\s+([0-9.]+)", out, re.M),
        "teams": re.search(r"top 10% of\s+(\d+)", out),
        "bronze": re.search(r"bronze line[^=\n]*=\s*(\d+)", out),
        # NOTE the sign class must include "+": lb_snapshot prints "margin to bronze = +50"
        # once we are INSIDE the medal cut, and a bare "-?\d+" silently fails to match it.
        "margin": re.search(r"margin to bronze\s*=\s*([-+]?\d+)", out),
    }
    missing = [k for k, v in got.items() if v is None]
    if missing:
        print(f"  !! could not parse {missing} from lb_snapshot output -- showing raw tail:")
        for l in [x.rstrip() for x in out.splitlines() if x.strip()][-8:]:
            print("     " + l.strip()[:150])
        return
    rank, score = got["us"].group(1), got["us"].group(2)
    print(f"  US: rank {rank} score {score} | teams {got['teams'].group(1)}"
          f" | bronze <= {got['bronze'].group(1)} | margin {got['margin'].group(1)}")
    band = re.search(r"^\s*(0\.\d+):\s*n=\s*(\d+)\s+ranks\s+(\d+)\.\.(\d+)", out, re.M)
    if band:
        print(f"  top band: {band.group(1)} n={band.group(2)} ranks {band.group(3)}..{band.group(4)}")


def main():
    print(f"daily sweep at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")

    step(1, "leaderboard snapshot (rank / margin)")
    rc, out = run_script("work/lb_snapshot.py")
    if rc != 0 or not out.strip():
        report_failure("lb_snapshot.py", rc, out)
    else:
        summary_line(out)

    step(2, "submission ledger + final-pair check")
    rc, out = run_script("work/rsna_daily.py")
    if rc != 0 or not out.strip():
        report_failure("rsna_daily.py", rc, out)
    else:
        keep = False
        printed = 0
        for line in out.splitlines():
            if "current pair check" in line or "best = " in line:
                keep = True
            if keep and line.strip():
                print("  " + line.strip()[:150])
                printed += 1
        if not printed:
            print("  (script succeeded but printed no pair-check block -- check its output format)")
            for line in [x.rstrip() for x in out.splitlines() if x.strip()][-5:]:
                print("     " + line.strip()[:150])

    step(3, "public sweep -- kernels claiming more than %.3f" % BEST)
    k = cli("kernels", "list", "--competition", COMP, "--sort-by", "dateCreated",
            "--page-size", "25")
    if "NETFAIL" in k:
        print("  transport failure; skipped (NOT a result)")
    else:
        rows = [l for l in k.splitlines() if l.strip()][1:]
        print(f"  {len(rows)} newest kernels; titles mentioning a score:")
        hits = 0
        for l in rows:
            m = re.search(r"0[\.,]9(\d\d)", l)
            if not m:
                continue
            val = float("0.9" + m.group(1))
            flag = "  <== CLAIMS ABOVE OURS -- VERIFY" if val > BEST else ""
            print(f"    0.9{m.group(1)}  {l.split()[0][:52]:<52}{flag}")
            hits += 1
        if not hits:
            print("    (none -- no title claims a score today)")

    d = cli("datasets", "list", "-s", "rsna knee", "--sort-by", "updated")
    if "NETFAIL" in d:
        print("  datasets: transport failure; skipped")
    else:
        rows = [l for l in d.splitlines() if l.strip()][2:9]
        print("  newest datasets (first 7):")
        for l in rows:
            print("    " + l.strip()[:130])

    step(4, "for anything found, the pipeline is (C101/C102 precedent)")
    print("  1. read the mechanism OUT OF THE ACTUAL CODE -- not from the title or prose")
    print("  2. write a SPEC under docs/ with the verbatim implementation quotes")
    print("  3. move ONE variable; bundling three changes makes the number uninterpretable (C86)")
    print("  4. measure on the 870 instrument (+/-0.004); gold-58 is +/-0.021 and only good for")
    print("     absolute numbers (D16)")
    print("  5. submit ONLY if the local effect is >= +0.005")
    print()
    print("  CLOSED, do not reopen without new evidence: density (C90), geometry matching (C95),")
    print("  v8/v9 warm starts (C92/C96), per-finding routing (C97), 3D (A175), mirror TTA (C102).")


if __name__ == "__main__":
    main()
