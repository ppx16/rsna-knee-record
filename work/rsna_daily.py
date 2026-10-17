# -*- coding: utf-8 -*-
"""
The steady RSNA loop: one command that advances the position every day, unattended.

WHAT THIS IS FOR, AND WHAT IT HONESTLY CANNOT DO
    Measured, not assumed (FACTS C90/C95/C96/C97 + A175): the recipe/composition side is
    exhausted. Every lever we can reach is closed by measurement:
        density (44/64/80)     solo +0.0065 -> member +0.0007, CI crosses 0     (C90)
        geometry matching      matched arm scored LOWER than the mismatched one (C95)
        warm starts v8/v9      guard-refused at 0.8899 / 0.8907 < bar 0.8963    (C92/C96)
        per-finding routing    hindsight ORACLE ceiling only +0.0044            (C97)
        3D volume arm          0.65-0.78, far under the 0.90 member gate        (A175)
        reader weight axis     the whole range is 0.942-0.944, i.e. 0.002 wide, while the
                               leaderboard quantises at 0.001 -> 0.944 is its ceiling

    So this loop does NOT promise a higher score. It does the one thing that IS positive
    expected value and has never been done: KEEP THE BEST PAIR CURRENT AND DIVERSE, and
    keep a dated, machine-written record so a later session never has to re-derive it.

THE TWO JOBS IT DOES
    1. LEDGER  -- pull every submission with its score and description, append any new ones
                  to state/rsna_submissions.jsonl, and print the current best pair.
    2. PAIR    -- classify every submission into a structural family by what it CHANGES
                  (base stack / reader leg / sampling geometry / foreign model / routing),
                  then report whether the current pair is co-failing (same family) and name
                  the best cross-family alternatives.

    Rule it encodes, from docs/STRATEGY-2026-10-07-final-pair-diversity.md:
        the platform keeps our BEST submission, so trading a little public score for a
        second submission from a DIFFERENT family is positive expected value; the pair's
        only job is to not fail together on the private split.
"""
import json
import pathlib
import re
import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = pathlib.Path(__file__).resolve().parent.parent
LEDGER = ROOT / "state" / "rsna_submissions.jsonl"
COMP = "rsna-knee-abnormality-detection"

# structural families, keyed on what the description says was CHANGED. Order matters:
# the first pattern that matches wins, so the most specific family is listed first.
FAMILIES = [
    ("reader-leg",   r"2\.5D ConvNeXt reader|reader as a separate|reader leg"),
    ("sampling-geo", r"sparse|62/42|span|k_eval|finespacing|dense"),
    ("foreign-model", r"M448|ConvNeXt fallback|dinov2|DINOv2|specialist"),
    ("routing",      r"routing|route|rad_cal|parent"),
    ("base-stack",   r"0943|community stack"),
]


def family_of(desc: str) -> str:
    for name, pat in FAMILIES:
        if re.search(pat, desc, re.I):
            return name
    return "other"


# THE OTHER HALF, AND THE ONE THAT ACTUALLY DECIDES CO-FAILURE.
# `family_of` answers "which axis did this submission change". But two submissions on
# DIFFERENT axes can still die together if they rest on the SAME substrate -- and our whole
# history rests on the `0943` community stack plus our own members. So the pair check must
# look at the substrate too, not just the changed axis. (This corrected my own earlier
# claim in docs/STRATEGY-...: by changed-axis the current pair IS cross-family, reader-leg
# vs routing -- it is the shared substrate that makes it effectively one submission.)
SUBSTRATES = [
    ("0943s (sparse stack)", r"0943s"),
    ("0943 speedy", r"speedy"),
    ("0943cfmsft2(+m)", r"0943cfmsft2"),
    ("0943cf(+448/m)", r"0943cf"),
    ("0943c", r"0943c"),
    ("0943o/f/h/p/r", r"0943[ofhpr]"),
    ("0943 (base)", r"0943"),
    ("v2src", r"v2src"),
    ("ens-*", r"ens-"),
]


def substrate_of(desc: str) -> str:
    for name, pat in SUBSTRATES:
        if re.search(pat, desc, re.I):
            return name
    return "unknown"


# --- BLOODLINE: the axis that now DOMINATES the substrate axis -------------------------------
# Substrate decides whether two submissions are statistically correlated. Bloodline decides
# whether one of them can be VOIDED by an external decision (the host's pending ruling on
# OAI-trained checkpoints). A rules risk is not a correlation, so it is checked first.
# The boolean comes from state/final_pair.json (the single source of truth) -- NEVER from
# substring-matching a display label: "ours-nonOAI".upper() contains "OAI" (FACTS A184).
PAIR_CFG = ROOT / "state" / "final_pair.json"
OAI_DESC_MARKERS = ("nartaa coatnet", "apex 0.950 stack", "triple-reader stack",
                    "apex grandmaster", "sota apex")


def load_oai_map() -> dict:
    try:
        cfg = json.loads(PAIR_CFG.read_text(encoding="utf-8"))
        return {ref: bool(c.get("oai")) for ref, c in cfg.get("candidates", {}).items()
                if isinstance(c.get("oai"), bool)}
    except Exception as exc:  # noqa: BLE001
        print(f"  !! could not read {PAIR_CFG.name} ({exc}); falling back to description markers")
        return {}


def looks_oai(desc: str) -> bool:
    """Fallback heuristic ONLY, used when a ref is absent from final_pair.json."""
    low = desc.lower()
    return any(m in low for m in OAI_DESC_MARKERS)


def bloodline_report(scored: list, oai_map: dict) -> list[str]:
    """Pure function so the branch can be exercised by --selftest (an untested guard is A184)."""
    out = []
    out.append("=== BLOODLINE (checked first: a rules risk is not a correlation) ===")
    out.append(f"  {len(oai_map)} ref(s) carry an explicit `oai` flag in {PAIR_CFG.name}")
    oai_scored = [r for r in scored if r.get("oai")]
    out.append(f"  scored submissions that are OAI bloodline: {len(oai_scored)}")
    for r in oai_scored[:6]:
        out.append(f"    {r['score']:.3f}  {r['ref']}  {r['desc'][:70]}")
    if not oai_scored:
        out.append("    (none yet -- the two OAI candidates are still PENDING,"
                   " so they do not appear here)")
    for r in scored:
        if r["ref"] not in oai_map and looks_oai(r["desc"]):
            out.append(f"  !! {r['ref']} looks OAI by description but has no entry in "
                       f"{PAIR_CFG.name} -- add it, do not guess (FACTS A184)")

    best = scored[0]
    out.append(f"  best = {best['ref']} {best['score']:.3f}  axis={best.get('family')}  "
               f"substrate={best.get('substrate')}  oai={best.get('oai')}")
    if best.get("oai"):
        allowed = [r for r in scored[1:] if not r.get("oai")]
        out.append("  -> the best entry is OAI bloodline, so the SECOND tick MUST be non-OAI.")
        if allowed:
            out.append(f"     highest non-OAI candidate: {allowed[0]['ref']} "
                       f"{allowed[0]['score']:.3f} "
                       f"(costs -{best['score'] - allowed[0]['score']:.3f})")
        else:
            out.append("     *** NO non-OAI candidate with a score -- do not tick two OAI ***")
        out.append("  *** NEVER tick two OAI submissions: the platform keeps our BEST, so the")
        out.append("  *** pair's only job is to not co-fail -- and one ruling can void BOTH.")
    else:
        out.append("  -> the best entry is non-OAI, so pairing it with another non-OAI entry")
        out.append("     keeps us immune to the OAI ruling; pairing it with an OAI entry buys the")
        out.append("     score upside while keeping one leg safe. Both legal; two OAI are NOT.")
    return out


def selftest() -> int:
    """Exercise both branches with synthetic data. No network, no ledger writes."""
    cases = [
        ("best is non-OAI",
         [{"ref": "A", "score": 0.944, "oai": False, "family": "reader-leg", "substrate": "x", "desc": "ours"},
          {"ref": "B", "score": 0.943, "oai": False, "family": "base", "substrate": "y", "desc": "ours"}],
         {"A": False, "B": False}),
        ("best is OAI, a non-OAI option exists",
         [{"ref": "O", "score": 0.951, "oai": True, "family": "apex", "substrate": "z", "desc": "nartaa"},
          {"ref": "A", "score": 0.944, "oai": False, "family": "reader-leg", "substrate": "x", "desc": "ours"}],
         {"O": True, "A": False}),
        ("best is OAI and NO non-OAI option has a score",
         [{"ref": "O", "score": 0.951, "oai": True, "family": "apex", "substrate": "z", "desc": "nartaa"},
          {"ref": "P", "score": 0.950, "oai": True, "family": "apex", "substrate": "w", "desc": "triple"}],
         {"O": True, "P": True}),
    ]
    ok = True
    for name, scored, omap in cases:
        print("=" * 78)
        print("CASE:", name)
        for ln in bloodline_report(scored, omap):
            print("  " + ln)
        joined = "\n".join(bloodline_report(scored, omap))
        if name.startswith("best is OAI and NO"):
            if "do not tick two OAI" not in joined:
                print("  *** SELFTEST FAIL: the no-non-OAI case did not warn")
                ok = False
        if name.startswith("best is OAI, a non-OAI"):
            if "MUST be non-OAI" not in joined or "highest non-OAI" not in joined:
                print("  *** SELFTEST FAIL: did not point at the non-OAI candidate")
                ok = False
    print("=" * 78)
    print("SELFTEST:", "OK" if ok else "FAILED")
    return 0 if ok else 1


def fetch():
    p = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP, "--csv"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       cwd=str(ROOT))
    out = (p.stdout or "") + (p.stderr or "")
    if "SSLError" in out or "Max retries" in out:
        print("  transport failure talking to Kaggle; kept the existing ledger")
        return None
    rows, hdr = [], None
    for line in out.splitlines():
        if not line.strip() or line.startswith("Warning"):
            continue
        if hdr is None:
            hdr = line
            continue
        rows.append(line)
    return rows


def parse(row):
    """Header (verified from the CLI, not assumed):
           ref,fileName,date,description,status,publicScore,privateScore
       The description may or may not be quoted and may contain commas, so anchor on the
       LAST THREE comma-separated fields (none of which contain a comma) and let the greedy
       group take everything before them."""
    m = re.match(r'^(\d+),([^,]*),([^,]*),(.*),([^,]*),([^,]*),([^,]*)$', row)
    if not m:
        return None
    ref, fn, date, desc, status, pub, priv = m.groups()
    desc = desc.strip()
    if desc.startswith('"') and desc.endswith('"'):
        desc = desc[1:-1]
    desc = desc.replace('""', '"')
    return {"ref": ref, "file": fn, "date": date[:19], "desc": desc,
            "status": status.split(".")[-1],
            "score": float(pub) if pub.strip() else None,
            "private": float(priv) if priv.strip() else None}


def main():
    print("=== RSNA steady loop ===")
    rows = fetch()
    seen = set()
    if LEDGER.exists():
        for l in LEDGER.read_text(encoding="utf-8").splitlines():
            try:
                seen.add(json.loads(l)["ref"])
            except Exception:
                pass
    new = []
    if rows:
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with LEDGER.open("a", encoding="utf-8") as f:
            for r in rows:
                rec = parse(r)
                if not rec or rec["ref"] in seen:
                    continue
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                new.append(rec)
        print(f"  {len(rows)} submissions on Kaggle; {len(new)} new -> appended to "
              f"{LEDGER.relative_to(ROOT)}")
        for r in new:
            print(f"    NEW {r['ref']}  {r['score']}  [{family_of(r['desc'])}]  {r['desc'][:70]}")
    else:
        print("  could not refresh from Kaggle; ledger left as-is")

    recs = []
    if LEDGER.exists():
        for l in LEDGER.read_text(encoding="utf-8").splitlines():
            try:
                recs.append(json.loads(l))
            except Exception:
                pass
    scored = [r for r in recs if r.get("score")]
    if not scored:
        print("  no scored submissions in the ledger yet")
        return
    scored.sort(key=lambda r: -r["score"])
    for r in scored:
        r["family"] = family_of(r["desc"])

    print()
    print("=== top candidates by public score ===")
    print(f"  {'score':>6}  {'family':<14} {'ref':<10} what it changes")
    for r in scored[:10]:
        print(f"  {r['score']:6.3f}  {r['family']:<14} {r['ref']:<10} {r['desc'][:80]}")

    best = scored[0]
    oai_map = load_oai_map()
    for r in scored:
        r["substrate"] = substrate_of(r["desc"])
        if r["ref"] in oai_map:
            r["oai"] = oai_map[r["ref"]]
        else:
            r["oai"] = looks_oai(r["desc"])

    print()
    for line in bloodline_report(scored, oai_map):
        print("  " + line)

    print()
    print("=== current pair check (substrate: correlation within a bloodline) ===")
    print(f"  best = {best['ref']} {best['score']:.3f}  axis={best['family']}  "
          f"substrate={best['substrate']}  oai={best['oai']}")
    # the axis that matters for co-failure is the SUBSTRATE, not the changed axis
    diff_sub = [r for r in scored[1:] if r["substrate"] != best["substrate"]]
    same_sub = [r for r in scored[1:] if r["substrate"] == best["substrate"]]
    print(f"  candidates on the SAME substrate (co-failure risk) : {len(same_sub)}")
    print(f"  candidates on a DIFFERENT substrate                 : {len(diff_sub)}")
    if diff_sub:
        print("  DIFFERENT-SUBSTRATE alternatives (these are what make a pair worth having):")
        for r in diff_sub[:5]:
            print(f"    {r['score']:.3f}  axis={r['family']:<14} sub={r['substrate']:<20} "
                  f"{r['ref']}  costs -{best['score'] - r['score']:.3f}")
    else:
        print("  *** every submission shares one substrate -> the pair is effectively ONE")
        print("  *** submission. Creating a different-substrate recipe is the loop's task.")
    print()
    print("  RULE: the platform keeps our BEST, so the pair's only job is to not fail")
    print("  together on the private split -- and that is decided by the shared SUBSTRATE.")
    print("  See docs/STRATEGY-2026-10-07-final-pair-diversity.md")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    main()
