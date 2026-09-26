# -*- coding: utf-8 -*-
"""Verify that every `§N` citation in state/FACTS.md resolves to a HANDOFF heading.

WHY THIS EXISTS.  On 2026-09-26 an audit found that FACTS rows A108-A117 cite HANDOFF
sections §182-§191 that **do not exist in any commit** (`git log -S "## 191." --all` is
empty).  Ten citations, twenty-four missing sections: the narrative record for several
sessions was never written, and every atomic fact from A108 on pointed into the void.

That failure is exactly the class this repository keeps re-learning: a citation *looks*
present (it is a plausible number), and nothing in the workflow makes the absence loud.
Reading the file cannot catch it; only a mechanical comparison can.  So the comparison is
a script, in the same spirit as `git check-ignore -v --no-index` for ignore rules (A103).

A citation is a promise that the reader can go read the reasoning.  This checks the promise.

    python scripts/check_facts_refs.py            # exit 1 if any unexplained dangling ref
    python scripts/check_facts_refs.py --list     # print all refs and whether they resolve

ACCEPTED GAPS.  Reconstructed citations are listed in ACCEPTED with the reason and the
authoritative substitute source.  An accepted gap is not "fine" -- it is a debt that is
recorded where the checker will keep showing it, rather than hidden.
"""
import argparse
import collections
import pathlib
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = pathlib.Path(__file__).resolve().parent.parent

# ref -> reason it may stay dangling (must name the substitute source)
ACCEPTED = {
    "168": "never written; superseded -- the sessions that would have carried it are "
           "summarised in FACTS A108 and HANDOFF 193",
    "169": "never written; see FACTS A109 (metadata signal) + HANDOFF 193",
    "170": "never written; see FACTS A109/A112 + HANDOFF 193",
    "171": "never written; see FACTS A110 + HANDOFF 193",
    "172": "never written; see FACTS A111 (tuning-transfer curve) + HANDOFF 193",
    "173": "never written; see FACTS A111/A112 + HANDOFF 193",
    "174": "never written; see FACTS A112 (our fold-0 ensemble) + HANDOFF 193",
    "175": "never written; see FACTS A96 (submission discipline) + HANDOFF 193",
    "176": "never written; see FACTS A113 (repository reorganisation) + HANDOFF 193",
    "177": "never written; see FACTS A113/A114 (paths, kpaths.py) + HANDOFF 193",
    "178": "never written; see FACTS A114 + HANDOFF 193",
    "179": "never written; see FACTS A115 (who tuned on what) + HANDOFF 193",
    "180": "never written; see FACTS A116 (public sweep) + HANDOFF 193",
    "181": "never written; see FACTS A116/A117 + HANDOFF 193",
    "182": "never written; see FACTS A108 (champ config schema) + HANDOFF 193",
    "183": "never written; see FACTS A109 (metadata signal) + HANDOFF 193",
    "184": "never written; see FACTS A110 (first two readouts) + HANDOFF 192",
    "185": "never written; see FACTS A111 (tuning-transfer curve) + HANDOFF 193",
    "186": "never written; see FACTS A112 (our fold-0 ensemble) + HANDOFF 193",
    "187": "never written; see FACTS A113 (repository reorganisation) + HANDOFF 193",
    "188": "never written; see FACTS A114 (kpaths.py, 17 scripts) + HANDOFF 193",
    "189": "never written; see FACTS A115 (tuning provenance) + HANDOFF 193",
    "190": "never written; see FACTS A116 + HANDOFF 193",
    "191": "never written; see FACTS A117 (calibrator probe) + HANDOFF 193",
}

REF_RE = re.compile(r"§\s*(\d+(?:\.\d+)?)")
# `## 167. x`, `### 167.3 x`, `## §182 x`
HEAD_RE = re.compile(r"^#{1,6}\s+(?:§\s*)?(\d+(?:\.\d+)?)\s*[.\s、:：]")


def headings(path):
    out = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = HEAD_RE.match(line)
        if m:
            out.add(m.group(1))
    return out


def refs(path):
    """ref -> sorted list of 'lineno' where it is cited."""
    found = collections.defaultdict(list)
    for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        for m in REF_RE.finditer(line):
            found[m.group(1)].append(str(i))
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", default=str(ROOT / "state" / "FACTS.md"))
    ap.add_argument("--handoff", default=str(ROOT / "docs" / "HANDOFF.md"))
    ap.add_argument("--list", action="store_true", help="print every ref, not just the failures")
    a = ap.parse_args()

    have = headings(pathlib.Path(a.handoff))
    want = refs(pathlib.Path(a.facts))

    resolved, dangling = [], []
    for r in sorted(want, key=lambda s: [int(p) for p in s.split(".")]):
        (resolved if r in have else dangling).append(r)

    if a.list:
        for r in resolved:
            print("  ok       §%-8s (FACTS line %s)" % (r, ",".join(want[r])))
    unexplained = [r for r in dangling if r not in ACCEPTED]
    for r in dangling:
        tag = "accepted" if r in ACCEPTED else "DANGLING"
        print("%-9s §%-8s cited at FACTS line %s" % (tag, r, ",".join(want[r])))
        if r in ACCEPTED:
            print("            reason: %s" % ACCEPTED[r])

    print("\nrefs cited: %d   resolve: %d   dangling: %d   unexplained: %d"
          % (len(want), len(resolved), len(dangling), len(unexplained)))
    if unexplained:
        print("\nFAIL: %d citation(s) resolve nowhere and carry no recorded reason: %s"
              % (len(unexplained), ", ".join("§" + r for r in unexplained)))
        return 1
    print("OK: every citation either resolves to a HANDOFF heading or has a recorded reason.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
