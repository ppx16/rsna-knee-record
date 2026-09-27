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


# Rows that legitimately trip the interior-cell rule, with a reason -- the same idiom as ACCEPTED
# for citations.  Keep this EMPTY unless the row is provably not a swallowed row.
ROW_OK = {
    "A66": "pre-existing cosmetic artifact: A66 is a single cell that uses '<br>·' bullets and "
           "contains a literal interior '| §152.15 |' fragment carried over from that section's "
           "text.  A66 and A67 are both present and distinct, so nothing was swallowed; rewriting "
           "unrelated verbatim content is the edit class this repo avoids.",
}


def check_rows(path):
    """Assert the PROPERTY that a FACTS table row is one well-formed row -- not its layout. that a FACTS table row is one well-formed row -- not its layout.

    WHY THIS EXISTS.  On 2026-09-27 the same mistake was made twice within an hour: adding a row
    by REPLACING another row's prefix (A154 and A155 each ate the A122 prefix, in that order).
    Both times the citation checker was happy, because the survivor still cited its own `§` --
    the damage was that two rows became one and the swallowed row's facts silently merged in.

    ⚠️ THE FIRST VERSION OF THIS GUARD WAS WRONG, and is worth remembering.  It required every row
    to END with a `| §n |` cell.  That gave (a) 29 false positives, because real rows legitimately
    end with cells like `| 同上 |` or `| §81；§84.1 再次确认 |`, and (b) **no detection at all** of
    the bug it was written for: the glued row ends with the swallowed row's `| §194.3 |` anyway.
    A guard that is simultaneously too strict and blind is exactly the failure this repo keeps
    recording -- so the rule below is derived from the two states, not guessed:

        correct : | A154 | ... | §211 |            <- the §-cell is the LAST cell
                  | A122 | ... | §194.3 |          <- and the next row carries its own id
        broken  : | A154 | ... | §211 | ...prose of A122... | §194.3 |
                                                    ^ the §-cell is INTERIOR -> detectable

    ⇒ The rule: if a row contains a `| §n |` cell at all, that cell must END the row.  A row whose
    §-cell is followed by more text has swallowed its neighbour.
    """
    row_re = re.compile(r"^\| A[0-9]+[a-z]? \|")
    id_re = re.compile(r"^\| (A[0-9]+[a-z]?) \|")
    interior_re = re.compile(r"\|\s*§[0-9]+(?:\.[0-9]+)*\s*\|")
    bad, seen, dupes = [], {}, []
    for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if not line.startswith("| A"):
            continue
        if not row_re.match(line):
            bad.append((i, "row does not start with '| A<n> |'", line[:70]))
            continue
        m = id_re.match(line)
        if m:
            if m.group(1) in seen:
                dupes.append((i, m.group(1), seen[m.group(1)]))
            seen[m.group(1)] = i
        if m and m.group(1) in ROW_OK:
            continue
        for hit in interior_re.finditer(line):
            # Tolerate TRAILING EMPTY CELLS: 9 pre-existing rows end with `| §146.1 | |`, which is
            # harmless markdown.  What is NOT harmless is prose after the §-cell -- that is the
            # signature of a row that ate its neighbour.
            if line[hit.end():].strip(" \t|"):
                bad.append((i, "a '| §n |' cell is followed by more text -- this row looks like it "
                               "swallowed the next row's ID", line[-90:]))
                break
    for i, rid, first in dupes:
        bad.append((i, "duplicate row id %s (first seen on line %d)" % (rid, first), ""))
    return bad, sorted(seen)


def selftest():
    """Negative control for check_rows(): the guard must FIRE on the bug it exists for.

    A gate that has never been shown to fire is not a gate (this repo's own rule -- A169, A88).
    The cases below include the ones that broke the FIRST version of this guard, so they cannot
    silently come back: a row ending in a non-§ cell must PASS, and the glued row -- which still
    ends in a §-cell -- must FAIL.
    """
    import tempfile

    clean = ("| A1 | a fact | §1 |\n"
             "| A2 | ends with a non-section cell | 同上 |\n"
             "| A3 | ends with a compound cell | §81；§84.1 再次确认 |\n"
             "| A4 | ends with a trailing empty cell | §146.1 | |\n")
    glued = ("| A1 | a fact | §1 |\n"
             "| A3 | a NEW fact | §3 | another | §2 |\n")          # A2's id is gone
    dupe = "| A1 | a fact | §1 |\n| A1 | the same id again | §2 |\n"
    cases = [("clean file (non-§ final cells are legal)", clean, False),
             ("swallowed-row edit", glued, True),
             ("duplicate row id", dupe, True)]
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d)
        for n, (what, text, want_bad) in enumerate(cases):
            f = p / ("c%d.md" % n)
            f.write_text(text, encoding="utf-8")
            bad, _ids = check_rows(f)
            if bool(bad) != want_bad:
                print("SELFTEST FAIL (%s): expected bad=%s, got %s%s"
                      % (what, want_bad, bool(bad), (" -- " + bad[0][1]) if bad else ""))
                return 1
            print("  ok  %-45s bad=%-5s %s" % (what, bool(bad), bad[0][1] if bad else ""))
    print("SELFTEST OK: guard fires on both corruptions and passes legal rows")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", default=str(ROOT / "state" / "FACTS.md"))
    ap.add_argument("--handoff", default=str(ROOT / "docs" / "HANDOFF.md"))
    ap.add_argument("--list", action="store_true", help="print every ref, not just the failures")
    ap.add_argument("--selftest", action="store_true", help="prove the row guard fires")
    a = ap.parse_args()

    if a.selftest:
        rc = selftest()
        if rc:
            return rc

    bad_rows, row_ids = check_rows(pathlib.Path(a.facts))
    if bad_rows:
        print("FAIL: %d malformed FACTS row(s):" % len(bad_rows))
        for i, why, text in bad_rows:
            print("  line %-5d %s\n        %s" % (i, why, text))
        print("       (this is what a row-eating edit looks like -- restore the swallowed row)")
    print("FACTS rows: %d" % len(row_ids))

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
    if bad_rows:
        # A swallowed-row edit usually leaves the survivor's § intact, so the citation check
        # above passes while the table is corrupt.  Fail on the structural check too.
        print("\nFAIL: %d malformed FACTS row(s) -- see above." % len(bad_rows))
        return 1
    print("OK: every citation either resolves to a HANDOFF heading or has a recorded reason.")
    print("OK: all %d FACTS rows are well-formed (unique ids, no row has swallowed its neighbour)."
          % len(row_ids))
    return 0


if __name__ == "__main__":
    sys.exit(main())
