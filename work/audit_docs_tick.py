"""Find every document that names a submission ref next to tick language, and flag any that disagrees
with state/final_pair.json.

Why this exists: FINAL-TICK.md sat for six days still naming `57127191` (0.956) as slot 1 while
final_pair.json had moved to `57250245` (0.958). Nothing caught it; I found it by accident, two days before
the deadline. In this repository the failure mode is not a wrong number, it is a **stale document that a
human follows at 07:59 on the deadline morning**. The JSON is the single source of truth, so every other
mention has to be checked against it rather than trusted.

Usage: python work/audit_docs_tick.py [--all]
Exit 0 = no document contradicts the JSON. Exit 1 = at least one does.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAIR = ROOT / 'state' / 'final_pair.json'

# language that marks a ref as "the thing to tick"
TICKISH = re.compile(r'(勾选|tick|TICK|槽位|slot\s*[12]|Selected|①|②)')
REF = re.compile(r'\b(5\d{7})\b')
# a line that explains itself as history, negation, or an explicit fallback is not an instruction to follow
CONTEXT = re.compile(r'(此前|previous|superseded|历史|已过期|不要|不用|别|NOT\b|❌|⛔|退回|退|撤销|曾是|旧的|'
                     r'不可用|已检验|不触发|待定|pending|近似|几乎一样|almost|look-alike|'
                     r'do not|don\'t|never|failed|被证伪|candidate|并列候选|fallback|备用|替代)', re.I)


# The documents a human could plausibly act on at the deadline.  Everything else in docs/ is a dated
# snapshot, a narrative, or an analysis, and treating those as instructions produced so many false hits
# that the check became noise.  This list is deliberately short and explicit.
LIVE_DOCS = ('FINAL-TICK.md', 'DEADLINE-CHECKLIST.md', 'README.md')

DATED = re.compile(r'\d{4}-\d{2}-\d{2}')


def is_live(p: Path) -> bool:
    """Only the runbook set, plus the newest handover, counts as an instruction."""
    if p.name in LIVE_DOCS:
        return True
    if p.name.startswith('HANDOVER-') and p.name == 'HANDOVER-%s.md' % newest_handover():
        return True
    return False


def newest_handover() -> str:
    hs = sorted(p.name for p in ROOT.glob('HANDOVER-*.md'))
    if not hs:
        return ''
    return hs[-1].replace('HANDOVER-', '').replace('.md', '')


def main(argv: list[str]) -> int:
    show_all = '--all' in argv
    d = json.loads(PAIR.read_text(encoding='utf-8'))
    tick = d['tick']
    allowed = set(tick)

    md = sorted(list(ROOT.glob('*.md')) + list((ROOT / 'docs').glob('*.md')))
    print('tick per final_pair.json : %s' % ' + '.join(tick))
    print('documents scanned        : %d' % len(md))
    print()

    problems = 0
    contexts = 0
    mentions = 0
    for p in md:
        try:
            txt = p.read_text(encoding='utf-8', errors='replace')
        except Exception as exc:  # noqa: BLE001
            print('  UNREADABLE %s (%r)' % (p.name, exc))
            continue
        instr, ctx = [], []
        for line_no, line in enumerate(txt.splitlines(), 1):
            if not TICKISH.search(line):
                continue
            refs = [r for r in REF.findall(line) if r not in allowed]
            if not refs:
                continue
            mentions += len(refs)
            (ctx if CONTEXT.search(line) else instr).append((line_no, refs, line.strip()))
        if instr and not is_live(p):
            print('   dated snapshot  %-40s %d line(s) -- history by filename'
                  % (str(p.relative_to(ROOT)), len(instr)))
            instr = []
        if instr:
            problems += 1
            print('\n!! INSTRUCTION  %s' % p.relative_to(ROOT))
            for line_no, refs, line in instr[:6]:
                print('   L%-5d %s  %s' % (line_no, ','.join(refs), line[:150]))
        if ctx:
            contexts += 1
            print('   context      %-46s %d line(s) name other refs with a history/negation marker'
                  % (str(p.relative_to(ROOT)), len(ctx)))
        elif show_all:
            print('%-46s ok' % str(p.relative_to(ROOT)))

    print()
    print('other refs in history/negation context: %d refs across %d docs' % (mentions, contexts))
    if problems:
        print('RESULT: %d document(s) name a non-tick ref in an instruction-shaped line.' % problems)
        print('        Read each one: it is either a real stale instruction or needs a marker.')
        return 1
    print('RESULT: no document instructs ticking anything outside final_pair.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
