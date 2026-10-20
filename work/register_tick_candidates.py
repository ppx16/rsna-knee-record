"""Register today's 0.958 artifacts as tick candidates, with the reasoning that makes the choice auditable.

Context (FACTS C204): we now hold FOUR artifacts that display 0.958, and the public half cannot separate
them -- the band is 1313 teams wide and the published noise floor exceeds every gap inside it.  So the
deadline choice among them is a judgement, and the point of this file is to write the judgement down
BEFORE the deadline rather than invent it then.

The candidates, in the order the evidence supports:
  57250245  0.958  the author's recipe as published (per-finding weights, no extra views)   <- current tick 1
  57376589  0.958  + the anatomical mirror on the 224 arm (one more averaged view, measured neutral)
  57394596  0.958  + plain/TTA rank split on BOTH arms (the most averaged of the four)
  57313021  0.958  + nartaa's S6 all-data arm at 0.60 (an independent, non-OAI model)

Tie-break rule recorded here: prefer the artifact with more independent views AVERAGED, because with the
public score pinned at 0.958 the only thing that can still differ is the unrounded value, and averaging
independent views lowers its variance.  That argues for 57394596 first and 57376589 second.  But the rule
is a judgement, not a measurement, and it is overturned by any measurement -- so 57250245 stays the
default until we have one.

Idempotent: refuses to run twice.
"""
import json
from pathlib import Path

P = Path(__file__).resolve().parents[1] / 'state' / 'final_pair.json'
d = json.loads(P.read_text(encoding='utf-8'))
if '_candidates_note_2026_10_20' in d:
    raise SystemExit('REFUSING: already applied')

cands = d['candidates']
for ref, label, why in (
    ('57376589', 'M224-SPARE',
     '0.958. Same recipe as 57250245 plus the anatomical mirror on the 224 arm (C196), which the '
     'published flip notebook reports as an unchanged displayed score with a rank gain. Measured here as '
     '0.958, i.e. neutral in the display. One more averaged view than tick 1.'),
    ('57394596', 'P384-SPARE',
     '0.958. The most averaged of our artifacts: plain/TTA rank split on BOTH Raptor arms (C197+C198). '
     'Note the arms disagreed on their own -- the 224 split alone measured 0.957 (57377106) while the 384 '
     'split alone measured 0.958 (C204) -- so the combination sitting at 0.958 is not evidence that each '
     'half is sound. Listed as the preferred candidate under the tie-break rule below, not as a measured '
     'improvement.'),
):
    if ref not in cands:
        cands[ref] = {
            'label': label,
            'kernel': None,
            'public': '0.958',
            'oai': True,
            'why': why,
            'clashes_with': [],
        }

d['_candidates_note_2026_10_20'] = {
    'situation': ('Four artifacts display 0.958 and the public half cannot separate them: the 0.958 band '
                  'holds 1313 teams and the published noise floor exceeds every gap inside it (C184).'),
    'tie_break_rule': ('Prefer the artifact with more independent views averaged, because the displayed '
                       'score is pinned and only the unrounded value can differ; averaging independent '
                       'views lowers its variance. That ranks 57394596 first, then 57376589, then '
                       '57250245.'),
    'why_the_tick_did_not_move': ('The rule is a judgement, not a measurement. 57250245 is the artifact '
                                  'whose recipe is someone else\'s published work with no addition of '
                                  'ours, so it is the one we can defend without appealing to a guess.'),
    'what_would_move_it': ('A public asset above 0.958, or a measurement separating two of these -- '
                           'neither is available: the band is saturated and the hidden rerun is the only '
                           'place the unrounded value exists.'),
}
P.write_text(json.dumps(d, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print('candidates now:', ', '.join(sorted(d['candidates'])))
print('tick stays   :', d['tick'])
