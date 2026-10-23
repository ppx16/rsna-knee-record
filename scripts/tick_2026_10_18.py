"""Revert TICK-1 to 57250245 (the plain 0.958 artifact, no S6 arm) on external evidence.

Why (FACTS C195): an independent team measured this exact question and published the negative --
`goodpjw2008/rsna-knee-does-s6-help-next-to-r6-lb-0-957`, verbatim:

    "Version 4 showed that S6 next to R6 costs 0.001 (0.957 against 0.958 for the same recipe
     without it) ... this notebook is now two measured negatives: do not add S6 to R6, and do not
     tune per-finding weights on the 58 gold studies."

Our own three shots measured the S6 version at 0.958, i.e. no visible cost.  So the evidence is 3-to-1
"N/C" vs "minus a thousandth" -- but the DISPLAYED difference is inside the noise of a ~640-study half
(SE ~0.003, C184), and the downside is asymmetric: the only measured effect of adding S6 ranges from
zero to minus one thousandth, never positive.  A tick exists to not lose, so it goes back to the plain
artifact.  Nothing is discarded: the S6 kernels remain approved and scored, they are just not the one
we would tick.

The per-finding weights stay as heliosli has them: our flat-weight twin measured 0.957 (C195), i.e.
those weights are worth the thousandth publicly, and the external team's gold-58-tuned variants scored
0.957 / 0.955 -- three independent negatives for re-tuning them.

Idempotent: refuses to run twice.
"""
import json
from pathlib import Path

P = Path(__file__).resolve().parents[1] / 'state' / 'final_pair.json'
d = json.loads(P.read_text(encoding='utf-8'))
if d.get('tick') == ['57250245', '56878092']:
    raise SystemExit('REFUSING: the tick is already this pair')

d['candidates']['57291788']['label'] = 'S6-SPARE'
d['candidates']['57291788']['why'] = (
    d['candidates']['57291788']['why'] +
    '  [2026-10-18] DEMOTED from tick 1: the independent notebook '
    'goodpjw2008/rsna-knee-does-s6-help-next-to-r6 published that adding S6 to R6 costs 0.001 '
    '(0.957 vs 0.958 without it). Our own three S6 shots displayed 0.958, so the measurements '
    'disagree inside the noise, but the only effect ever measured for S6 is zero to minus a '
    'thousandth -- never positive -- so the tick goes to the artifact without it.')
d['candidates']['57250245']['label'] = 'TICK-1'
d['candidates']['57250245']['clashes_with'] = ['57127191', '57291788']
d['candidates']['57250245']['why'] = (
    d['candidates']['57250245']['why'] +
    '  [2026-10-18] RESTORED as tick 1: it is the same 0.958 recipe without the S6 arm, and the only '
    'independent measurement of that arm says it costs a thousandth.')
d['tick'] = ['57250245', '56878092']
d['_tick_change_2026_10_18'] = {
    "old_tick": ["57291788", "56878092"],
    "why": ("External replication. goodpjw2008 published 'does S6 help next to R6': version 4 with S6 "
            "scored 0.957 against 0.958 for the same recipe without it, and they conclude 'do not add "
            "S6 to R6'. Our three S6 shots displayed 0.958, so the two measurements disagree at the "
            "0.001 level -- inside the noise of a 640-study half -- but the asymmetry decides it: S6 "
            "has never measured positive, only zero or minus one thousandth."),
    "what_is_kept": ("Nothing is thrown away: 57291788 and 57313021 stay scored and approved as the "
                     "S6-SPARE. If the private split rewards the extra model, we can still tick it."),
    "the_weights_stay": ("The per-finding weights stay as heliosli set them: our flat twin scored 0.957 "
                         "(so they are worth the thousandth publicly) and the external team's "
                         "gold-58-tuned variants scored 0.957 / 0.955."),
    "still_oai": "Both artifacts are OAI-lineage (Raptor stages), so tick 2 stays 56878092 (non-OAI).",
}
P.write_text(json.dumps(d, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print('tick ->', d['tick'])
