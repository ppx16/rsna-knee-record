"""Move TICK-1 to the 0.958 submission (2026-10-16) and declare the description clash.

The verifier flagged `57250245` as an UNDECLARED CLASH with `57127191`: both descriptions begin with
"byte-for-byte fork of the public n...", which is exactly the confusion the uniqueness check exists to
catch.  It is a true positive -- they ARE different notebooks -- so the clash is declared here and the
tick moves, because 57250245 scored a full displayed step higher (0.958 vs 0.956).

Tick 2 stays `56878092` (0.944, non-OAI): the OAI question is still unresolved, and the whole point of
the pair is that one entry survives an adverse OAI ruling.

Idempotent: refuses to run twice.
"""
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / 'state' / 'final_pair.json'

d = json.loads(P.read_text(encoding='utf-8'))
if d.get('tick') == ['57250245', '56878092']:
    raise SystemExit('REFUSING: the tick is already the 2026-10-16 pair')

new = {
    "label": "TICK-1",
    "score": 0.958,
    "oai": True,
    "bloodline": "nartaa CoAtNet-384 OAI-supervised (stage 1) + nartaa's independent 224-crop checkpoint (stage 1b) + heliosli R6 frozen ConvNeXt-small with flip TTA (stage 2)",
    "match": "Byte-for-byte fork of the public notebook heliosli/blend-mine-in-private-lb-luck-to-all-upvoters",
    "unique_token": "blend-mine-in-private-lb-luck-to-all-upvoters",
    "clashes_with": ["57127191"],
    "why": ("SCORED 0.958 on 2026-10-16 -- one full displayed step above the 0.956 we had held, "
            "and the first time our position has been comfortably inside BOTH medal lines "
            "(rank 231; bronze cut 608 -> +377; silver cut 304 -> +73). Kernel "
            "ppxl16/rsna-knee-0958-blend: a byte-for-byte fork (only edit: one non-ASCII character "
            "the Windows pusher cannot encode) of the highest-scoring public notebook, with all its "
            "inputs public -- the attached bundle dataset heliosli/rsna-knee-r6-frozen-fe055406-w1 is "
            "the single-version copy of the R6 bundle its manifest expects, so unlike the 0.9576 "
            "version it needs no substitution. Verified in the run: ASSETS_READY lists that dataset, "
            "the manifest bundle_id/sha256 assertions passed, OURS_STAGE_COMPLETE PASS 3/3 fallback 0, "
            "and the receipt carries a224=0.35. OAI-lineage (stage 1 is nartaa's OAI-supervised "
            "CoAtNet), so tick 2 must stay non-OAI."),
}
d['candidates']['57250245'] = new
old = d['candidates']['57127191']
old['label'] = 'OAI-SPARE'
old['clashes_with'] = ['57250245']
old['why'] = (old['why'] + '  [2026-10-16] Superseded as tick 1 by 57250245 (0.958, a full displayed '
              'step higher). Retained as the interchangeable OAI spare: it is still the strongest '
              'artifact we hold that does not depend on the newest public bundle.')
d['tick'] = ['57250245', '56878092']
d['_tick_change_2026_10_16'] = {
    "old_tick": ["57127191", "56878092"],
    "why": ("57250245 SCORED 0.958 -- one full displayed step above 0.956, and the census that day "
            "reported 'none above ours', i.e. we hold the reproducible public ceiling. Position moved "
            "from rank 565 (bronze +40, silver -263) to rank 231 (bronze +377, silver +73)."),
    "the_control_that_pins_the_cause": ("The same day, 57250300 = the SAME base plus the band32 leg at "
                                        "w=0.08 returned 0.957, and the same leg at w=0.15 returned "
                                        "0.956, while w=0.04 on the older base returned 0.956. So the "
                                        "+0.002 came from adopting the newest public base bundle "
                                        "(R6 + flip TTA + the per-finding reweight), NOT from the leg: "
                                        "on this base the added leg costs about a thousandth."),
    "honest_uncertainty": ("The per-finding reweight inside that notebook was changed the day before "
                           "and is fitted on the public half; the published analysis of this "
                           "competition puts the public half's standard error near 0.003, larger than "
                           "the whole spread of the top 400 teams. So part of the 0.958 may be public-LB "
                           "fitting. What is solid is that we now hold the same artifact as the "
                           "public ceiling, and the private split decides the medal."),
    "still_oai": "57250245 IS OAI-lineage (nartaa's OAI-supervised CoAtNet in stage 1), so tick 2 stays 56878092 (non-OAI).",
}
P.write_text(json.dumps(d, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print('tick ->', d['tick'])
print('candidates now:', sorted(d['candidates']))
