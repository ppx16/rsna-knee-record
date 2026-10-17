"""Move TICK-1 to 57291788 (the 0.958 artifact that also carries the non-OAI S6 arm).

Same displayed score as 57250245 (0.958), so the choice cannot be made on the board.  The reasons it
is the better tick are about the PRIVATE split, which is what decides the medal:

  * 57291788 = 57250245's recipe PLUS nartaa's S6 ConvNeXt all-data model folded into the compact half
    at 0.40 -- the split the published analysis of this competition uses.  An independent model, so
    more diversity in the ensemble that the hidden half will judge;
  * that model's README states "No OAI data was used for this model", so this artifact spans both
    provenances instead of being purely OAI-lineage;
  * it scored 0.958 publicly, i.e. adding it cost nothing measurable (which is exactly what the
    published noise analysis predicts: SE ~0.003 on the public half).

Honest uncertainty is recorded in the JSON: the two artifacts are indistinguishable on the public
board, so this is a judgment about which one generalises, not a measurement.

Idempotent: refuses to run twice.
"""
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / 'state' / 'final_pair.json'

d = json.loads(P.read_text(encoding='utf-8'))
if d.get('tick') == ['57291788', '56878092']:
    raise SystemExit('REFUSING: the tick is already this pair')

d['candidates']['57291788'] = {
    "label": "TICK-1",
    "score": 0.958,
    "oai": True,
    "bloodline": ("nartaa CoAtNet-384 + 224-crop (OAI-supervised) + heliosli R6 ConvNeXt (flip TTA) "
                  "+ **nartaa S6 ConvNeXt all-data, which used no OAI data**"),
    "match": "First artifact that adds nartaa's S6 ConvNeXt ALL-DATA model",
    "unique_token": "S6 ConvNeXt ALL-DATA",
    "clashes_with": [],
    "why": ("SCORED 0.958 on 2026-10-17 -- the same displayed score as 57250245, and the only artifact we "
            "hold that carries an independent, non-OAI model inside an otherwise OAI-lineage ensemble. "
            "Built by runnning nartaa's S6 all-data checkpoint (0.950 alone, sha256 0c7297d5..., "
            "'No OAI data was used for this model') as a SECOND compact bundle through the same "
            "published pipeline and folding it into the compact half at 0.40, which is the split the "
            "published 0.957 recipe uses: compact = 0.60*rank(R6) + 0.40*rank(S6). Run verified: S6 "
            "stage PASS, arm E_prior, 3/3 scored, 0 fallbacks, and the log line '[s6] folded into the "
            "compact half at 0.4 over 3 studies'. Chosen over 57250245 for the PRIVATE split: more "
            "model diversity and both provenances, at no measurable public cost."),
}
d['tick'] = ['57291788', '56878092']
d['_tick_change_2026_10_17'] = {
    "old_tick": ["57250245", "56878092"],
    "why": ("57291788 scored the same 0.958 as 57250245, so the board cannot separate them; it wins on "
            "diversity for the hidden half (an independent model added at the published 0.40 share) and "
            "on provenance (that model used no OAI data)."),
    "the_control_that_pins_it": ("The same day's other three shots isolate the knobs: A224=0.50 -> 0.957, "
                                 "A224=0.65 -> 0.957, r9all leg @0.10 -> 0.957, while the plain base and "
                                 "the S6 version both give 0.958. So the S6 arm costs nothing measurable "
                                 "and the other two axes cost a thousandth."),
    "honest_uncertainty": ("The two 0.958 artifacts are indistinguishable on the public board; the "
                           "unrounded order inside a displayed band is not observable (C113/C141/C189). "
                           "This is a judgment about which one generalises to the private half, not a "
                           "measurement. If the S6 arm turned out to hurt the private split, tick 2 "
                           "still holds 0.944 and the exposure is one medal step, not the whole entry."),
    "still_oai": "The Raptor stages are still OAI-supervised, so 57291788 is OAI-lineage overall; tick 2 stays 56878092 (non-OAI).",
}
P.write_text(json.dumps(d, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print('tick ->', d['tick'])
