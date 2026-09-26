# -*- coding: utf-8 -*-
"""Push a one-variable experiment off the 0.941-class ensemble notebook.

Why this exists
---------------
The competition allows 5 submissions/day and now through 2026-10-22, so roughly 205
submissions remain. Every unresolved question in the public recipe - the arm-weight
vector, the RadImageNet exclude list, whether a fifth Raptor arm helps - is a
one-variable A/B, and doing those by hand-editing a 396 KB notebook is how you end up
with a result you cannot attribute.

So: each experiment is a named list of literal (old, new) replacements. The tool applies
them to the notebook JSON *as text*, asserts each one fired exactly once, compiles every
code cell in memory to prove the notebook still parses, reads the file back to prove the
edit actually persisted, and only then writes the push directory.

Two hard rules this tool encodes
--------------------------------
1. NEVER push a new version of a kernel whose submission is still PENDING. Pushing
   destroys/cancels the in-flight version's output, and only the latest version stays
   downloadable - so a careless push can turn a scored submission into nothing. Each
   experiment therefore gets its OWN kernel id (rsna-knee-ens-<name>), which is also what
   makes two experiments runnable at once.
2. NEVER let two experiments differ in more than the named edit. The edits are asserted
   against a pristine copy of the source notebook, and the produced notebook's hash is
   recorded, so "which notebook scored what" is answerable later.

Usage
-----
    python scripts/ens_experiment.py --list
    python scripts/ens_experiment.py --experiment exclude-fracture --dry-run
    python scripts/ens_experiment.py --experiment exclude-fracture --push
    python scripts/ens_experiment.py --experiment exclude-fracture --submit   # needs --push first

After a push, register the version with the director so its output is landed and scored:
    python scripts/director.py --record-push ppxl16/rsna-knee-ens-exclude-fracture:1
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
from kpaths import kdir          # kernels/ layout, see kpaths.py
SRC_DIR = kdir("kernel-ens")
SRC_NB = SRC_DIR / "rsna-knee-ensemble-v1.ipynb"
SRC_META = SRC_DIR / "kernel-metadata.json"
EXP_ROOT = kdir("kernel-exp")
COMPETITION = "rsna-knee-abnormality-detection"

# An experiment may fork a DIFFERENT upstream notebook.
#
# On 2026-09-11 maverickss26 refreshed the source we forked: 42 -> 51 cells. Two changes
# matter. It added an optional fifth Raptor view (`raptor_extra_arm_w`, off by default,
# with a ready-made `probe22+v8rev` preset at 0.10), and - more valuable - it grouped the
# Raptor arms by "volume recipe" so each study is decoded from DICOM once per recipe
# instead of once per arm. The author's words: "Same volumes, same windows, same models,
# same numbers." That is a pure speed-up, and speed is the one thing that could turn a
# scored submission into a timeout.
#
# The refreshed source is used for NEW work only. The original fork stays untouched
# because its submissions are still PENDING, and pushing a new version of a kernel with a
# pending submission can destroy that submission.
ALT_SOURCES = {
    "v2src": (kdir("kernel-fresh") / "rsna-knee-ens-v2src.ipynb",
              kdir("kernel-fresh") / "kernel-metadata.json"),
}


def source_paths(exp):
    key = exp.get("source")
    if not key:
        return SRC_NB, SRC_META
    if key not in ALT_SOURCES:
        raise SystemExit("ABORT: unknown source %r; known: %s"
                         % (key, ", ".join(sorted(ALT_SOURCES))))
    return ALT_SOURCES[key]

# ---------------------------------------------------------------------------
# Experiments. Each edit must match EXACTLY ONCE in the notebook text, or the run
# aborts - a silently-unapplied edit is the failure mode this repo has already been
# bitten by twice (see HANDOFF 12.1 and the generator ordering trap).
# ---------------------------------------------------------------------------
EXPERIMENTS = {
    "exclude-fracture": {
        "desc": ("Stop excluding Fracture from the RadImageNet blend. Every top public "
                 "notebook ships _RAD_EXCLUDE=(\"Baker's\",'Fracture'). An independent "
                 "audit (dariushafshar) measured Fracture as 4th of 12 in blend lift "
                 "(+0.0077) while Baker's goes negative on gold-58; flipping only this "
                 "line scored 0.921 vs 0.920 on the real LB (ref 55654006). +0.001 there, "
                 "unverified on the 0.941 stack."),
        "edits": [
            ('_RAD_EXCLUDE = ("Baker\'s", \'Fracture\')',
             '_RAD_EXCLUDE = ("Baker\'s",)'),
        ],
    },
    "arm-weights-alt": {
        "desc": ("Swap the Raptor arm-weight vector from 0.55/0.10/0.15/0.20 to "
                 "0.60/0.10/0.10/0.20. The public sources CONTRADICT each other on this: "
                 "evgendvorkin credits the 0.941 step with one vector and maverickss26 "
                 "claims 0.941 with the other. A genuine +/-0.001 unknown, and the "
                 "cheapest possible A/B."),
        # The four Raptor arms are declared together; the exact literal is located at
        # build time so a reformat cannot silently no-op this experiment.
        "edits": [],
        "special": "arm_weights_alt",
    },
    "halfway-routing": {
        "desc": ("Pull the per-target outer CoAtNet routing halfway back to the flat 0.60 "
                 "it was tuned from (the notebook's own `halfway` preset). The author "
                 "warns in the source that weights fitted to the public split can do real "
                 "damage privately; this measures whether the +0.002 survived."),
        "edits": [
            ('PRESET = os.environ.get("RSNA_PRESET", "probe22")',
             'PRESET = "halfway"'),
        ],
    },
    "parent-routing": {
        "desc": ("Drop the per-target outer routing entirely and use the flat 0.60 parent "
                 "that scored 0.939. This is the control for `halfway-routing`. It was "
                 "written as 'if flat 0.60 scores >= probe22, the per-target tuning was "
                 "fitting noise' - but that criterion CANNOT FIRE, and the measurement "
                 "came out 0.939 < 0.941 anyway: both numbers are read off the SAME public "
                 "split, probe22's is a maximum over 22 probes (hence inflated ~+0.002), "
                 "and the unbiased gap is ~+0.0005, which a 3-decimal instrument cannot "
                 "resolve. Do not cite this run as 'the routing was validated' (HANDOFF "
                 "53, 57.6)."),
        "edits": [
            ('PRESET = os.environ.get("RSNA_PRESET", "probe22")',
             'PRESET = "parent"'),
        ],
    },
    "add-widedense-arm": {
        "desc": ("Add a fifth Raptor arm: raptor_ft_coatnet_v4_full.pt from "
                 "dreaddevelopment/raptor-knee-widedense, at a real weight. This is the "
                 "ONLY extra checkpoint whose view is both documented and non-redundant. "
                 "Config copied from a working notebook (sankurero/rsna-knee-coatnet-"
                 "raptor-only), not inferred: IMG 336, _SLOTS64, span (0.06, 0.94), "
                 "k_eval 62, res from the checkpoint. The four shipped arms are "
                 "(336,64,.02-.98) x2, (384,64,.02-.98), (384,44,.06-.94) - none of them "
                 "is (336,64,.06-.94), so this is a genuinely new view. The standalone "
                 "widedense notebook reports 0.924 public LB for it alone. The original "
                 "four keep their exact ratios, scaled by 0.8, so this isolates the "
                 "addition rather than reshuffling the blend."),
        "edits": [],
        "special": "add_widedense_arm",
        "dataset_sources_add": ["dreaddevelopment/raptor-knee-widedense"],
    },
    "add-finespacing-arm": {
        "desc": ("Add the v9 `finespacing` view as a fifth Raptor arm: "
                 "raptor_ft_coatnet_v9_full.pt from dreaddevelopment/raptor-knee-finespacing. "
                 "WHY THIS ONE, and not another view: (a) its geometry is DOCUMENTED, not "
                 "inferred -- the dataset description states 80 slots (22/18/15/10/15) over "
                 "2-98 percent, 140 mm crop, 336 px, and k_eval 78; no public notebook carries "
                 "a v9 ARMS entry, so without that description this would have been a guess. "
                 "(b) It is the STRONGEST single model in the family: 0.932 public LB solo, "
                 "vs 0.926/0.927 for the maxspan view we already ship, and its checkpoint "
                 "reports 0.9031 on the 58 official labels (above the ~0.90 harm line). "
                 "(c) The author explicitly names the pairing -- 'pair it with the wide span "
                 "and wide dense checkpoints for a diversity boost' -- i.e. our v5 and v10. "
                 "(d) It sits at the END of the sampling axis we measured as the only large "
                 "lever: the author's own numbers are 44 slices 0.926 -> 64 0.928 -> 80 0.932, "
                 "and his wide-span-44 result (0.917) independently reproduces our gap0 "
                 "(-0.012). See FACTS C47/C48. The four shipped arms keep their exact ratios "
                 "scaled by 0.8, so this isolates the addition."),
        "edits": [],
        "special": "add_finespacing_arm",
        "dataset_sources_add": ["dreaddevelopment/raptor-knee-finespacing"],
    },
    "add-finespacing-heavy": {
        "desc": ("THE DOSE TEST for v9 `finespacing`. The first attempt gave it 0.20 and read "
                 "0.941 (flat, D14) -- but the author's OWN score table says v9 (80 slices, "
                 "1.20% of stack per slice) scores 0.932 against maxspan-v5's 0.927, while our "
                 "shipped weights give maxspan 0.55 and v9 nothing. So the ensemble "
                 "under-weights its best view by a lot, and 0.20 may simply have been too "
                 "small a dose to show. This gives v9 0.50 and scales the shipped four to "
                 "0.50 in their exact original ratios (0.275/0.075/0.05/0.10) -- the same "
                 "shape n942 (titled 0.942) uses, where `0.5 * arms + 0.5 * raptor`. "
                 "One submission, ~0.5 GPU-h, and it is the only untested composition with "
                 "author data behind it."),
        "edits": [],
        "special": "add_finespacing_heavy",
        "dataset_sources_add": ["dreaddevelopment/raptor-knee-finespacing"],
    },
    # -----------------------------------------------------------------------------
    # Combinations. Only worth pushing once the components have scored on the LB and
    # both moved the same way - a combo of two inert edits is just a wasted slot.
    # -----------------------------------------------------------------------------
    "combo-fracture-widedense": {
        "desc": ("Both single-variable changes at once: stop excluding Fracture from the "
                 "RadImageNet blend AND add the widedense-v4 Raptor arm. Build only if "
                 "`exclude-fracture` and `add-widedense-arm` each scored >= the baseline; "
                 "otherwise this just stacks two unknown deltas."),
        "compose": ["exclude-fracture", "add-widedense-arm"],
    },
    "lean-3arm": {
        "desc": ("INSURANCE, not an improvement. Drop `maxspan-v5-reverse` - the same "
                 "checkpoint as arm 0 with a horizontal flip, so the most redundant entry "
                 "in the table - and renormalise the survivors. Cuts roughly a quarter of "
                 "the Raptor stage's wall clock. Push this ONLY if the full notebook turns "
                 "out to exceed the 9-hour scoring limit: every variant of it would fail "
                 "together, and a lean one is then the only way to get any score at all. "
                 "NOTE: the refreshed upstream (see `v2src`) makes this largely obsolete - "
                 "it decodes each study once per volume recipe instead of once per arm."),
        "edits": [],
        "special": "drop_arm",
    },
    # -----------------------------------------------------------------------------
    # Work built on the refreshed upstream (2026-09-11, 42 -> 51 cells).
    # -----------------------------------------------------------------------------
    "v2src-default": {
        "desc": ("The refreshed upstream with nothing changed - the CONTROL. It exists to "
                 "separate two things the refresh did at once: the arm-grouping rewrite "
                 "(advertised as numerically identical) and the new optional fifth view. "
                 "If `v2src-v8rev` differs from the old baseline, this tells us whether the "
                 "rewrite alone moved anything."),
        "source": "v2src",
        "edits": [],
    },
    "v2src-v8rev": {
        "desc": ("THE author's own new lever: `probe22+v8rev` enables a fifth Raptor view "
                 "- the native384-v8 checkpoint read BACK TO FRONT - at weight 0.10. The "
                 "upstream comment calls 0.10 'a sane first try'. It is nearly free because "
                 "the reversed view shares a volume recipe with native384-v8, so the "
                 "arm-grouping rewrite decodes the study once for both. This is a view "
                 "nobody in the public ecosystem has reported scoring, and it is the only "
                 "lever found so far that is BOTH new and cheap."),
        "source": "v2src",
        "edits": [
            ('PRESET = os.environ.get("RSNA_PRESET", "probe22")',
             'PRESET = "probe22+v8rev"'),
        ],
    },
    "v2src-denserev": {
        "desc": ("Complete the back-to-front set. `reverse` is `windows.flip(1)` - it "
                 "reverses the order of the three adjacent slices forming a window - and "
                 "upstream clearly believes in it: the 0.940->0.941 step raised "
                 "maxspan-v5-reverse from 0.10 to 0.15, and the refresh added exactly such "
                 "a twin for native384-v8. `native384dense-v10` is the ONE arm with no "
                 "reversed counterpart, so this adds `native384dense-v10-reverse` at 0.10. "
                 "The refreshed module groups arms by volume recipe, so the twin shares the "
                 "base arm's decode and costs GPU passes only."),
        "source": "v2src",
        "edits": [],
        "special": "add_reverse_arm",
    },
    "v2src-private25": {
        "desc": ("Probe `coat_private_alpha` DOWN. The full RUN config has ten fusion "
                 "weights; upstream's own evolution only ever moved two of them "
                 "(`coatnet_w` worth +0.002, and `raptor_view_w`). The other six are at the "
                 "values they were first typed in at, so `coat_private_alpha` - the share "
                 "of the private residual-gated CoAt e4/e6/e8 INSIDE the Raptor arm - has "
                 "never been tuned. It is the highest-leverage untested weight available: "
                 "the Raptor arm carries 0.60-1.00 of the final blend, and 40% of that arm "
                 "is this one component. Paired with `v2src-private55` to bracket the "
                 "current 0.40."),
        "source": "v2src",
        "edits": [('"coat_private_alpha": 0.40,', '"coat_private_alpha": 0.25,')],
    },
    "v2src-private55": {
        "desc": ("Probe `coat_private_alpha` UP, 0.40 -> 0.55. The other half of the "
                 "bracket around the never-tuned value: if the private residual-gated CoAt "
                 "is stronger than its 40% share assumes, this direction gains; if the "
                 "axis is flat, BOTH directions read as no rank movement and the weight can "
                 "be left alone for the rest of the competition."),
        "source": "v2src",
        "edits": [('"coat_private_alpha": 0.40,', '"coat_private_alpha": 0.55,')],
    },
    "v2src-a5w30": {
        "desc": ("Next axis in the never-tuned region: `a5_w`, the share of the A5 folds "
                 "(mattiaangeli's DINOv3 five-fold stage) against the DINO base inside the "
                 "transformer stack. 0.45 is the value it was first typed in at and "
                 "upstream's evolution never revisited it. The transformer stack carries "
                 "0.00-0.40 of the final blend (inverse of `coatnet_w`), so this is the "
                 "second-highest-leverage untested weight. Probes DOWN to 0.30; paired with "
                 "`v2src-a5w60`. Read the RANK, not the score."),
        "source": "v2src",
        "edits": [('"a5_w": 0.45,', '"a5_w": 0.30,')],
    },
    "v2src-a5w60": {
        "desc": ("The other half of the `a5_w` bracket, 0.45 -> 0.60. If A5 is stronger "
                 "than its 45% share assumes this gains; if both directions read as no rank "
                 "movement the axis is flat and `a5_w` can be left alone."),
        "source": "v2src",
        "edits": [('"a5_w": 0.45,', '"a5_w": 0.60,')],
    },
    # ------------------------------------------------------------------------------
    # The three never-tuned fusion weights the pre-registered four axes do NOT cover
    # (HANDOFF 57).  RUN has ten weights; upstream's own evolution only ever moved two
    # (`coatnet_w`, `raptor_view_w`).  Of the remaining eight, the four axes above
    # bracket a5_w / rad_alpha / rad_cal_w / coat_private_alpha - and these three were
    # in NO plan at all, not even the reserve list of HANDOFF 37.4 (which names TTA
    # pooling knobs, not RUN weights).
    #
    # All three are CONSUMED, not dead config (HANDOFF 33.2: cell 44 lines 40/41/42),
    # and all three sit in cell 44's RadImageNet branch - the second architecture that
    # HANDOFF 18's per-label gain table found to be doing real work.  Each is still at
    # the value it was first typed in at, including `rad_twin_alt_w`'s tell-tale
    # 0.500001.
    #
    # WHY THEY ARE NAMED NOW, BEFORE ANY AXIS HAS SCORED: the argument for adding them
    # is COMPLETENESS - bracket every never-tuned fusion weight, so that "the search is
    # exhausted" has a precise meaning instead of "four axes happened to read flat".
    # The same three axes proposed after a flat result would be indistinguishable from
    # cherry-picking, so they go in the plan first and get pushed only when the axes
    # ahead of them are done.
    #
    # BUILT BUT NOT PUSHED, same policy as the radcal/radalpha pairs.
    # ------------------------------------------------------------------------------
    "v2src-rade13w35": {
        "desc": ("Bracket `rad_e13_member_w` (0.50) DOWN to 0.35. Share of the e13 RadImageNet "
                 "member inside the radiomics arm. 0.50 is the value it was first typed in "
                 "at; upstream never revisited it. Paired with `v2src-rade13w65`. Read the "
                 "RANK, not the score."),
        "source": "v2src",
        "edits": [('"rad_e13_member_w": 0.50,', '"rad_e13_member_w": 0.35,')],
    },
    "v2src-rade13w65": {
        "desc": ("The other half of the `rad_e13_member_w` bracket, 0.50 -> 0.65. If the e13 "
                 "member is worth more than an even split this direction gains; if both "
                 "directions read as no rank movement the weight can be left alone."),
        "source": "v2src",
        "edits": [('"rad_e13_member_w": 0.50,', '"rad_e13_member_w": 0.65,')],
    },
    "v2src-radsec05": {
        "desc": ("Bracket `rad_second_alpha` (0.15) DOWN to 0.05 - nearly switching the second "
                 "RadImageNet pass off, which is the informative end for a weight this small. "
                 "The bracket is deliberately NOT the symmetric +/-0.15 used on the 0.40-0.50 "
                 "weights: at 0.15 that would run to 0.00 and test only 'is it needed at all', "
                 "while 0.05..0.30 spans 'almost off' to 'twice as strong'. Paired with "
                 "`v2src-radsec30`."),
        "source": "v2src",
        "edits": [('"rad_second_alpha": 0.15,', '"rad_second_alpha": 0.05,')],
    },
    "v2src-radsec30": {
        "desc": ("The other half of the `rad_second_alpha` bracket, 0.15 -> 0.30 (twice the "
                 "never-tuned value). If the second RadImageNet pass matters more than its 15% "
                 "share assumes this gains; otherwise the weight stays as first typed."),
        "source": "v2src",
        "edits": [('"rad_second_alpha": 0.15,', '"rad_second_alpha": 0.30,')],
    },
    "v2src-radtwin35": {
        "desc": ("Bracket `rad_twin_alt_w` (0.500001) DOWN to 0.35. The leading 0.500001 is "
                 "itself the evidence that this value was never tuned - it is a first-typed "
                 "number nudged off an exact tie. Paired with `v2src-radtwin65`."),
        "source": "v2src",
        "edits": [('"rad_twin_alt_w": 0.500001,', '"rad_twin_alt_w": 0.35,')],
    },
    "v2src-radtwin65": {
        "desc": ("The other half of the `rad_twin_alt_w` bracket, 0.500001 -> 0.65. If the "
                 "twin-alt member deserves more than an even split this direction gains; if "
                 "both directions read as no rank movement the axis is flat."),
        "source": "v2src",
        "edits": [('"rad_twin_alt_w": 0.500001,', '"rad_twin_alt_w": 0.65,')],
    },
    # ==============================================================================
    # THE ROUTING AXIS, WIDENED AND DEEPENED   (HANDOFF 67.3 / 69)
    #
    # `coatnet_w` = per-finding weight of the CoAtNet/Raptor arm against the transformer
    # stack.  Three points measured on OUR pipeline:
    #     0.60 flat (parent) -> 0.939 | halfway -> 0.940 | probe22 -> 0.941
    # Monotone and evenly spaced, and only the 100% point was ever selected on the public
    # split - so the gradient across the OTHER points is not a selection artefact.  It is
    # the first unbiased evidence that the routing gain is real (HANDOFF 67.3).
    #
    # That gradient says "more CoAtNet is better" and exposes two directions the author's
    # 22 probes never covered:
    #   (1) WIDEN.  Probe #22 raised only FIVE findings.  The other SEVEN - MCL, Medial OA,
    #       PF OA, Effusion, Synovitis, Baker's, Contusion - still sit at the parent's flat
    #       0.60 and have NEVER been moved.  Macro-AUC weights all twelve labels equally,
    #       so seven untouched columns is the largest untested area in the RUN config.
    #   (2) DEEPEN.  Four of the raised findings are below 1.00 and the measured response
    #       is still rising at 100%, so they can be pushed toward the ceiling the author
    #       already chose for Lateral Meniscus.
    #
    # Both are NEW configurations, so their public readings are unbiased - and they answer
    # "can we gain?" and "is the routing real?" with the same submissions.
    # Built but not pushed.
    # ==============================================================================
    "v2src-routew75": {
        "desc": ("ROUTING, WIDENED. `coatnet_w.__default__` 0.60 -> 0.75, moving the SEVEN "
                 "findings upstream's 22 probes never touched (MCL, Medial OA, PF OA, "
                 "Effusion, Synovitis, Baker's, Contusion) toward the CoAtNet arm. The "
                 "measured dose-response says more CoAtNet is better and macro-AUC weights "
                 "all twelve labels equally, so the untouched seven are the largest untested "
                 "area in the RUN config. Paired with `routew85`."),
        "source": "v2src",
        "edits": [('"coatnet_w": {"__default__": 0.60,', '"coatnet_w": {"__default__": 0.75,')],
    },
    "v2src-routew85": {
        "desc": ("ROUTING, WIDENED FURTHER. Same seven findings, `__default__` 0.60 -> 0.85. "
                 "Second point of the widening bracket: if `routew75` gains this says whether "
                 "to keep going; if `routew75` is flat this bounds the direction."),
        "source": "v2src",
        "edits": [('"coatnet_w": {"__default__": 0.60,', '"coatnet_w": {"__default__": 0.85,')],
    },
    "v2src-routedeep": {
        "desc": ("ROUTING, DEEPENED. Push the four non-maximal raised findings toward the "
                 "1.00 ceiling the author already chose for Lateral Meniscus: ACL 0.75 -> "
                 "0.875, Medial Meniscus 0.80 -> 0.90, Lateral OA 0.75 -> 0.875, Fracture "
                 "0.75 -> 0.875 (Lateral Meniscus stays 1.00). Extrapolates along the only "
                 "direction with a measured positive gradient (HANDOFF 67.3)."),
        "source": "v2src",
        "edits": [('"ACL": 0.75, "Medial Meniscus": 0.80,',
                   '"ACL": 0.875, "Medial Meniscus": 0.90,'),
                  ('"Lateral OA": 0.75, "Fracture": 0.75},',
                   '"Lateral OA": 0.875, "Fracture": 0.875},')],
    },
    # ==============================================================================
    # PER-LABEL WINDOW POOLING  (HANDOFF 74)  -- INFERENCE ONLY, NO TRAINING
    #
    # `apply_target_window_pool` overrides the window->study aggregation ONLY for labels
    # present in the map; every other label keeps the plain all-window mean (jitter
    # included).  The shipped map sets EIGHT of twelve:
    #     max            Fracture, Contusion, Medial Meniscus, Lateral Meniscus, Baker's
    #     top2           ACL, MCL
    #     original_mean  Synovitis   <- deliberately reverted to the NO-JITTER mean
    # FOUR are left at the default: Medial OA, Lateral OA, PF OA, Effusion.
    #
    # WHY THAT LOOKS UNTUNED: `Baker's` and `Effusion` are BOTH joint-fluid findings, and
    # `Baker's` is 'max' while `Effusion` keeps the plain mean.  Same clinical object, two
    # different aggregations - at least one was never chosen deliberately.
    #
    # These cost one submission and ~12 minutes of GPU, with NO training - the cheapest
    # experiments left in the project (HANDOFF 63: push -> land ~12 min, concurrency 2).
    #
    # PRIOR WARNING: the last time I extrapolated a per-label map in the "untouched labels"
    # direction - `coatnet_w` widening, HANDOFF 69 - the result was WORSE (0.940 vs 0.941).
    # Treat these as tests, not as expected gains.
    # ==============================================================================
    "v2src-poolfluidmax": {
        "desc": ("POOLING / FLUID. Make `Effusion` aggregate like `Baker's` - the other joint-fluid "
                 "finding - by adding it to the per-label pool map as 'max' instead of leaving it "
                 "at the plain all-window mean. Single-variable; nothing else changes."),
        "source": "v2src",
        "edits": [("\"Baker's\": 'max'}", "\"Baker's\": 'max', 'Effusion': 'max'}")],
    },
    "v2src-pooloamax": {
        "desc": ("POOLING / CARTILAGE. The three OA labels are the only cartilage targets and, with "
                 "Effusion, the only ones left at the default pooling. Cartilage loss is regional, "
                 "like the meniscal signal already pooled 'max'/'top2' - so push the three OA labels "
                 "to 'top2' (mean of the two strongest windows)."),
        "source": "v2src",
        "edits": [("\"Baker's\": 'max'}",
                   "\"Baker's\": 'max', 'Medial OA': 'top2', 'Lateral OA': 'top2', 'PF OA': 'top2'}")],
    },
    "v2src-poolefforig": {
        "desc": ("POOLING / FLUID, the OTHER side of the same bracket. `Synovitis` was explicitly "
                 "pulled back to 'original_mean' - the NO-JITTER mean - which is evidence that "
                 "jittered windows hurt at least one diffuse finding. `Effusion` is diffuse too and "
                 "keeps the jittered mean. This gives it the exact treatment Synovitis received. "
                 "Paired with `poolfluidmax`: one of the two directions, or neither, is the answer."),
        "source": "v2src",
        "edits": [("'Synovitis': 'original_mean'}",
                   "'Synovitis': 'original_mean', 'Effusion': 'original_mean'}")],
    },
    "v2src-w6505": {
        "desc": ("The OTHER direction on the arm-weight axis, taken from an author rather "
                 "than extrapolated. Diffing mattiaangeli's refreshed notebook (2026-09-12) "
                 "against the release shows the two upstream authors moved in OPPOSITE "
                 "directions on this one axis: the V17 release was 0.60/0.10, maverickss26 "
                 "(our 0.941 baseline) went to 0.55/0.15, and mattiaangeli's current build "
                 "went to 0.65/0.05. So `w5050` is not the only plausible next step - this is "
                 "the mirror step, and it is an actual author's choice rather than my "
                 "extrapolation. With `arm-weights-alt` (0.60/0.10) and the canonical "
                 "0.55/0.15, three submissions map the axis. BUILT BUT NOT PUSHED."),
        "source": "v2src",
        "edits": [
            ('"raptor_view_w": {"maxspan-v5": 0.55, "native384dense-v10": 0.10,\n'
             '                      "maxspan-v5-reverse": 0.15, "native384-v8": 0.20},',
             '"raptor_view_w": {"maxspan-v5": 0.65, "native384dense-v10": 0.10,\n'
             '                      "maxspan-v5-reverse": 0.05, "native384-v8": 0.20},'),
        ],
    },
    "v2src-radalpha35": {
        "desc": ("Fourth and LAST pre-registered axis (HANDOFF 29.4): `rad_alpha`, the weight "
                 "given to the RadImageNet head against the DINO baseline inside the "
                 "transformer stack. 0.50 has never been revisited by upstream, and the "
                 "audit in HANDOFF 18.1 measured a large spread in how much this second "
                 "architecture helps across labels - so a single global 0.50 is a plausible "
                 "place to leave value on the table. BUILT BUT NOT PUSHED: push only if the "
                 "first three axes all read flat, because a bracket on a dead axis is a slot "
                 "burned for nothing."),
        "source": "v2src",
        "edits": [('"rad_alpha": 0.50,', '"rad_alpha": 0.35,')],
    },
    "v2src-radalpha65": {
        "desc": ("Other half of the `rad_alpha` bracket, 0.50 -> 0.65. Built but not pushed, "
                 "same reason as `v2src-radalpha35`."),
        "source": "v2src",
        "edits": [('"rad_alpha": 0.50,', '"rad_alpha": 0.65,')],
    },
    "v2src-radcal25": {
        "desc": ("Third pre-registered axis: `rad_cal_w`, the weight of the RadImageNet "
                 "calibrator (0.40, never revisited by upstream). BUILT BUT NOT PUSHED - the "
                 "plan tests one axis at a time so the rank reading stays attributable "
                 "(HANDOFF 29.4), and a bracket whose axis is already known flat is a "
                 "submission slot burned for nothing. Push only if `a5_w` reads flat."),
        "source": "v2src",
        "edits": [('"rad_cal_w": 0.40,', '"rad_cal_w": 0.25,')],
    },
    "v2src-radcal55": {
        "desc": ("Other half of the `rad_cal_w` bracket, 0.40 -> 0.55. Built but not pushed, "
                 "same reason as `v2src-radcal25`."),
        "source": "v2src",
        "edits": [('"rad_cal_w": 0.40,', '"rad_cal_w": 0.55,')],
    },
    "v2src-radcal00": {
        # 2026-09-14: read out of the CURRENT public frontier, not invented here.
        # `mattiaangeli/bend-the-knee-to-the-dinosaurs` (the notebook our fork came from, 215
        # votes, last commit 2026-09-14) now carries, as a module constant,
        #     _RAD_CAL_W = 0.00
        # feeding the SAME expression our fork uses in cell[30]:
        #     (1.0 - _RAD_CAL_W) * values[:, index] + _RAD_CAL_W * adjusted[:, index]
        # while our fork reads RUN['rad_cal_w'] = 0.40.  Same knob, same semantics, different
        # value -- and 0.00 sits OUTSIDE the 0.25..0.55 bracket we built and never pushed.
        # So the frontier has DISABLED the RadImageNet calibration branch and we still run it
        # at 40%.  This is the cheapest untested point we have.
        "desc": ("`rad_cal_w` 0.40 -> 0.00, i.e. switch the RadImageNet calibration branch OFF. "
                 "Value taken from the current public frontier (HANDOFF 87). Everything else "
                 "identical to the canonical recipe."),
        "source": "v2src",
        "edits": [('"rad_cal_w": 0.40,', '"rad_cal_w": 0.00,')],
    },
    "v2src-seedalt": {
        # NOT a candidate to promote -- a REPRODUCIBILITY PROBE.
        # The kernel's jitter passes are budget-driven (HANDOFF 89.2) and the jitter seeds derive
        # from SEED, so two runs of the SAME recipe are not bit-identical.  Every reading we take
        # on this competition is 3 decimals, so the question "how much does our own pipeline move
        # between runs?" is exactly the question that decides whether a 0.940-vs-0.941 reading
        # means anything.  Changing SEED (2026 -> 2027) leaves the recipe identical and measures
        # only that.
        "desc": ("Reproducibility probe: SEED 2026 -> 2027, recipe otherwise identical. Measures "
                 "how much our own submission pipeline moves run-to-run, which is the scale "
                 "against which the radcal00 / radcal00-parent readings must be judged."),
        "source": "v2src",
        "edits": [("SEED = 2026", "SEED = 2027")],
    },
    "v2src-radcal00-parent": {
        # The FOURTH corner of the 2x2 that HANDOFF 100.1 describes: the fusion layer has exactly
        # two structural switches we know how to turn off --
        #   (1) the per-target outer CoAtNet routing  -> `PRESET = "parent"` (flat 0.60)
        #   (2) the RadImageNet calibration branch    -> `rad_cal_w` 0.40 -> 0.00
        # The canonical has both ON.  `parent-routing` (0.939) has (1) off; `radcal00` has (2) off.
        # This variant turns BOTH off, which completes the 2x2 and is the only way to see whether
        # the two switches interact -- with two final-selection slots we can only ever pick two of
        # the four corners, so knowing all four shapes the endgame choice (HANDOFF 100.3).
        #
        # PRE-FLIGHT (done, because a silently-overridden edit is this project's most expensive
        # failure mode, HANDOFF 16.1): the preset block in cell[00] is
        #     if   PRESET == "parent":        RUN["coatnet_w"] = {"__default__": 0.60}
        #     elif PRESET == "halfway":       RUN["coatnet_w"] = {...}
        #     elif PRESET == "probe22+v8rev": RUN["raptor_extra_arm_w"] = 0.10
        # -- i.e. NO preset touches `rad_cal_w`, so the two edits below cannot fight each other.
        "desc": ("Both structural switches OFF: flat 0.60 outer routing AND `rad_cal_w` 0.00. "
                 "Completes the 2x2 with the canonical / parent-routing / radcal00 (HANDOFF 100.1)."),
        "source": "v2src",
        "edits": [('PRESET = os.environ.get("RSNA_PRESET", "probe22")', 'PRESET = "parent"'),
                  ('"rad_cal_w": 0.40,', '"rad_cal_w": 0.00,')],
    },
    "v2src-w5050": {
        "desc": ("Extrapolate the one weight axis with direct LB evidence. The 0.940->0.941 "
                 "step moved 0.05 of weight from maxspan-v5 onto its back-to-front twin "
                 "(0.60/0.10/0.10/0.20 -> 0.55/0.10/0.15/0.20). This takes the next equal "
                 "step in the same direction: 0.50/0.10/0.20/0.20. If back-to-front "
                 "averaging is what earned that +0.002, another 0.05 should still be "
                 "positive; if the axis is flat or already past its optimum, this reads as "
                 "no movement (rank) and the axis can be abandoned cheaply."),
        "source": "v2src",
        "edits": [
            ('"raptor_view_w": {"maxspan-v5": 0.55, "native384dense-v10": 0.10,\n'
             '                      "maxspan-v5-reverse": 0.15, "native384-v8": 0.20},',
             '"raptor_view_w": {"maxspan-v5": 0.50, "native384dense-v10": 0.10,\n'
             '                      "maxspan-v5-reverse": 0.20, "native384-v8": 0.20},'),
        ],
    },
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def apply_edits(nb: dict, edits, label: str) -> dict:
    """Apply literal replacements to CELL SOURCE, never to the raw .ipynb text.

    The notebook on disk is JSON, so every quote inside a cell is escaped and a
    raw-text search for `PRESET = os.environ.get("RSNA_PRESET", "probe22")` finds
    nothing. Editing cell source is also the only way to keep the JSON valid without
    re-escaping by hand.
    """
    for i, (old, new) in enumerate(edits):
        hits = []
        for cell in nb["cells"]:
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell.get("source", []))
            n = src.count(old)
            if n:
                hits.append((cell, n))
        total = sum(n for _, n in hits)
        if total != 1:
            raise SystemExit(
                "ABORT %s: edit %d matched %d time(s) across %d cell(s), expected exactly 1.\n"
                "  old=%r\n"
                "  A miss here means the notebook was reformatted and this experiment "
                "would have silently done nothing - which is worse than failing."
                % (label, i, total, len(hits), old[:140]))
        cell = hits[0][0]
        src = "".join(cell["source"]).replace(old, new)
        cell["source"] = src.splitlines(keepends=True)
        print("  edited: %r -> %r" % (old[:70], new[:70]))
    return nb


WIDEDENSE_ARM = {
    "name": "widedense-v4",
    "file": "raptor_ft_coatnet_v4_full.pt",
    "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",
    "res": 384,
    "img": 336,
    "slots": "_SLOTS64",
    "span": (0.06, 0.94),
    "k_eval": 62,
    "reverse": False,
    "w": 0.20,
}
# The shipped four keep their exact ratios; scaling all of them by the same factor is what
# makes this "add an arm" rather than "reshuffle the blend".
WIDEDENSE_SCALE = 0.8


def _arms_from_module(module_src: str):
    """Parse the embedded RAPTOR module and return its ARMS list as real Python objects.

    This is the verification that matters: the arm table lives inside a quoted string, so
    a text edit can look right and still be wrong. Parsing the module back and evaluating
    the literal is the only check that proves the run will see five arms.

    ARMS is not a pure literal - each entry has `"slots": _SLOTS64` - so the slot tables
    are pulled out first and bound as the only two names in the eval namespace.
    """
    tree = ast.parse(module_src)
    wanted = {"_SLOTS64", "_SLOTS44", "_SLOTS80", "ARMS"}
    found = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            name = getattr(target, "id", None)
            if name in wanted:
                found[name] = node.value
    if "ARMS" not in found:
        raise SystemExit("ABORT: no top-level ARMS assignment in the embedded module")
    env = {}
    for name in ("_SLOTS64", "_SLOTS44", "_SLOTS80"):
        if name in found:
            env[name] = ast.literal_eval(found[name])
    return eval(compile(ast.Expression(found["ARMS"]), "<ARMS>", "eval"),  # noqa: S307
                {"__builtins__": {}}, env)


def special_add_widedense_arm(nb: dict) -> dict:
    """Append a fifth Raptor arm and rescale the shipped four to keep their ratios."""
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "ARMS = [" not in src:
            continue
        # The module is a single-quoted Python string inside the cell, so every newline in
        # it is the two characters backslash + n. Anchor on the last shipped arm and the
        # closing bracket of ARMS.
        marker = '"reverse": False, "w": 0.20},\\n]'
        if src.count(marker) != 1:
            raise SystemExit("ABORT add-widedense-arm: the ARMS terminator %r matched %d "
                             "times, expected 1" % (marker, src.count(marker)))
        new_entry = (
            '"reverse": False, "w": %s},\\n'
            '    {"name": "%s", "file": "%s",\\n'
            '     "arch": "%s", "res": %d,\\n'
            '     "img": %d, "slots": %s, "span": (%s, %s), "k_eval": %d,\\n'
            '     "reverse": False, "w": %s},\\n]'
            % (WIDEDENSE_ARM["w"], WIDEDENSE_ARM["name"], WIDEDENSE_ARM["file"],
               WIDEDENSE_ARM["arch"], WIDEDENSE_ARM["res"], WIDEDENSE_ARM["img"],
               WIDEDENSE_ARM["slots"], WIDEDENSE_ARM["span"][0], WIDEDENSE_ARM["span"][1],
               WIDEDENSE_ARM["k_eval"], WIDEDENSE_ARM["w"])
        )
        src = src.replace(marker, new_entry)
        cell["source"] = src.splitlines(keepends=True)
        print("  inserted 5th arm %s (w=%s)" % (WIDEDENSE_ARM["name"], WIDEDENSE_ARM["w"]))
        break
    else:
        raise SystemExit("ABORT add-widedense-arm: no cell defines ARMS")

    # Rescale the shipped four. Same name-keyed rewrite as arm-weights-alt, so a reorder
    # of the arm table cannot silently retarget this.
    targets = {"maxspan-v5": 0.55, "native384dense-v10": 0.10,
               "maxspan-v5-reverse": 0.15, "native384-v8": 0.20}
    pattern = re.compile(
        r'("name":\s*"(?P<name>[a-z0-9\-]+)".*?"reverse":\s*(?:True|False),\s*"w":\s*)'
        r'(?P<w>[0-9.]+)', re.S)
    for _ in range(12):
        progressed = False
        for cell in nb["cells"]:
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell.get("source", []))
            if "ARMS = [" not in src:
                continue
            for m in pattern.finditer(src):
                name = m.group("name")
                if name not in targets:
                    continue
                want = "%.2f" % (targets[name] * WIDEDENSE_SCALE)
                if m.group("w") == want:
                    continue
                src = src[:m.start("w")] + want + src[m.end("w"):]
                cell["source"] = src.splitlines(keepends=True)
                print("  rescaled %-20s w: %s -> %s" % (name, m.group("w"), want))
                progressed = True
                break
            if progressed:
                break
        if not progressed:
            break

    module_src = extract_module(nb)
    arms = _arms_from_module(module_src)
    print("  verified ARMS from the parsed module: %d arms" % len(arms))
    for a in arms:
        print("     %-20s img=%s slots=%s span=%s k_eval=%s w=%s"
              % (a["name"], a.get("img"), getattr(a.get("slots"), "__len__", lambda: "?")(),
                 a.get("span"), a.get("k_eval"), a.get("w")))
    total = sum(float(a["w"]) for a in arms)
    if len(arms) != 5 or abs(total - 1.0) > 1e-6:
        raise SystemExit("ABORT: expected 5 arms summing to 1.0, got %d summing to %s"
                         % (len(arms), total))
    if arms[-1]["name"] != WIDEDENSE_ARM["name"]:
        raise SystemExit("ABORT: the fifth arm is %r" % arms[-1]["name"])
    return nb


# --------------------------------------------------------------------------- #
# v9 `finespacing` -- the third leg the author tells you to ensemble.
#
# GEOMETRY PROVENANCE (this is why the experiment is worth a submission at all):
#   `dreaddevelopment/raptor-knee-finespacing`'s dataset DESCRIPTION states the corpus
#   "samples 80 slices per study across 2 to 98 percent of each series, cropped to a
#   140 mm box ... resized to 336 pixels", filled "22 ... sagittal fluid sensitive,
#   18 ... second sagittal, 15 ... coronal fluid sensitive, 10 ... second coronal and
#   15 ... axial", and that "Scoring uses 78 windows per study, which is every window
#   position an 80 slice volume holds."
#   No public notebook carries a v9 ARMS entry (all six 0.941 forks list the same four
#   arms), so without that description the only options would have been guessing -- which
#   is the silent-no-op failure class (HANDOFF 16.1). The description removes the guess.
#   The SAME description also warns the window count is not separable from the weights:
#   the identical checkpoint read at 62 windows "is not getting the geometry it was
#   trained on, and the 0.932 will not be there". So k_eval MUST be 78, not the 62 the
#   shipped four use.
#   And it names the pairing: "pair it with the wide span and wide dense checkpoints for a
#   diversity boost" -- i.e. exactly the v5 `maxspan` and v10 `native384dense` we already ship.
# --------------------------------------------------------------------------- #
FINESPACING_ARM = {
    "name": "finespacing-v9",
    "file": "raptor_ft_coatnet_v9_full.pt",
    "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",
    "res": 384,
    "img": 336,
    "slots": "_SLOTS80",
    "span": (0.02, 0.98),
    "k_eval": 78,
    "reverse": False,
    "w": 0.20,
}
# Same 0.8 as the widedense arm and for the same reason: scaling all four shipped arms by
# one factor keeps their exact ratios, so this is "add a view" and not "reshuffle the blend".
FINESPACING_SCALE = 0.8
SLOTS80_SRC = ('_SLOTS80 = [("Sagittal", 1, 22), ("Sagittal", 0, 18),\n'
               '            ("Coronal", 1, 15), ("Coronal", 0, 10), ("Axial", -1, 15)]\n')
ARM_W_PATTERN = re.compile(
    r'("name":\s*"(?P<name>[a-z0-9\-]+)".*?"reverse":\s*(?:True|False),\s*"w":\s*)'
    r'(?P<w>[0-9.]+)', re.S)


def _rescale_named_arms(module_src: str, targets: dict, scale: float) -> tuple:
    """Scale the named arms' weights in the embedded module, name-keyed.

    Name-keyed so that a reorder of the arm table cannot silently retarget the rewrite.
    Returns (new_module_src, {name: "old -> new"}).
    """
    changes = {}

    def _sub(m):
        name = m.group("name")
        if name not in targets:
            return m.group(0)
        want = "%.2f" % (targets[name] * scale)
        if m.group("w") != want:
            changes[name] = "%s -> %s" % (m.group("w"), want)
        return m.group(1) + want

    return ARM_W_PATTERN.sub(_sub, module_src), changes


def special_add_finespacing_arm(nb: dict, w: float | None = None,
                                scale: float | None = None) -> dict:
    """Append the v9 `finespacing` view (80 slots / 78 windows) as a fifth Raptor arm.

    `w` is the weight given to v9 and `scale` shrinks the shipped four by the same factor, so
    the vector still sums to 1.0. Exposing both is what makes the DOSE testable: the shipped
    weights give 0.55 to `maxspan-v5`, while the author's own table scores v9 (80 slices)
    **0.932** against maxspan's 0.927 -- so the ensemble under-weights its best view, and
    0.20 (the first attempt, flat at 0.941) may simply have been too small a dose.
    """
    if w is None:
        w = float(FINESPACING_ARM["w"])
    if scale is None:
        scale = FINESPACING_SCALE
    module_src = extract_module(nb)
    arms = _arms_from_module(module_src)
    if any(a["name"] == FINESPACING_ARM["name"] for a in arms):
        raise SystemExit("ABORT add-finespacing-arm: %r already present"
                         % FINESPACING_ARM["name"])

    # (1) the 80-slot table the v9 volume recipe needs does NOT exist in the module.
    #     Referencing it without inserting it would NameError *inside the kernel*, which
    #     is exactly the failure mode `scripts/check_names.py` exists to catch.
    m = re.search(r"(?m)^_SLOTS44 = \[[^\]]*\]\n", module_src)
    if m is None:
        raise SystemExit("ABORT add-finespacing-arm: no `_SLOTS44 = [...]` line to anchor")
    module_src = module_src[:m.end()] + SLOTS80_SRC + module_src[m.end():]

    # (2) the arm itself, appended to ARMS before its closing bracket.
    start = module_src.index("ARMS = [")
    end = module_src.index("\n]", start) + 2
    entry = (
        '    {"name": "%s", "file": "%s",\n'
        '     "arch": "%s", "res": %d,\n'
        '     "img": %d, "slots": %s, "span": (%s, %s), "k_eval": %d,\n'
        '     "reverse": False, "w": %s},\n'
        % (FINESPACING_ARM["name"], FINESPACING_ARM["file"], FINESPACING_ARM["arch"],
           FINESPACING_ARM["res"], FINESPACING_ARM["img"], FINESPACING_ARM["slots"],
           FINESPACING_ARM["span"][0], FINESPACING_ARM["span"][1],
           FINESPACING_ARM["k_eval"], w)
    )
    module_src = module_src[:end - 1] + entry + module_src[end - 1:]

    # (3) keep the shipped four in their exact ratios while making room for the fifth.
    shipped = {"maxspan-v5": 0.55, "native384dense-v10": 0.10,
               "maxspan-v5-reverse": 0.15, "native384-v8": 0.20}
    module_src, changes = _rescale_named_arms(module_src, shipped, scale)
    for name in sorted(changes):
        print("  rescaled %-20s w: %s" % (name, changes[name]))

    # (4) verify by parsing the module BACK -- the arm table lives in a quoted string, so
    #     a text edit can look right and still be wrong. This is the check that matters.
    after = _arms_from_module(module_src)
    if [a["name"] for a in after] != [a["name"] for a in arms] + [FINESPACING_ARM["name"]]:
        raise SystemExit("ABORT add-finespacing-arm: table reads %s" % [a["name"] for a in after])
    new = after[-1]
    slots = new.get("slots")
    if len(slots) != 5 or sum(s[2] for s in slots) != 80:
        raise SystemExit("ABORT add-finespacing-arm: v9 slots read %r" % (slots,))
    if new["k_eval"] != 78:
        raise SystemExit("ABORT add-finespacing-arm: v9 k_eval is %r, MUST be 78 -- the "
                         "author states the window count is not separable from the weights"
                         % (new["k_eval"],))
    if new["span"] != FINESPACING_ARM["span"] or new["file"] != FINESPACING_ARM["file"]:
        raise SystemExit("ABORT add-finespacing-arm: v9 recipe reads %r" % (new,))
    total = sum(float(a["w"]) for a in after)
    if len(after) != 5 or abs(total - 1.0) > 1e-6:
        raise SystemExit("ABORT add-finespacing-arm: %d arms, weights sum to %s"
                         % (len(after), total))
    print("  added v9 %s: slots=80 k_eval=78 img=%s span=%s w=%s"
          % (new["name"], new["img"], new["span"], new["w"]))
    return replace_module(nb, module_src)


def extract_module(nb: dict) -> str:
    """Return the source text of `_KE_SRC`, the embedded RAPTOR module."""
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "_KE_SRC" not in src:
            continue
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Assign) and any(
                    getattr(t, "id", None) == "_KE_SRC" for t in node.targets):
                return ast.literal_eval(node.value)
    raise SystemExit("ABORT: could not find the _KE_SRC module in the notebook")


def replace_module(nb: dict, new_module_src: str) -> dict:
    """Splice `new_module_src` back in as the `_KE_SRC` literal, in place.

    Editing the module as Python and re-embedding it is far safer than doing regex surgery
    on the escaped cell text: `repr()` reproduces exactly the single-quoted, backslash-n
    escaped form the notebook already uses, so the cell stays valid JSON and stays
    readable. The replacement is located by AST position, then read back and compared.
    """
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "_KE_SRC" not in src:
            continue
        tree = ast.parse(src)
        target = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                    getattr(t, "id", None) == "_KE_SRC" for t in node.targets):
                target = node.value
                break
        if target is None:
            continue
        lines = src.splitlines(keepends=True)
        offsets, pos = [], 0
        for ln in lines:
            offsets.append(pos)
            pos += len(ln)

        def abspos(lineno, col):
            return offsets[lineno - 1] + col

        start = abspos(target.lineno, target.col_offset)
        end = abspos(target.end_lineno, target.end_col_offset)
        new_src = src[:start] + repr(new_module_src) + src[end:]
        cell["source"] = new_src.splitlines(keepends=True)

        readback = None
        for node in ast.walk(ast.parse(new_src)):
            if isinstance(node, ast.Assign) and any(
                    getattr(t, "id", None) == "_KE_SRC" for t in node.targets):
                readback = ast.literal_eval(node.value)
                break
        if readback != new_module_src:
            raise SystemExit("ABORT: the re-embedded _KE_SRC does not read back identically")
        if _arms_from_module(readback) != _arms_from_module(new_module_src):
            raise SystemExit("ABORT: the re-embedded module's ARMS table changed")
        return nb
    raise SystemExit("ABORT: no cell assigns _KE_SRC")


def special_drop_arm(nb: dict, arm_name: str = "maxspan-v5-reverse") -> dict:
    """Remove one Raptor arm and renormalise the rest, for a cheaper run.

    Insurance, not an improvement: `maxspan-v5-reverse` is the same checkpoint as arm 0
    with a horizontal flip, so it is the most redundant arm in the table. Dropping it cuts
    roughly a quarter of the Raptor stage's wall clock. That matters only if the full
    notebook turns out to exceed the 9-hour limit at scoring time - in which case every
    variant of it fails together and a lean one is the only way to get a score at all.
    """
    module_src = extract_module(nb)
    arms_before = _arms_from_module(module_src)

    pattern = re.compile(
        r'\n?\s*\{"name":\s*"%s".*?"reverse":\s*(?:True|False),\s*"w":\s*[0-9.]+\},'
        % re.escape(arm_name), re.S)
    hits = pattern.findall(module_src)
    if len(hits) != 1:
        raise SystemExit("ABORT drop-arm: entry %r matched %d times, expected 1"
                         % (arm_name, len(hits)))
    new_module_src = pattern.sub("", module_src, count=1)

    arms_after = _arms_from_module(new_module_src)
    kept = [a for a in arms_before if a["name"] != arm_name]
    if len(arms_after) != len(kept) or [a["name"] for a in arms_after] != [a["name"] for a in kept]:
        raise SystemExit("ABORT drop-arm: after removal the table reads %r, expected %r"
                         % ([a["name"] for a in arms_after], [a["name"] for a in kept]))

    # Renormalise the survivors so the blend still sums to 1 without relying on the
    # module's own normalisation to hide a weight that no longer means what it says.
    total = sum(float(a["w"]) for a in arms_after)
    for a in arms_after:
        want = "%.4f" % (float(a["w"]) / total)
        old = '"name": "%s"' % a["name"]
        idx = new_module_src.index(old)
        wpos = new_module_src.index('"w":', idx)
        endpos = new_module_src.index("}", wpos)
        new_module_src = (new_module_src[:wpos] + '"w": ' + want
                          + new_module_src[endpos:])
    final = _arms_from_module(new_module_src)
    print("  dropped arm %r; %d arms remain: %s"
          % (arm_name, len(final), {a["name"]: a["w"] for a in final}))
    if abs(sum(float(a["w"]) for a in final) - 1.0) > 1e-6:
        raise SystemExit("ABORT drop-arm: weights sum to %s"
                         % sum(float(a["w"]) for a in final))
    return replace_module(nb, new_module_src)


def special_add_reverse_arm(nb: dict, base_name: str = "native384dense-v10",
                            new_name: str = None, weight: float = 0.10) -> dict:
    """Append a `reverse=True` twin of an existing arm.

    `reverse` in the Raptor module is `windows.flip(1)` - it reverses the ORDER of the three
    adjacent slices that make up a window's channels, i.e. "read back to front". Two pieces
    of upstream behaviour say this direction is worth extending:
      * the 0.940 -> 0.941 step raised `maxspan-v5-reverse` from 0.10 to 0.15, and
      * the 2026-09-11 refresh added exactly this kind of twin for `native384-v8`.
    `native384dense-v10` is the one remaining arm with no reversed counterpart, so this
    completes the set. Because the refreshed module groups arms by volume recipe, the twin
    shares the base arm's decode and costs GPU passes only - not a second pass over DICOM.

    The new entry is written as a plain dict literal (not `{**base, ...}`) so that
    `_arms_from_module` can still evaluate the table and prove the arm landed.
    """
    new_name = new_name or (base_name + "-reverse")
    module_src = extract_module(nb)
    arms = _arms_from_module(module_src)
    base = next((a for a in arms if a["name"] == base_name), None)
    if base is None:
        raise SystemExit("ABORT add-reverse-arm: no arm named %r (have %s)"
                         % (base_name, [a["name"] for a in arms]))
    if any(a["name"] == new_name for a in arms):
        raise SystemExit("ABORT add-reverse-arm: %r already present" % new_name)

    start = module_src.index("ARMS = [")
    end = module_src.index("\n]", start) + 2
    slots_name = "_SLOTS64" if len(base["slots"]) == 5 and sum(s[2] for s in base["slots"]) == 64 \
        else "_SLOTS44"
    entry = (
        '    {"name": "%s", "file": "%s",\n'
        '     "arch": "%s", "res": %d,\n'
        '     "img": %d, "slots": %s, "span": (%s, %s), "k_eval": %d,\n'
        '     "reverse": True, "w": %s},\n'
        % (new_name, base["file"], base["arch"], base["res"], base["img"], slots_name,
           base["span"][0], base["span"][1], base["k_eval"], weight))
    new_module_src = module_src[:end - 1] + entry + module_src[end - 1:]

    after = _arms_from_module(new_module_src)
    if [a["name"] for a in after] != [a["name"] for a in arms] + [new_name]:
        raise SystemExit("ABORT add-reverse-arm: table reads %s"
                         % [a["name"] for a in after])
    twin = after[-1]
    if not twin.get("reverse") or twin["k_eval"] != base["k_eval"]:
        raise SystemExit("ABORT add-reverse-arm: twin did not copy the base recipe: %r" % twin)
    print("  added reverse twin %r of %r (w=%s, shares its volume recipe)"
          % (new_name, base_name, weight))
    return replace_module(nb, new_module_src)


def special_arm_weights_alt(nb: dict) -> dict:
    """Rewrite the Raptor arm-weight vector inside the embedded RAPTOR module.

    The arm table is not a top-level literal: it lives in `_KE_SRC`, a single-quoted
    Python string holding the whole RAPTOR module with escaped newlines, inside cell 39.
    Measured layout (2026-09-11):

        ARMS = [
            {"name": "maxspan-v5",         ..., "reverse": False, "w": 0.55},
            {"name": "native384dense-v10", ..., "reverse": False, "w": 0.10},
            {"name": "maxspan-v5-reverse", ..., "reverse": True,  "w": 0.15},
            {"name": "native384-v8",       ..., "reverse": False, "w": 0.20},
        ]

    We match on the arm NAME and rewrite the `"w": <float>` that follows in the same
    brace group, so a reordering of the arms cannot silently retarget the experiment.
    """
    targets = {"maxspan-v5": "0.60", "native384dense-v10": "0.10",
               "maxspan-v5-reverse": "0.10", "native384-v8": "0.20"}
    pattern = re.compile(
        r'("name":\s*"(?P<name>[a-z0-9\-]+)".*?"reverse":\s*(?:True|False),\s*"w":\s*)'
        r'(?P<w>[0-9.]+)', re.S)

    edited = 0
    for _ in range(12):                     # rewrite one value per pass; offsets shift
        progressed = False
        for cell in nb["cells"]:
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell.get("source", []))
            if "ARMS = [" not in src:
                continue
            for m in pattern.finditer(src):
                name = m.group("name")
                if name in targets and m.group("w") != targets[name]:
                    print("  arm %-20s w: %s -> %s"
                          % (name, m.group("w"), targets[name]))
                    src = src[:m.start("w")] + targets[name] + src[m.end("w"):]
                    cell["source"] = src.splitlines(keepends=True)
                    edited += 1
                    progressed = True
                    break
            if progressed:
                break
        if not progressed:
            break

    # Assert on the FINAL state, not on the number of edits: two of the four target
    # weights already equal the desired value, so an edit count is the wrong invariant.
    # Read the four names and weights back out of the rewritten cells.
    final = {}
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "ARMS = [" not in src:
            continue
        for m in pattern.finditer(src):
            final[m.group("name")] = m.group("w")
    if final != targets:
        raise SystemExit("ABORT arm-weights-alt: after %d edit(s) the arm table reads %r, "
                         "expected %r. The embedded ARMS layout has changed."
                         % (edited, final, targets))
    print("  arm table verified: %r" % final)
    return nb


def verify_notebook(nb: dict, path: Path) -> None:
    """Prove the result still parses, cell by cell, without touching the filesystem."""
    code_cells = 0
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        try:
            compile(src, "<cell%d>" % i, "exec")
        except SyntaxError as exc:
            raise SystemExit("ABORT: cell %d no longer compiles: %s" % (i, exc))
        code_cells += 1
    if code_cells < 30:
        raise SystemExit("ABORT: only %d code cells - the notebook looks truncated"
                         % code_cells)
    print("  verified: %d code cells compile" % code_cells)


def resolve_plan(name: str, seen=None):
    """Flatten an experiment into an ORDERED list of operations, plus its dataset adds.

    Each operation is ("edits", [(old, new), ...]) or ("special", "<name>"). Steps and
    specials were originally two parallel lists, which silently misaligned the moment an
    experiment had edits but no special: zip() then handed the edits to the wrong handler
    and the edit was never applied - the exact "looks applied, is not" failure this file
    exists to prevent. One ordered list removes the possibility.
    """
    if seen is None:
        seen = []
    if name in seen:
        raise SystemExit("ABORT: %s composes itself (cycle: %s)" % (name, seen + [name]))
    if name not in EXPERIMENTS:
        raise SystemExit("ABORT: unknown experiment %r" % name)
    exp = EXPERIMENTS[name]
    components = exp.get("compose")
    if not components:
        ops = []
        if exp.get("edits"):
            ops.append(("edits", exp["edits"]))
        if exp.get("special"):
            ops.append(("special", exp["special"]))
        return ops, list(exp.get("dataset_sources_add") or [])

    ops, datasets, specials = [], [], []
    for comp in components:
        sub_ops, sub_data = resolve_plan(comp, seen + [name])
        for kind, payload in sub_ops:
            if kind == "special":
                specials.append(payload)
        ops.extend(sub_ops)
        datasets.extend(sub_data)
    dupes = {s for s in specials if specials.count(s) > 1}
    if dupes:
        raise SystemExit("ABORT %s: two components both use the %s rewrite, which would "
                         "edit the same lines twice" % (name, ", ".join(sorted(dupes))))
    return ops, datasets


def build(name: str, dry: bool):
    exp = EXPERIMENTS[name]
    out_dir = EXP_ROOT / name
    print("experiment: %s" % name)
    print("  %s" % exp["desc"])

    ops, datasets = resolve_plan(name)
    if exp.get("compose"):
        print("  composes: %s" % ", ".join(exp["compose"]))

    src_nb, src_meta = source_paths(exp)
    print("  source: %s" % src_nb.relative_to(ROOT))
    raw = src_nb.read_text(encoding="utf-8")
    print("  source notebook sha256: %s" % sha256_text(raw)[:16])

    nb = json.loads(raw)
    applied_edits = []
    for kind, payload in ops:
        if kind == "special":
            if payload == "arm_weights_alt":
                nb = special_arm_weights_alt(nb)
            elif payload == "add_widedense_arm":
                nb = special_add_widedense_arm(nb)
            elif payload == "add_finespacing_arm":
                nb = special_add_finespacing_arm(nb)
            elif payload == "add_finespacing_heavy":
                nb = special_add_finespacing_arm(nb, w=0.50, scale=0.50)
            elif payload == "drop_arm":
                nb = special_drop_arm(nb)
            elif payload == "add_reverse_arm":
                nb = special_add_reverse_arm(nb)
            else:
                raise SystemExit("ABORT: unknown special %r" % payload)
        else:
            nb = apply_edits(nb, payload, name)
            applied_edits.extend(payload)

    new_raw = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
    parsed = json.loads(new_raw)
    written = "".join(c for cl in parsed["cells"] for c in cl.get("source", []))
    for old, new in applied_edits:
        if new not in written:
            raise SystemExit("ABORT: an edit reported as applied is not in the written "
                             "cells: %r" % (new[:120]))

    verify_notebook(parsed, out_dir)
    print("  result notebook sha256: %s" % sha256_text(new_raw)[:16])

    if dry:
        print("  dry-run: nothing written")
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "experiment.json").write_text(
        json.dumps({"name": name, "desc": exp["desc"],
                    "source": str(src_nb.relative_to(ROOT)),
                    "source_sha256": sha256_text(raw),
                    "result_sha256": sha256_text(new_raw)},
                   indent=2), encoding="utf-8")
    (out_dir / src_nb.name).write_text(new_raw, encoding="utf-8")

    meta = json.loads(src_meta.read_text(encoding="utf-8"))
    slug = "rsna-knee-ens-" + name
    meta["id"] = "ppxl16/" + slug
    meta["title"] = "RSNA Knee ens " + name
    meta["code_file"] = src_nb.name
    meta["is_private"] = True
    meta.pop("id_no", None)
    meta.pop("docker_image", None)      # the author's private BYOD image is not ours to use
    added = datasets
    for ref in added:
        if ref not in meta["dataset_sources"]:
            meta["dataset_sources"].append(ref)
    if added:
        print("  attached dataset(s): %s" % ", ".join(added))
    (out_dir / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    readback = json.loads((out_dir / src_nb.name).read_text(encoding="utf-8"))
    if sha256_text((out_dir / src_nb.name).read_text(encoding="utf-8")) != \
            sha256_text(new_raw):
        raise SystemExit("ABORT: read-back of the written notebook differs")
    verify_notebook(readback, out_dir)
    print("  wrote %s" % out_dir)
    print("  kernel id: %s" % meta["id"])
    return out_dir


def run_cli(args, timeout=1800):
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run(["kaggle"] + list(args), capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=env, timeout=timeout)
    lines = [l.rstrip() for l in ((proc.stdout or "") + "\n" + (proc.stderr or "")).splitlines()
             if l.strip() and not l.startswith("Warning: Looks like you're using an outdated")]
    return proc.returncode, lines


def push(name: str) -> int:
    out_dir = EXP_ROOT / name
    if not (out_dir / "kernel-metadata.json").is_file():
        print("build it first: --experiment %s (without --dry-run)" % name)
        return 2
    rc, lines = run_cli(["kernels", "push", "-p", str(out_dir)])
    for ln in lines:
        print("  " + ln)
    if rc != 0:
        return rc
    meta = json.loads((out_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    m = re.search(r"Kernel version (\d+) successfully pushed", "\n".join(lines))
    # ⚠️ `kaggle kernels push` EXITS 0 even when it prints "Kernel push error" -- observed
    # 2026-09-21 with "Maximum weekly GPU quota of 30.00 hours reached.". So `rc` cannot be
    # the check; the success SENTENCE has to be. Without this branch the command returned 0
    # having pushed NOTHING, which reads exactly like success (same silent-success class as
    # `kernels output` truncating on an SSL error, HANDOFF 15.5).
    if not m:
        print("\n  PUSH FAILED (rc=%d) -- the message above is the reason" % rc)
        return 1
    if m:
        ver = m.group(1)
        print("\n  -> register with the director:")
        print("     python scripts/director.py --record-push %s:%s" % (meta["id"], ver))
        # Make the director aware immediately; it is additive, never destructive.
        try:
            subprocess.run([sys.executable, str(ROOT / "scripts" / "director.py"),
                            "--record-push", "%s:%s" % (meta["id"], ver)],
                           cwd=str(ROOT), timeout=120)
        except Exception as exc:
            print("  (could not auto-register with the director: %s)" % exc)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    if args.list or not args.experiment:
        print("experiments:")
        for name, exp in EXPERIMENTS.items():
            print("  %-18s %s" % (name, exp["desc"][:110] + "..."))
        print("\nNOTE: each experiment pushes its OWN kernel id, because pushing a new "
              "version of a kernel with a PENDING submission can destroy it.")
        return 0

    if args.experiment not in EXPERIMENTS:
        print("unknown experiment %r; --list to see them" % args.experiment)
        return 2

    build(args.experiment, dry=args.dry_run or not args.push)
    if not args.push:
        return 0
    return push(args.experiment)


if __name__ == "__main__":
    sys.exit(main())
