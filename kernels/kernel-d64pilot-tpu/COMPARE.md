# The paired comparison this run completes

| | 44-slot arm | 64-slot arm |
|---|---|---|
| tag | `nx3360` (EXISTS) | `d64nx3360` (this kernel) |
| corpus | published 44 slots, span 0.152-0.847 | built 64 slots, span 0.02-0.98 |
| arch / res / bs / k / k_eval / seed / epochs | convnext_base.fb_in22k_ft_in1k / 336 / 8 / 12 / 24 / 42 / 12 | IDENTICAL |
| backend | TPU | TPU |
| fold | 0 | 0 |

Everything is fixed except the corpus, and the backend matches, so the delta-OOF is
attributable to the corpus. Compare with:

    python scripts/cv_score.py --dir downloads/tpu --pair nx3360 d64nx3360

Pre-registered reading (HANDOFF 150.2): delta > 0.004 soft = real; <= 0.004 = abandon the
dense-corpus line.
