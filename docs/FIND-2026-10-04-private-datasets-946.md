# FIND 2026-10-04 — why the public "0.946" d4-blend notebook cannot be reproduced: PROVEN, not guessed

Question asked: *"why can't we replicate this kind of 0.946 code-area notebook — is it because its
private datasets are hard to reproduce?"* **Answer: yes, and it is not a matter of difficulty — the
delta is exactly three private datasets, and everything else is already mounted in our own kernel.**

## The proof

Pulled `pjmathematician/rsna-knee-d4-blend` (`work/d4blend/`) and diffed its sources against our
0.943 stack (`kernels/kernel-0943s-2coat/`):

```
THEIRS (d4-blend, screen shows Public Score 0.946):  17 dataset_sources, 3 of them EMPTY placeholders
OURS   (0943s-2coat, 0.943):                         14 dataset_sources, 0 placeholders

public datasets they mount that we also mount ........ 14 / 14
public datasets only in theirs ........................ (none)
only in ours .......................................... (none)
kernel_sources ........................................ IDENTICAL  ['sofiaanjenje/rsna-knee-e11-train',
                                                                     'sofiaanjenje/rsna-knee-e13-train']
model_sources ......................................... IDENTICAL  ['metaresearch/dinov2/PyTorch/small/1']
competition_sources ................................... IDENTICAL
machine_shape ......................................... IDENTICAL  NvidiaTeslaT4
enable_gpu / enable_internet .......................... IDENTICAL  True / False
```

**⇒ The only difference is three private datasets.** Everything public about that notebook is
already in our kernel — in fact our `kernel-0943` is a **byte-identical fork of the frontier source**
(`docs/AUDIT-2026-10-04-provenance.md`: 0 of 51 cells differ), and this notebook is that same public
stack with private extras bolted on.

## How the private refs are visible at all
Kaggle **strips private dataset refs from public metadata**, leaving `''`. The pulled
`kernel-metadata.json` literally reads
`dataset_sources: ['', '', '', 'dreaddevelopment/raptor-knee-maxspan', ...]` — three empty strings,
matching the three greyed-out **"(Private Dataset)"** rows with lock icons in the UI screenshot.

## ⚠️ The notebook's own banner disagrees with its header — and the banner is the honest one
The Kaggle header reports **Public Score 0.946**, but the notebook's body banner says
**"Public LB: 0.943"**, and describes itself as combining "the verified **0.943** SOTA recipe
(Mattia Angeli v34 / Jiwei Liu v9)".

**0.943 is exactly what we score**, and it is exactly the recipe. So the body banner is the *public
recipe's* number and the 0.946 header is the number **with the private data**. This is the same
family-level pattern the artifact scan already recorded (`ryokucha/rsna-knee-d4-blend-0946-ours10`,
`yamadan96/rsna-knee-d4-public0946` — both carry the same "this scores 0.943" correction).

## It is NOT a hardware problem, and it is not a skills problem
The notebook runs on **`NvidiaTeslaT4`** — the same hardware we are entitled to (F43/F44 measured
that we get 2× T4). So unlike the ARC case, this one is not gated on hardware. **It is gated on
private data we cannot obtain.**

## And the size of the prize is below the instrument anyway
0.946 − 0.943 = **0.003**, against a public-LB **SE ≈ 0.008**. So even with the private datasets we
could not *confirm* the gain on the public board — which is the same conclusion
`docs/FIND-2026-10-04-medal-band-density.md` reached about small deltas: they may matter for the
private ranking inside the 794-team band, but they cannot be verified here.

## The generalisable rule
**"A public notebook scored X" is not a replicateable claim when its `dataset_sources` contains empty
strings.** The check costs one `kaggle kernels pull` and takes seconds:

```
kaggle kernels pull <owner>/<slug> -p <dir> -m
# then count the '' entries in dataset_sources
```

Two of the three `>0.943` artifacts found in this competition were inflated this way. Apply it before
spending any time on a public notebook.
