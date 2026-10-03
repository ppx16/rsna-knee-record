# MCL arm — the one remaining label-level lever (2026-10-03)

## What was found

`_RAD_EXCLUDE` decides which findings the RadImageNet head is blended into (alpha 0.55). It shipped
as `("Baker's", 'Fracture')` — **hand-picked, with a 58-study audit as post-hoc justification**.

C81 falsified the `'Fracture'` half on a 4185-study instrument (`Fracture AUC 0.9187 → 0.9305`,
+0.0118 on that column, +0.00099 macro, reproduced by five label sources, **with the 58-study set the
only dissenter**). We adopted that fix; it is in `0943cf`.

**The obvious next question — "then is `"Baker's"` also wrong?" — has a clean NEGATIVE answer.**
Re-measured independently on the same instrument:

| | value |
|---|---|
| instrument reproduction | **8/8 anchors exact** (incl. both alpha=0 anchors) |
| Baker's, n=3776, 528 positives | 0.9635 (excluded) → 0.9586 (blended) = **−0.0049**, CI [−0.0096, +0.0003] |
| sign stability | **negative in 35/35** {6 label sources} × {5 Rad assemblies} |
| alpha curve | peaks at 0.10–0.20; **alpha=0 already beats the shipped 0.55** — the shipped dose is past the peak |
| folds | 4/5 negative |
| macro cost of un-excluding | **−0.000403** |
| ⭐ the discriminator | the 58-study audit gave Baker's **−0.0036 / −0.0018 / −0.0045** — negative at every alpha. **Unlike Fracture, small-sample and large-n AGREE here.** |

⇒ **Keep `"Baker's"` excluded. Do nothing.** Recorded as a negative result.

## ⭐ But the audit's OWN worst label was never excluded: MCL

The exclusion set was never derived from the audit it cites. Proof:

| label | 58-audit Δ (α=.20/.35/.50) | excluded? |
|---|---|---|
| **MCL** | **−0.0159 / −0.0408 / −0.0828** (worst of 12, by 2×) | ❌ blended at 0.55 |
| Effusion | −0.0075 / −0.0211 / −0.0404 | ❌ blended |
| Baker's | −0.0036 / −0.0018 / −0.0045 | ✅ |
| Fracture | −0.0014 / −0.0097 / −0.0181 | ✅ |

Large-n re-measurement of MCL, same instrument that validated the Fracture fix:

```
n=3802, 210 positives:  0.8930 -> 0.8821 = -0.0109
paired-bootstrap CI [-0.0215, -0.0006],  P(>0) = 0.022
negative in 35/35 {6 label sources} x {5 Rad assemblies}
per-fold 3/5 negative (-0.0199 / +0.0032 / -0.0028 / -0.0100 / +0.0022)
expected macro gain if excluded: +0.000904   (the Fracture fix was +0.000824)
```

## Why this is NOT one of the closed levers (the distinction matters)

The closed list forbids **tuning** fusion weights / per-label routing / calibration. This is a
different shape, and the five tests that separate them:

1. **A correction, not a fit** — `_RAD_EXCLUDE` is a binary set-membership; one member flips because
   its **sign reversed**. No grid, no optimum, no value chosen to maximise a score.
   (`coatnet_w` and per-label alpha ARE continuous values fitted against a score.)
2. **Direction pre-registered** by an earlier independent measurement — the 58-study audit named MCL
   worst of twelve, independently of the large-n run. Confirming a pre-existing sign ≠ searching.
3. **The instrument is independent of the decision's provenance** — OOF from three different authors,
   labels from four to six more, none of whom made the original choice.
4. **Invariance** — the sign survives 35 perturbations. A fitted value has no invariance test.
5. **Honest-fold discipline** — where the same *operation* was done as a fit, the gain vanished:
   per-fold per-label weights gave **+0.0005 fitted → −0.0003 honest** (n=4407).

## What was built and is running

`scripts/build_mcl_arm.py` → **`kernels/kernel-0943cfm`** (`ppxl16/rsna-knee-0943cfm`).

Source is **`kernel-0943cf`** (our best arm: `cn0` 5th Raptor view + the Fracture fix, public 0.943),
one token changed:

```
- _RAD_EXCLUDE = ("Baker's",)
+ _RAD_EXCLUDE = ("Baker's", "MCL")
```

That stacks the three changes this repo actually has large-n measurements for. Assertions in the
builder: exactly one `_RAD_EXCLUDE` match, the new line landed, the old line is gone, `Fracture` did
not reappear, cell count unchanged (51). GPU=T4, internet=False, kernel_type=notebook, `-t 34200`.

```
Kernel version 1 successfully pushed.
ppxl16/rsna-knee-0943cfm has status "KernelWorkerStatus.RUNNING"
```

## ⚠️ 2026-10-04 CORRECTION — the "identical stage file" test is NOT a duplicate test

While automating the EffB0 acceptance check I ran that same comparison on a **known-applied** change and
it came back "identical":

| file | `0943cf` (no MCL) | `0943cfm` (+MCL) | |
|---|---|---|---|
| `submission.csv` | `a3a6bc5832b1` | `a3a6bc5832b1` | **same** |
| `_pipeline_stage.csv` | `a3a6bc5832b1` | `a3a6bc5832b1` | **same** |
| `v558_rad_receipt.json` | `70a38f2a6cd3` | `69f920bd10b8` | **DIFFERS** |
| `_coat_arm_receipt.json` | `5b175a8d8689` | `ec4675bc5ce9` | **DIFFERS** |

**The MCL change IS applied** — the Rad receipt and the CoAt-arm receipt differ, and those are exactly
where `_RAD_EXCLUDE` acts. The final ranks coincide only because the public run covers **3 visible
studies** and `_pipeline_stage.csv` holds **ranks**: with 3 studies there are 6 possible orderings per
label, so a genuinely applied change can leave it byte-identical by chance.

⇒ **`0943cfm` / `56791368` is NOT a duplicate of `0943cf`.** It is a legitimate, distinct submission.

⇒ **Methodology correction, now encoded in `scripts/verify_effb0_arm.py`:** a byte-identical
`_pipeline_stage.csv` is **corroborating evidence only, never a verdict**. The primary signal that a
change reached the graph is the **per-stage receipts** (`v558_rad_receipt.json`,
`_coat_arm_receipt.json`), which carry per-stage hashes and counts; the **log** arbitrates whether the
leg ran at all. The first version of that script would have **falsely rejected a working leg**.

⚠️ **The EffB0 v1 conclusion still stands** — but note *why*: it rested on the **explicit log line**
`[effb0] LEG FAILED … ValueError: 'SAG_FS' is not in list`, with the matching stage file as
corroboration. The log was the evidence; the hash was the confirmation. Relying on the hash alone would
have meant relying on an unsound test.

## ⚠️ Honest weakness — do not bury this

MCL is **weaker than the Fracture case**: the CI only just excludes zero, only 3/5 folds agree, and it
emerged from a 12-label scan (its sign was pre-registered, which is what makes it usable, but the
replication is not overwhelming). The instrument's parent is **stage-1 (public20)**, not the deployed
stage-2 (stage1 + A5 at 0.52); a stronger parent could move a marginal sign, and MCL is exactly the
marginal case.

And the decisive caveat: **the public board cannot measure this.** SE ≈ 0.008, the expected effect is
+0.0009. Nine arms spanning four axes came back across a total spread of 0.002. So the submission is
**a guard against a crash, not a measurement** — the same framing `0943f` was submitted under.

## Next step (when the run reaches COMPLETE)

1. **Acceptance test before submitting** (the arms are fail-CLOSED, not fail-soft — read the log, per
   `docs/RUNBOOK-0943.md` §3): confirm `DINOv2 fail-closed gate PASS: 20/20 fingerprints`,
   `[coat-arm] CoAt family members: ...` includes our view, and
   `COMPLETE: /kaggle/working/submission.csv`.
2. Then:

```
kaggle competitions submit -c rsna-knee-abnormality-detection \
    -k ppxl16/rsna-knee-0943cfm -v 1 -m "0943cfm - 0943cf + MCL added to _RAD_EXCLUDE; +0.000904 macro on n=3802/210pos, 35/35 sign-stable"
```

3. If it reads **≤ 0.940** that is a real regression signal (the board can catch that); anything
   0.941–0.945 is **no information** and the arm stays a candidate for the final pair.

## Untestable, left alone (recorded so nobody re-opens it)

`_RAD_CAL['gate'] = [ACL, Medial OA, Lateral OA, PF OA, Effusion, Baker's, Contusion]` is a genuine
per-label routing map with **unknown provenance** (an embedded base64/zlib artifact carrying two of
the four audit-negatives). It is **completely unmeasurable** — no calibrator OOF exists, because it
would need the twelve metadata protocol columns. Do not touch it.
