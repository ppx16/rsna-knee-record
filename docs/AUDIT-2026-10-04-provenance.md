# AUDIT 2026-10-04 — provenance chain of the shipped arms, cell by cell

## Why
Two earlier checks of this kind each caught something real:
* `AMP_PREF='bf16'` looked like a possible deviation and turned out to be **byte-identical in the
  frontier source** → line closed on provenance, not on a rule.
* the reference is said to feed Raptor **already rank-normalised to [0,1]** into the outer blend;
  a rank-affine renormalisation inside a rank-average is **not** blend-invariant, so a mismatch would
  mean running a different reduction. Ours matches the [0,1] convention.

So the same question was asked of the **whole notebook**: is what we submit the frontier's artifact
plus only the changes we deliberately made?

## Result — the chain is exactly what we intend, and nothing else

```
frontier -> kernel-0943       : 0 differing cells of 51      <-- byte-identical fork
kernel-0943 -> kernel-0943cf  : 4 differing cells of 51
    cell  2  len 4146 -> 4186   insert: "cn0-ours": 0.20
    cell 12  len 23652 -> 23746 insert: the ppxl16/rsna-knee-cn0-weights dataset path
    cell 43  len 29296 -> 29285 delete: 'Fracture'          <-- the C81 fracture fix
    cell 45  len 44987 -> 46062 insert: the cn0 member wiring
kernel-0943cf -> kernel-0943cfm: 1 differing cell of 51
    cell 43  len 29285 -> 29291 insert: "MCL"
```

Reference notebook: `downloads/research/fresh0924/maverickss26_rsna_knee_restructured_version_3/`.

## What this establishes
1. **`kernel-0943` is a byte-identical copy of the frontier source** — so when we say "we are on the
   public recipe, not below it", that is literal, not a claim about scores.
2. **Every downstream change is deliberate and attributable**, one axis at a time:
   `0943cf` = 0943 + {`cn0-ours` view: config weight, dataset mount, and the member wiring} + {the
   fracture fix}; `0943cfm` = `0943cf` + {`MCL`}.
3. **No accidental deviations anywhere** — including nothing that would silently alter the reduction,
   the precision, or the routing. That was the specific risk being tested.
4. It independently confirms **`MCL` is present in cell 43** of `0943cfm`, corroborating that the
   submission `56791368` is a genuinely distinct arm rather than a re-run. (The receipts said the same;
   this is a second, independent line of evidence.)

## ⚠️ One over-claim found and recorded, not fixed by re-running
`state/EXPERIMENTS.md` **D1** closes "fusion-weight tuning (**all pre-registered axes**)" with the
evidence "all ≤ 0.941; axis size 0.0046, artefact 0.0012". But **`rad_alpha` has never been submitted** —
there is no `radalpha` row in `state/lb_ground_truth.csv` — and `v2src-radalpha35` / `v2src-radalpha65`
are marked **"Built but not pushed"** in `scripts/ens_experiment.py --list`. D1 predates that axis, so
"all" was written before there was anything to be exhaustive about.

**It is still closed by category** — `radalpha` self-describes as a pre-registered fusion-weight axis,
and D1 closes that category on evidence (five readings at 0.941, axis size 0.0046 against an artefact
of 0.0012). Also, `rad_alpha`'s current 0.55 is **not untested**: the config comment records
"0.941 lineage used 0.50", i.e. the 0.50 → 0.55 step had LB backing. What is un-run is the *bracket*
(0.35 / 0.65) around it.

⇒ **Do not re-open it.** Recorded so that the ledger's "all" is not mistaken for a completeness proof,
and so nobody later "discovers" `rad_alpha` and believes it is a fresh lever.
