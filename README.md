# RSNA Knee Abnormality Detection — working record

Private record for a Kaggle **code competition**: 12-label macro ROC-AUC, deadline **2026-10-22**.
The hidden test is 1322 studies; the visible test is 3, which is why almost nothing here can be
settled on the public board.

**Start with [`STATUS.md`](STATUS.md)** — one screen: what is live, what is closed, what is next.
This file is the map.

---

## The rule this repository is built on

> **A conclusion is only as good as the instrument that produced it, and an instrument is only
> trusted after it has been shown to pass on a known-good input and fail on a known-bad one.**

Everything below follows from that. Three separate times in this project a check reported a
confident verdict it could not actually see: a scale-invariant "faithful" test, a log decoder that
turned two successful runs into FAILs, and a submission pipeline that passed every gate while
producing three all-zero rows. All three are recorded, with the controls that now catch them.

---

## Where things live

| Path | What it is |
|---|---|
| `STATUS.md` | **Live state.** Rewritten every session. One screen. |
| `state/FACTS.md` | **Atomic facts**, append-only, numbered `A1..An`. Each entry is one thing that is true and how it was measured. |
| `HANDOFF.md` | **Narrative**, numbered `§N` sections. What was attempted, what happened, what it cost. |
| `RUNBOOK-0943.md` | The copy-pasteable sequence for pushing and submitting the 0.943-line arms. Read it before touching the submission line. |
| `PLAN-arm-image-metadata.md` | **Pre-registered plan** (not started) for the one live research direction: an arm that consumes images *and* acquisition metadata. Read §3's kill conditions before spending anything on it. |
| `state/EXPERIMENTS.md` | Registered experiments and their verdicts, including the ones closed as negative. |
| `scripts/` | Every tool worth keeping. Reusable analysis lives here, not in ad-hoc commands. |
| `kernel-*/` | Kaggle kernels we own: the five 0.943-line arms, the A5-OOF kernel, the CPU site probe. |
| `downloads/` | Pulled artifacts. **`*.npz` and `*.json` are tracked** (raw OOF predictions and audits are the evidence); weights, data blobs, logs and third-party notebooks are not. |

## The recording protocol

Four files, four different jobs. Putting the wrong kind of thing in the wrong file is what made the
record hard to use:

1. **A new measured fact** → one row appended to `state/FACTS.md`, with the number, the measurement,
   and the section it came from. Never rewrite an old row; if it turns out wrong, add a new row that
   says so and cross-reference it. (A77 was corrected by A95 this way.)
2. **A block of work** → a new `§N` section in `HANDOFF.md`: what was tried, what was observed, what
   it cost, what changed as a result.
3. **Live state** → `STATUS.md`, rewritten in place. If it grows past one screen, it is a HANDOFF
   section, not a status.
4. **A tool** → `scripts/`, with a docstring that says what problem it solves and what it would get
   wrong. Scripts that exist to answer one question still belong here if the question will recur.

Write facts **at the moment they are measured**. An unrecorded measurement is one we will pay for
twice — the repository is the only memory that survives a closed session.

## Standing policies

**Resources.** TPU has a slow queue and takes non-urgent work. Kaggle GPU is scarce. AutoDL carries
urgent, long-running experiments. The deciding question is not which machine is cheaper but
**whether the experiment needs the images**:

- needs the competition DICOM → **Kaggle only** (the data is mounted there; moving it costs hundreds
  of gigabytes)
- does not need images (fusion, calibration, routing) → **local CPU, seconds, free** — never an
  accelerator
- training or long runs that avoid the DICOM → AutoDL when urgent, TPU when not

**Submissions.** Scoring reruns do not consume our quota, but a submission is **only for confirming
an improvement that local evidence already supports** — never for discovering whether one exists.
Public leaderboard noise is ~0.008 while the effects we chase are 0.001–0.003, so "submit and see" is
multiple comparisons on noise. Pre-register the threshold before looking.

**Process.** Diagnose one cause, fix that cause, and be able to say which cause an edit addressed. A
diagnosis that lists three causes is not resolved by fixing one of them (A89).

## Reproducing the two live tools

```powershell
# measure accelerator quota (CLI 2.1.2 has no `quota`; 2.2.4 in an isolated venv does)
powershell -File scripts\quota.ps1

# re-verify the five arms before pushing anything
python scripts\preflight_0943.py          # must print PREFLIGHT OK
```

`scripts/build_0943_arms.py` rebuilds the five arms deterministically from the frontier notebook,
with exact-match assertions on every edit. The four originals come back byte-identical; that property
is what makes a rebuild safe to trust.
