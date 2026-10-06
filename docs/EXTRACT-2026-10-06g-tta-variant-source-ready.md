# EXTRACT 2026-10-06 (g) — the TTA variant's source is now in hand and verified

Sixth round of the autonomous goal, with both live measurements still pending (`56872230` the reader leg;
the distilled gate QUEUED). Rather than re-scan a public surface that three sweeps have already shown is
converged, this round **prepared the next experiment** — the genuine "prepare, don't fire" work.

---

## 1. What was extracted

`work/stackpp_src/{preprocess,knee,infer}.py`, taken verbatim from the `%%writefile` cells of
`romantamrazov/rsna-knee-dinosaur-v5`. Our current reader arm runs `goodpjw2008`'s versions of the same
three files; the variant needs these.

**Verified rather than assumed:** all three parse under `ast`; their definitions are complete and coherent —

| file | defines |
|---|---|
| `preprocess.py` | `MM_PER_PX SIZE MAX_SLICES PLANES CANON _HDR_TAGS` + `_natural_key _floats _header _decode _canon_slice _resample _select_headers load_series` |
| `knee.py` | `LABELS PLANES N_SLOTS MEAN STD` + `slot_of build_study_table window_centres make_views KneeNet` |
| `infer.py` | `TTA_MODES` + `DicomStudyDataset load_models _rank_members run` + the `__main__` entry |

`preprocess.py` carrying `_header` / `_HDR_TAGS` / `_select_headers` is exactly the *"decode only slices that
can actually be used"* change; `knee.py` being smaller than the base is correct, because it is the
inference-only cut. `infer.py` requires **≥ 3 checkpoints** and we have exactly three.

## 2. ⚠️ Two of my own checks produced false alarms, and both were caught by looking further

* **A symbol check flagged `_read` as missing from `preprocess.py`.** `_read` is the *base* file's helper
  name; the variant splits it into `_header` + `_decode`. **Asserting that a variant must contain the
  baseline's private helper names is not a valid check** — it tests the wrong thing.
* **An AST probe reported `infer.py` had no `__main__` block.** It does. Finding `if __name__ == "__main__":`
  requires walking `ast.If`, not `FunctionDef`/`Assign`. **The probe's negative was wrong, and a wrong
  negative would have caused the file to be rebuilt needlessly.**

⇒ Both are the same shape as this repository's recurring error: **a check that cannot distinguish "absent"
from "I looked in the wrong place" will eventually report absence.** The mitigation is the same as always —
when a structural check fails, open the artefact before believing it.

## 3. The guard that matters passed

```
leaderboard / public_lb references in the variant source: 0
TTA modes: 6 references      rank ensemble: 2      Synovitis handling: 1
```

⇒ The admissibility claim from the prereg (`READ-2026-10-06e`) **holds on the extracted code**: the change
set transforms inputs and aggregates outputs, and consults nothing fitted on the public split.

## 4. State: ready, not fired

Building the arm is now a small step — package these three files instead of the base three, and set
`RSNA_READER_TTA="mid,edge"` for the 3-checkpoint case. **Nothing is built and nothing is pushed**, because
the entry gate is unchanged: the variant only runs if the **base** reader leg moves the in-block percentile
by ≥ 3 points against **16.03 %**, with the margin reported against **−118**.

⚠️ And when it is built, `Synovitis_w = 0.08` (vs `other_w = 0.30`) should be the **first** thing varied —
it is the only item in the change set that alters a fitted quantity, and its justification is 58 gold
studies whose CI is ±0.021.
