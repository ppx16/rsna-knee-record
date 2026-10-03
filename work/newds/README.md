# Kaggle Competition Leaderboard Intelligence

An original, timestamped panel of Kaggle competition board shape: **240 competition-scrape rows**, **57 fields**, rank-derived medal lines, exact displayed-score tie blocks, movement fields, and submission mechanics.

The useful unit is not “a leaderboard screenshot.” It is one competition at one measured UTC scrape, with the bronze line and the shape around it preserved so later releases can show movement.

## Files

- `competition_intelligence_panel.csv` — analysis-ready panel.
- `data_dictionary.csv` — every panel column defined, one row per column.
- `provenance.csv` — source URLs, timestamps, roles, and licensing/terms boundary.
- `source_competitions.json` — exact nested source snapshot for audit.
- `build_dataset.py` — deterministic standard-library rebuild.
- `build_summary.json` — validation results and source hash.
- `UI-PASTE.md` — exact file/column descriptions queued for Kaggle's browser-only fields.

## What is measured, derived, and unknown

Source values are copied without filling gaps. Empty cells mean the queried source did not expose a measurement. JSON list fields retain explicit “not measured” records where the collector supplied them.

Derived fields are narrow arithmetic over measured values:

- `bronze_score_gap = bronze_score - our_score` for maximize metrics.
- `bronze_rank_gap = our_rank - bronze_rank`.
- `bronze_tie_fraction = bronze_tie_count / public_leaderboard_rows`.
- movement deltas compare only to the prior measured scrape for the same competition.
- `days_to_deadline` is UTC deadline minus UTC scrape time.

Medal cutoffs are rank-derived using the collection convention recorded verbatim in `medal_line_basis`; they are not claimed to be direct medal-table fields. Competitions with `medals_allowed=false` do not receive synthetic medal lines.

## Quick start

```python
import pandas as pd

panel = pd.read_csv("competition_intelligence_panel.csv", parse_dates=["scraped_utc", "deadline_utc"])
eligible = panel[panel.medals_allowed.eq(True)]
dense_bronze = eligible.sort_values("bronze_tie_fraction", ascending=False)
print(dense_bronze[["competition_slug", "bronze_score", "bronze_tie_count"]])
```

## Coverage and refresh cadence

Current competitions: `biohub-cell-tracking-during-development`, `kaggriculture`, `playground-series-s6e8`, `pokemon-tcg-ai-battle-challenge-strategy`, `rsna-knee-abnormality-detection`.

Planned cadence: **weekly**, plus an out-of-cycle refresh after a material mechanics or eligibility change. Each release appends observations; it does not rewrite old scrapes.

## Rebuild and validation

From this directory in the source repository:

```bash
python build_dataset.py --source ../intel/competitions.json --output . --expect-rows 240
```

The release source hash is `1182457f40aa0ef8b4f36571098077c1edc3fb279461e899e44c8661a863dac3`. The builder asserts unique `(competition_slug, scraped_utc)` keys, parseable UTC timestamps, exact dictionary/header agreement, equality between every non-null flattened bronze score and its nested source movement record, and same-timestamp alignment for each raw observation's board, mechanics, and public-knowledge fields.

## Provenance and licence

Meta Kaggle is Kaggle's official daily public-activity dataset and is cited in `provenance.csv`. Live leaderboard and mechanics measurements link to each public competition page. The collection includes no Kaggle competition data files and no inferred KYC/runtime values.

The original compilation, schema, documentation, and build code are released under **Apache 2.0**. Upstream Kaggle pages, APIs, and competition materials remain governed by their own terms; attribution here does not relicense upstream content.

## Limitations

- Observation count per competition grows with each refresh (see `build_summary.json` for the count as of this release). Movement deltas are populated from the second scrape onward for a competition, and only when both adjacent scrapes expose the required measurement.
- Five earliest movement observations survive only inside a later nested history and have no same-timestamp raw payload. Their movement fields are retained, while supplemental mechanics, notebook, discussion, and resource-policy fields remain empty rather than being backfilled from a later scrape.
- Public leaderboard score can be a noisy or unrepresentative proxy for final performance.
- Team counts can disagree across live and Meta Kaggle endpoints; both are retained rather than reconciled by guesswork.
- Notebook votes are measured by the kernels endpoint, but linked competition scores are not exposed and remain empty.
- `medals_allowed` is the eligibility gate; Playground and judged Hackathon rows remain useful intelligence but cannot contribute competition-track medals.
