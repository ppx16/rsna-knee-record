# RSNA Knee Abnormality Detection — Competitive Intelligence
Research date: 2026-09-11. Read-only research; nothing pushed to Kaggle.
Artifacts in this directory: kernels_raw.json, kernels_ranked.json, kernel_attach.json,
kernel_meta_full.json, datasets_raw.json, datasets_raw2.json, datasets_all_refs.json,
datasets_view.json, forum_topics.json, competition_meta.json, our_submissions.json,
comp_pages.txt (all 10 competition pages, HTML-stripped).
Notebook sources + DIGEST.py (cell-by-cell text) under nb/.

## ENUMERATION COMPLETENESS WARNING (self-correction)
The dataset list at `datasets_raw.json` (94 refs) was built from a fixed set of 10 search terms and is a
**ceiling, not a floor**. A second pass with 11 further terms (`rsna-knee-2`, `knee checkpoints`, `knee oof`,
`knee pseudo`, `coatnet knee`, `knee dinov3`, etc.) found **38 additional refs** — total 132, in
`datasets_raw2.json` / `datasets_all_refs.json`. The search endpoint's coverage is term-dependent, so any
enumeration built from a fixed term set should be assumed incomplete. The 132 figure is itself only a
lower bound; per-owner listing (`datasets/list?user=<owner>`) is the more reliable primitive.

## HARD LIMITATION, STATED UP FRONT
The Kaggle public API does **not** expose public-LB scores for notebooks in this competition.
Verified absent: `kernels/list` (no score field at all — full key list is
`id, ref, title, author, slug, lastRunTime, language, kernelType, isPrivate, enableGpu, enableTpu,
enableInternet, categoryIds, datasetDataSources, kernelDataSources, competitionDataSources,
modelDataSources, totalVotes, currentVersionNumber, dockerImage, machineShape`),
`kernels/pull` metadata (same), `kernels/output` (404), `kernels/view` (404), `kernels/status` (404),
and `competitions/{slug}/leaderboard` (404/400).
So the "previous public LB rank" ordering used by the website is **not** retrievable directly.
Two proxies were used instead, both stated per row in the table below:
  (A) `sortBy=scoreDescending` ordering — note `totalVotes` also changes in this order, so it is a
      composite popularity/score ordering, NOT a pure score ordering. Treat as approximate.
  (B) Explicit LB numbers written by the authors themselves (notebook title or markdown), which are
      the only hard score evidence available.
Discussion bodies are **not retrievable**: `/discussion/{id}` and `/topics/{id}` return 404, and the
HTML page for a discussion returns a 5,680-byte JavaScript shell with zero occurrences of the post
text. `web_fetch` on the same URLs returns only "RSNA Knee Abnormality Detection | Kaggle".
The topic *list* works and returns 20 of 108 topics (hard cap; pageSize/page/sortBy/offset are all
ignored — every variant returned the same 20 ids). Those 20 titles are reported in section 3.
