import sys, csv, pathlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
p = pathlib.Path("work/newds/competition_intelligence_panel.csv")
if not p.exists():
    print("panel not downloaded"); raise SystemExit
rows = list(csv.DictReader(p.open(encoding="utf-8", errors="replace")))
print("panel rows:", len(rows))
mine = [r for r in rows if "rsna-knee" in (r.get("competition_slug") or "")]
print("rows for our competition:", len(mine))
for r in mine:
    for k in ["scraped_utc","metric_name","metric_direction","public_leaderboard_rows",
              "live_api_team_count","our_rank","our_score","our_score_tie_count",
              "bronze_rank","bronze_score","bronze_tie_count","bronze_tie_fraction",
              "bronze_score_gap","bronze_rank_gap","medal_line_basis",
              "gold_rank","gold_score","silver_rank","silver_score",
              "top1_score","daily_submission_limit","days_to_deadline"]:
        v = r.get(k)
        if v not in (None, ""): print("   %-26s %s" % (k, str(v)[:90]))
    print("   ---")
