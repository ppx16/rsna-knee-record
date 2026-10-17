"""Watch the two pending RSNA submissions until Kaggle publishes their public scores.

Rank resolution is 0.001, and which rank we get INSIDE the 0.950 block depends on the
unrounded score (FACTS C113), so the moment the real number lands it must be written down.

Behaviour:
  * polls the submissions API, retrying transport failures in-call (A173/A174);
  * a transport failure is reported as UNKNOWN, never as "no score";
  * exits 0 as soon as every watched ref has a public score;
  * exits 2 on the time limit (still pending);
  * on exit it reconciles the local ledger (A181: the server list is the only truth).

Usage: python work/watch_scores.py [ref ...] [--max-min N]
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMP = "rsna-knee-abnormality-detection"
DEFAULT_REFS = ["57026328", "57026501"]   # apex, triple-reader
TRANSPORT = ("Max retries", "SSLError", "UNEXPECTED_EOF", "Timeout", "timed out",
             "Connection aborted", "ConnectionReset", "Read timed out")


def fetch() -> list[dict] | None:
    for a in range(5):
        try:
            r = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP, "--csv"],
                               capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=600)
        except Exception:  # noqa: BLE001
            time.sleep(15 * (a + 1))
            continue
        out = "\n".join(l for l in ((r.stdout or "") + (r.stderr or "")).splitlines()
                        if "outdated `kaggle`" not in l).strip()
        if not out or any(t in out for t in TRANSPORT):
            time.sleep(15 * (a + 1))
            continue
        rows = []
        for line in out.splitlines()[1:]:
            parts = re.split(r",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", line)
            if len(parts) >= 6 and parts[0].strip().isdigit():
                rows.append({"ref": parts[0].strip(), "date": parts[2].strip(),
                             "status": parts[4].strip(), "score": parts[5].strip().strip('"')})
        return rows
    return None


def main(argv: list[str]) -> int:
    max_min = 720.0
    if "--max-min" in argv:
        i = argv.index("--max-min")
        max_min = float(argv[i + 1])
        del argv[i:i + 2]
    refs = [a for a in argv[1:] if a.isdigit()] or DEFAULT_REFS

    print(f"watching {refs} for public scores (limit {max_min:.0f} min)")
    t0 = time.time()
    while True:
        el = (time.time() - t0) / 60.0
        rows = fetch()
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%SZ")
        if rows is None:
            print(f"[{stamp} +{el:.0f}m] TRANSPORT FAILURE -- score status UNKNOWN "
                  f"(NOT 'still pending' as a fact, just unread)", flush=True)
        else:
            by = {r["ref"]: r for r in rows}
            done = 0
            for ref in refs:
                r = by.get(ref)
                if r is None:
                    print(f"[{stamp} +{el:.0f}m] {ref}: NOT on the server's list", flush=True)
                    continue
                sc = r["score"] or None
                if sc:
                    done += 1
                print(f"[{stamp} +{el:.0f}m] {ref}: status={r['status']} score={sc}", flush=True)
            if done == len(refs):
                print("\n=== all watched submissions have a public score ===")
                subprocess.run([sys.executable, str(ROOT / "work" / "reconcile_submissions.py"),
                                "--append"], check=False)
                return 0
        if el > max_min:
            print(f"\n=== still pending after {el:.0f} min; giving up this watch ===")
            return 2
        time.sleep(300)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
