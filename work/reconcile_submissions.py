"""Reconcile the Kaggle submissions API against our local ledger.

Written because of a real incident (2026-10-10): `kaggle competitions submit` LOST ITS
RESPONSE to an SSL EOF, so we could not tell whether the submission had registered.
It HAD (ref 57026328). Retrying would have burned a second slot out of the 5/UTC-day.

So the rule is: never conclude "the submit failed" from a transport error -- read the
server's own list of submissions and reconcile. This script does exactly that and is
idempotent: it only appends refs that are not already in the ledger.

Usage:
    python work/reconcile_submissions.py            # report
    python work/reconcile_submissions.py --append   # append missing rows
"""
from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMP = "rsna-knee-abnormality-detection"
LEDGER = ROOT / "state" / "rsna_submissions.jsonl"
TRANSPORT = ("Max retries", "SSLError", "UNEXPECTED_EOF", "Timeout", "timed out",
             "Connection aborted", "ConnectionReset", "Read timed out")


def api_submissions() -> list[dict] | None:
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
        if out and not any(t in out for t in TRANSPORT):
            rows = list(csv.DictReader(io.StringIO(out)))
            return [r for r in rows if r.get("ref")]
        time.sleep(15 * (a + 1))
    return None


def ledger_rows() -> list[dict]:
    if not LEDGER.is_file():
        return []
    out = []
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def main(argv: list[str]) -> int:
    do_append = "--append" in argv
    api = api_submissions()
    if api is None:
        print("!! TRANSPORT FAILURE reading the submissions API -- this is NOT 'no submissions'")
        return 1

    ledger = ledger_rows()
    have = {str(r.get("ref")) for r in ledger}
    print(f"server has {len(api)} submissions; ledger has {len(ledger)} rows")
    missing = [r for r in api if str(r["ref"]) not in have]

    # Updating EXISTING rows matters as much as appending: a submission logged while PENDING
    # keeps score=None forever unless something writes the score back (found 2026-10-10:
    # 56910273 was scored 0.943 on the server but still None locally).
    by_ref = {str(r.get("ref")): r for r in ledger}
    updates = []
    for a in api:
        ref = str(a["ref"])
        row = by_ref.get(ref)
        if row is None:
            continue
        new_score = float(a["publicScore"]) if a.get("publicScore") else None
        old_score = row.get("score")
        new_status = (a.get("status") or "").replace("SubmissionStatus.", "") or None
        if new_score is not None and old_score != new_score:
            updates.append((ref, old_score, new_score, new_status))

    if not missing and not updates:
        print("ledger is complete and all scores are current -- nothing to reconcile")
        return 0

    if missing:
        print(f"\n{len(missing)} submission(s) on the server but NOT in our ledger:")
        for r in missing:
            print(f"  {r['ref']}  {r['date']}  score={r.get('publicScore')!r}  {r['description'][:80]}")
    if updates:
        print(f"\n{len(updates)} ledger row(s) whose score needs updating:")
        for ref, old, new, st in updates:
            print(f"  {ref}: score {old!r} -> {new!r}  status={st}")

    if not do_append:
        print("\n(run again with --append to write these)")
        return 0

    # Rewrite in place so appends and updates land together and order is preserved.
    for r in sorted(missing, key=lambda x: x["date"]):
        row = {
            "date": r["date"].replace(" ", "T") + "Z",
            "ref": str(r["ref"]),
            "file": r.get("fileName") or "submission.csv",
            "desc": r["description"],
            "status": (r.get("status") or "COMPLETE").replace("SubmissionStatus.", ""),
            "score": float(r["publicScore"]) if r.get("publicScore") else None,
            "private": float(r["privateScore"]) if r.get("privateScore") else None,
            "reconciled": True,
        }
        ledger.append(row)
        print(f"  appended {row['ref']} score={row['score']}")

    for ref, old, new, st in updates:
        by_ref[ref]["score"] = new
        if st:
            by_ref[ref]["status"] = st
        by_ref[ref]["reconciled"] = True
        print(f"  updated  {ref} score={new}")

    with LEDGER.open("w", encoding="utf-8") as fh:
        for r in ledger:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(ledger)} rows to {LEDGER.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
