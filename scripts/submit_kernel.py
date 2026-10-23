"""Submit a completed RSNA kernel version to the code competition, then log it.

Refuses unless the kernel's CURRENT status is COMPLETE: for a code competition the
kernel must have really run (FACTS A7), and a transport failure must never be read as
a verdict (A173/A174). Acceptance of the submission is judged by the CLI's own success
line plus a submission ref, never by the bare return code (A28/A32/A179).

Usage:
    python work/submit_kernel.py <kernel-dir> <version> "<message>"
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMP = "rsna-knee-abnormality-detection"
LEDGER = ROOT / "state" / "rsna_submissions.jsonl"

EXPECTED = {
    "kernel-raptor-s6-v1ds": ("ppxl16/rsna-knee-raptor-s6-v1ds", "raptor+S6 fork on compact-v1"),
    "kernel-raptor-s6-fork": ("ppxl16/rsna-knee-raptor-s6-fork", "raptor+S6 fork (public 0.956 recipe)"),
    "kernel-blendgold-restored": ("ppxl16/rsna-knee-blendgold-restored", "blendgold restored 0.951"),
    "kernel-fitted-0954": ("ppxl16/rsna-knee-fitted-0954", "fitted 0.954"),
    "kernel-triplereader-0951": ("ppxl16/rsna-knee-triple-reader-0951", "triple-reader 0.951"),
    "kernel-blendgold-0951": ("ppxl16/rsna-knee-blendgold-0951", "blend-gold 0.951"),
    "kernel-apex-repro": ("ppxl16/rsna-knee-apex-0949", "apex 0.950"),
    "kernel-apex-pure0949": ("ppxl16/rsna-knee-apex-pure0949", "apex pure 0.949"),
}

TRANSPORT = ("Max retries", "SSLError", "UNEXPECTED_EOF", "Timeout", "timed out",
             "Connection aborted", "ConnectionReset", "Read timed out")


def clean(out: str) -> str:
    return "\n".join(l for l in out.splitlines() if "outdated `kaggle`" not in l).strip()


def ledger_refs() -> set[str]:
    """Refs already recorded locally (used to tell 'the submit registered' from 'it did not')."""
    if not LEDGER.is_file():
        return set()
    out = set()
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line).get("ref")
        except json.JSONDecodeError:
            continue
        if r:
            out.add(str(r))
    return out


def kernel_status(slug: str, tries: int = 5) -> str | None:
    for a in range(tries):
        try:
            r = subprocess.run(["kaggle", "kernels", "status", slug], capture_output=True,
                               text=True, encoding="utf-8", errors="replace", timeout=300)
        except Exception:  # noqa: BLE001
            time.sleep(10 * (a + 1))
            continue
        o = clean((r.stdout or "") + (r.stderr or ""))
        if o and not any(t in o for t in TRANSPORT):
            m = re.search(r"KernelWorkerStatus\.([A-Z_]+)", o)
            return m.group(1) if m else o[:160]
        time.sleep(10 * (a + 1))
    return None


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(__doc__)
        return 2
    name, version, message = argv[1], argv[2], argv[3]
    if name not in EXPECTED:
        print(f"REFUSING: {name!r} not in allow-list {sorted(EXPECTED)}")
        return 2
    slug, label = EXPECTED[name]

    md = ROOT / "kernels" / name / "kernel-metadata.json"
    got = json.loads(md.read_text(encoding="utf-8")).get("id")
    if got != slug:
        print(f"REFUSING: metadata id {got!r} != {slug!r}")
        return 2

    st = kernel_status(slug)
    print(f"{slug} v{version} status = {st!r}")
    if st is None:
        print("REFUSING: status UNKNOWN after retries (transport failure, NOT a verdict)")
        return 1
    if st != "COMPLETE":
        print(f"REFUSING: status is {st}, not COMPLETE -- a code competition needs a real run (A7)")
        return 1

    print(f"submitting {slug} v{version} ...")
    r = subprocess.run(
        ["kaggle", "competitions", "submit", "-c", COMP, "-k", slug, "-v", version,
         "-f", "submission.csv", "-m", message],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800,
    )
    out = clean((r.stdout or "") + (r.stderr or ""))
    print(out[:1200])

    if any(t in out for t in TRANSPORT) or not out:
        # A181, reproduced 2/2 times on this client: the code-competition submit reliably
        # loses its response to an SSL EOF even though the submission DID register.
        # Never retry (that burns a second slot of the 5/UTC-day) -- reconcile instead.
        print("\n!! TRANSPORT FAILURE during submit -- NOT a result. Reconciling against the server...")
        before = ledger_refs()
        rec = subprocess.run([sys.executable, str(ROOT / "work" / "reconcile_submissions.py"),
                              "--append"], capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=1200)
        print(clean(rec.stdout or "")[-800:])
        new = ledger_refs() - before
        if new:
            print(f"RECONCILED: the submission DID register -- ref(s) {sorted(new)}")
            print("(do NOT resubmit; the slot is already spent)")
            return 0
        print("no new submission appeared on the server -- the submit genuinely did not go through")
        return 1
    ok = re.search(r"(Successfully submitted|successfully submitted)", out)
    ref = re.search(r"\b(5\d{7})\b", out)
    if not ok and not ref:
        print(f"\nSUBMIT FAILED (rc={r.returncode}) -- no success line and no submission ref")
        print("(a bare rc==0 is not evidence: A28/A32/A179)")
        return 1

    row = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ref": ref.group(1) if ref else None,
        "file": "submission.csv",
        "desc": f"[{label}] {message}",
        "status": "PENDING",
        "score": None,
        "private": None,
        "kernel": f"{slug}:{version}",
    }
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nLOGGED to {LEDGER.name}: {json.dumps(row, ensure_ascii=False)}")
    print("score will appear in 10-11 h (A7); rsna_daily.py refreshes it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
