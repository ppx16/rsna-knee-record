# -*- coding: utf-8 -*-
"""Poll kernel-eval-mirror, then fetch and print both receipts (plain vs mirrored)."""
import json
import pathlib
import subprocess
import time

KID = "ppxl16/rsna-knee-eval-mirror"
OUT = pathlib.Path(__file__).resolve().parent.parent / "state" / "eval_mirror"


def cli(*a):
    last = ""
    for attempt in range(4):
        try:
            p = subprocess.run(["kaggle", *a], capture_output=True, text=True,
                               encoding="utf-8", errors="replace",
                               cwd=r"D:\kaggle\rsna-knee")
            o = (p.stdout or "") + (p.stderr or "")
            if o.strip():
                return o
            last = o
        except Exception as e:                      # noqa: BLE001
            last = repr(e)
        time.sleep(20 * (attempt + 1))
    return "NETFAIL " + last


def main():
    for i in range(60):
        s = cli("kernels", "status", KID)
        low = s.lower()
        if any(t in low for t in ("netfail", "sslerror", "unexpected_eof", "max retries")):
            print(f"[{time.strftime('%H:%M:%S')}] transport failure ({i+1}/60), retrying",
                  flush=True)
            time.sleep(300)
            continue
        print(f"[{time.strftime('%H:%M:%S')}] {i+1}/60  "
              f"{s.strip().splitlines()[-1] if s.strip() else '(none)'}", flush=True)
        if "complete" in low or "error" in low:
            break
        time.sleep(300)
    else:
        print("still running", flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    print(cli("kernels", "output", KID, "-p", str(OUT)).strip()[-400:], flush=True)
    got = {}
    for f in sorted(OUT.rglob("raptor_evalonly_*.json")):
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"unreadable {f.name}: {e}", flush=True)
            continue
        got[j.get("tag")] = j.get("gold_auc")
        print(f"  {j.get('tag'):<16} gold_auc={j.get('gold_auc')} n_gold={j.get('n_gold')} "
              f"k_eval={j.get('k_eval')}", flush=True)
    p, m = got.get("ev_v5_plain"), got.get("ev_v5_mirror")
    print()
    if p is None or m is None:
        print("  one or both passes are MISSING -- read the kernel log", flush=True)
        return
    print(f"  plain  {p:.4f}")
    print(f"  mirror {m:.4f}")
    print(f"  delta  {m - p:+.4f}")
    print("  NOTE: this is the SOLO delta on gold-58 (+/-0.021 instrument). The decisive")
    print("  number is the MEMBER delta on the 870 instrument: average the two gold npz")
    print("  files and run scripts/member_add_test.py --strong. See SPEC-2026-10-08.")


if __name__ == "__main__":
    main()
