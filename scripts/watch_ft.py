# -*- coding: utf-8 -*-
"""
Parameterised watcher for a CoAtNet warm-start fine-tune kernel (msft2 lineage).

    python work/watch_ft.py --kid ppxl16/rsna-knee-v8ft --tag v8ft0 --bar 0.8963 [--beat 0.0084]

Reads the pre-registration that matters (FACTS C90/C91/C92/C93):
  * msft2's own guard refuses a warm start scoring below `--bar` on our corpus, so a
    SystemExit naming that bar is a CLOSED line, not a crash to debug.
  * the solo score decides NOTHING. C90: a solo +0.0065 arm measured +0.0007 as a member.
    The only decisive number is the member delta, which this watcher cannot compute --
    it prints the exact command that can, from local OOF files, with no GPU.

Transient-network hardened (see C92-adjacent note: the first version of this family died
on one SSL blip while its kernel was healthy).
"""
import argparse
import json
import pathlib
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent


def cli(*args):
    last = ""
    for attempt in range(4):
        try:
            p = subprocess.run(["kaggle", *args], capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
            out = (p.stdout or "") + (p.stderr or "")
            if out.strip():
                return out
            last = out
        except Exception as e:                      # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
        time.sleep(20 * (attempt + 1))
    return "NETFAIL " + last


def is_transport_failure(low: str) -> bool:
    """True when `low` is a NETWORK failure, not a kernel verdict.

    This check MUST run before any `"error" in low` test. Learned the hard way: the kaggle
    CLI does not raise on a network failure -- it prints the requests exception to stderr and
    exits non-zero, so the text comes back as an ordinary result. That text contains
    "SSLError", so a naive `"error" in low` classified a TRANSPORT failure as a TERMINAL
    kernel state and the watcher announced "did NOT complete" for a kernel that was RUNNING.
    A false "the kernel failed" verdict is worse than no verdict: it invites closing a live
    line. (Observed 2026-10-07 on kernel-v8ft, which was healthy throughout.)
    """
    return any(t in low for t in (
        "netfail", "sslerror", "ssl:", "unexpected_eof", "max retries exceeded",
        "connectionpool", "connection refused", "connection aborted", "timed out",
        "temporary failure", "introspecttoken", "proxyerror", "remotedisconnected",
        "read timed out", "name or service not known",
    ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kid", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--bar", type=float, default=0.8963,
                    help="msft2's guard bar: a warm start below this is refused")
    ap.add_argument("--beat", type=float, default=0.0084,
                    help="msft2's member delta, the number to beat")
    ap.add_argument("--wait-push", type=int, default=60)
    ap.add_argument("--polls", type=int, default=80)
    a = ap.parse_args()

    out = ROOT / "state" / a.tag
    for i in range(a.wait_push):
        s = cli("kernels", "status", a.kid)
        low = s.lower()
        if "netfail" in low or "not found" in low or "404" in low:
            print(f"[{time.strftime('%H:%M:%S')}] not up yet ({i+1}/{a.wait_push})", flush=True)
            time.sleep(240)
            continue
        break

    for i in range(a.polls):
        s = cli("kernels", "status", a.kid)
        low = s.lower()
        if is_transport_failure(low):
            print(f"[{time.strftime('%H:%M:%S')}] transport failure ({i+1}/{a.polls}), "
                  f"retrying -- NOT a kernel verdict", flush=True)
            time.sleep(240)
            continue
        print(f"[{time.strftime('%H:%M:%S')}] {i+1}/{a.polls}  "
              f"{s.strip().splitlines()[-1] if s.strip() else '(none)'}", flush=True)
        if "complete" in low:
            break
        if "error" in low or "fail" in low or "cancel" in low:
            print(f"\n*** {a.tag} did NOT complete. Read the log; do NOT guess:", flush=True)
            print(f"    kaggle kernels logs {a.kid}", flush=True)
            print(f"    A line naming 'scored X vs bar {a.bar}' is msft2's GUARD FIRING,", flush=True)
            print("    i.e. the warm start is worse than v5's on our corpus -> line closed.", flush=True)
            return
        time.sleep(240)
    else:
        print("still running after the poll budget; re-run this watcher", flush=True)
        return

    out.mkdir(parents=True, exist_ok=True)
    print(cli("kernels", "output", a.kid, "-p", str(out)).strip()[-600:], flush=True)
    recs = sorted(out.rglob(f"raptor_ft_{a.tag}.json"))
    best = None
    for f in recs:
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"unreadable {f.name}: {e}", flush=True)
            continue
        best = j.get("best_gold_auc")
        print(f"  {f.name}: best_gold_auc={best} epochs={j.get('epochs')} "
              f"n_gold={j.get('n_gold')} src={j.get('src')}", flush=True)
    if best is None:
        print("NO RECEIPT. Check the log for the zero-tensor guard or the warm-start bar.",
              flush=True)
        return

    print("\n" + "=" * 88)
    print(f"{a.tag} fine-tuned gold_auc = {float(best):.4f}   (bar was {a.bar})")
    print()
    print("*** THE SOLO NUMBER DECIDES NOTHING.  Run the member test: ***")
    print(f"    python scripts/member_add_test.py --arm {a.tag}@f0 --strong")
    print(f"    bar to beat = msft2's +{a.beat}   (C91; CI [+0.0060,+0.0107])")
    print("    C90: a solo +0.0065 arm measured +0.0007 as a member. Solo higher != useful.")
    (ROOT / "state" / f"{a.tag}_result.json").write_text(json.dumps(
        {"tag": a.tag, "best_gold_auc": float(best), "bar": a.bar, "beat": a.beat},
        indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
