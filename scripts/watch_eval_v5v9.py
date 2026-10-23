# -*- coding: utf-8 -*-
"""
Watch the eval-only kernel and read the four measurements against the prereg.

Prereg (docs/PREREG-2026-10-07-v9-finespacing-two-directions.md):
  v5 maxspan @ k_eval 24 must land at 0.9214 +/- 0.005  -> proves loader + geometry.
      If it does NOT, the loader/geometry is wrong and NO conclusion may be drawn from
      the v9 number either. That is the only reason v5 is in the run.
  v9 finespacing @ k_eval 24 is the number that matters for 方向二 (the new member).
      Gate: solo >= 0.90 (A167's member gate).
  the two k_eval=62 runs reproduce the author's own scoring protocol.
"""
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
KID = "ppxl16/rsna-knee-eval-v5v9"
OUT = ROOT / "state" / "eval_v5v9"
POLL_S = 180
MAX_POLLS = 60

PREREG = {
    "ev_v5maxspan":   ("0.9214", 0.005, "control: the loader/geometry proof"),
    "ev_v5maxspan62": (None, None, "author scoring protocol (62 windows)"),
    "ev_v9fine":      ("0.9031", 0.005, "THE new member, comparable to existing arms"),
    "ev_v9fine62":    (None, None, "author scoring protocol (62 windows)"),
}


def cli(*args):
    """Call the Kaggle CLI, retrying transient network failures.

    The sibling watcher (watch_v9ft.py) died outright on one SSL blip
    (`SSLEOFError UNEXPECTED_EOF_WHILE_READING`) while its kernel was healthy, and the
    measurement was lost. This one must not repeat that: on total failure it returns a
    NETFAIL sentinel with rc=1, and the poll loop keeps waiting instead of crashing.
    """
    last = ""
    for attempt in range(4):
        try:
            p = subprocess.run(["kaggle", *args], capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
            out = (p.stdout or "") + (p.stderr or "")
            if out.strip():
                return p.returncode, out
            last = out
        except Exception as e:                      # noqa: BLE001 -- any transport failure
            last = f"{type(e).__name__}: {e}"
        time.sleep(20 * (attempt + 1))
    return 1, "NETFAIL " + last


def main():
    print(f"watching {KID}", flush=True)
    for i in range(MAX_POLLS):
        rc, out = cli("kernels", "status", KID)
        if "netfail" in out.lower():
            print(f"[{time.strftime('%H:%M:%S')}] network down ({i+1}/{MAX_POLLS}), retrying",
                  flush=True)
            time.sleep(POLL_S)
            continue
        tail = out.strip().splitlines()[-1] if out.strip() else "(no output)"
        print(f"[{time.strftime('%H:%M:%S')}] {i+1}/{MAX_POLLS}  {tail}", flush=True)
        low = out.lower()
        if "complete" in low:
            break
        if "error" in low or "fail" in low or "cancel" in low:
            print("kernel did NOT complete; nothing to read", flush=True)
            return
        time.sleep(POLL_S)
    else:
        print("still running after the poll budget; re-run this watcher", flush=True)
        return

    OUT.mkdir(parents=True, exist_ok=True)
    rc, out = cli("kernels", "output", KID, "-p", str(OUT))
    print(f"output download rc={rc}", flush=True)
    print(out.strip()[-1500:], flush=True)

    found = sorted(OUT.rglob("raptor_evalonly_*.json"))
    print(f"\n{len(found)} eval-only receipts", flush=True)
    if not found:
        print("NO RECEIPTS -- read the kernel log, do not guess", flush=True)
        return

    results = {}
    for f in found:
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  unreadable {f.name}: {e}")
            continue
        tag, auc, n = j.get("tag"), j.get("gold_auc"), j.get("n_gold")
        results[tag] = auc
        print(f"  {tag:<18} gold_auc={auc}  n_gold={n}  res={j.get('res')} "
              f"k_eval={j.get('k_eval')}  init={pathlib.Path(str(j.get('init_ckpt'))).name}")

    print("\n" + "=" * 90)
    print("READ AGAINST THE PREREG")
    print("=" * 90)
    control_ok = None
    for tag in ("ev_v5maxspan", "ev_v9fine"):
        if tag not in results or results[tag] is None:
            print(f"  {tag}: MISSING -- no read")
            continue
        got = float(results[tag])
        exp, tol, why = PREREG[tag]
        exp = float(exp)
        ok = abs(got - exp) <= tol
        if tag == "ev_v5maxspan":
            control_ok = ok
        verdict = "AS PREDICTED" if ok else ("HIGHER" if got > exp else "LOWER")
        print(f"  {tag:<18} got {got:.4f}  expected {exp:.4f}+/-{tol}  -> {verdict}   ({why})")

    print()
    if control_ok is False:
        print("  *** The v5 CONTROL did not reproduce. The loader or the geometry is wrong.")
        print("  *** Do NOT read the v9 number as 'v9 is weak'. Fix the setting first.")
    elif control_ok:
        print("  -> control reproduced: the loader and the corpus geometry are correct, so the")
        print("     v9 number below is a real measurement of that checkpoint on OUR instrument.")
        v9 = results.get("ev_v9fine")
        if v9 is not None:
            print(f"  -> v9 finespacing gold-58 = {float(v9):.4f}")
            print(f"     A167 member gate is ~0.90 -> "
                  f"{'PASSES' if float(v9) >= 0.90 else 'FAILS'}")
            print()
            print("  *** THE v9 NUMBER FROM THIS KERNEL IS INVALID -- NOT MERELY PESSIMISTIC ***")
            print("  scripts/dump_arms.py cross-checks the author's dataset descriptions and shows")
            print("  each checkpoint belongs to a DIFFERENT corpus:")
            print("      maxspan-v5        (336, 64 slots, span .02-.98) -> 62 windows")
            print("      native384dense-v10(384, 64 slots)               -> 62")
            print("      native384-v8      (384, 44 slots, span .06-.94) -> 42")
            print("      finespacing-v9    (336, 80 slots, span .02-.98) -> 78")
            print("  THIS kernel feeds v9 the 64-slot maxspan corpus at k_eval 24/62. That is exactly")
            print("  the mismatch the author warns about: 'the 0.932 will not be there'.")
            print("  => the v9 figure here measures the WRONG CORPUS, not v9. Do not record it as")
            print("     'v9 is weak' and do not use it to open or close 方向二.")
            print("  => the v5 CONTROL remains valid (v5 and our corpus are BOTH 64-slot maxspan).")
            print("  => to read v9 at all you need an 80-SLOT corpus; the precomputed")
            print("     knee-raptor-corpus is 64-slot, so the Raptor trainer cannot supply it.")
            print("     (D14's add-finespacing-arm DID set 80 slots / k_eval 78 -- check whether that")
            print("      path rendered slots from DICOM instead of the precomputed corpus.)")
            print()
            print("  Prereg verdict on this run: it establishes v5's loader+geometry, and gives")
            print("  v9 an ABSOLUTE number. It CANNOT settle whether v9 helps as a member --")
            print("  no local instrument resolves that scale (870-OOF CI +/-0.004 vs an expected")
            print("  +0.0002 from the 11x decay law). Any proposal to submit is a belief bet.")


if __name__ == "__main__":
    main()
