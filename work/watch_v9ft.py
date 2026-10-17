# -*- coding: utf-8 -*-
"""
Watch the v9 warm-start fine-tune (kernel-v9ft) and read it against the pre-registration.

Step (3) of the objective was the only untried variant in the two directions. It is built
(work/build_v9ft.py proves a 3-line diff vs kernel-msft2, so the only variable is the
starting checkpoint) and pushed by work/push_v9ft_when_free.py once a GPU session frees.
Until this script existed, nothing collected its result.

Pre-registered read rule (docs/PREREG-2026-10-07-v9-finespacing-two-directions.md):
    fine-tuned gold_auc  >  0.9031   (must beat its own starting checkpoint)
    AND                  >= 0.9069   (must beat msft2, the v5 warm-start)
    Below either -> the v5 warm-start stays the only measured gain, and this line stops.

Bounded on purpose: it waits for the kernel to exist, then polls to a round cap and exits.
It is a wait on one specific blocker, not a scheduled task.
"""
import json
import pathlib
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
KID = "ppxl16/rsna-knee-v9ft"
OUT = ROOT / "state" / "v9ft"
START_AUC = 0.9031     # the v9 checkpoint's own reported gold_auc
MSFT2_AUC = 0.9069     # msft2 = the v5 warm-start, our only board-measured gain


def cli(*args):
    """Call the Kaggle CLI, retrying transient network failures.

    A single SSL blip used to kill this watcher outright: on 2026-10-07 it died at poll
    5/80 with `SSLError(SSLEOFError(8, 'UNEXPECTED_EOF_WHILE_READING'))` while the kernel
    was still perfectly healthy, and the measurement was lost. A watcher whose whole job is
    to survive a long wait must not be less durable than the thing it waits on. On total
    failure it returns a NETFAIL sentinel, which the loops treat as 'keep polling'.
    """
    last = ""
    for attempt in range(4):
        try:
            p = subprocess.run(["kaggle", *args], capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
            out = (p.stdout or "") + (p.stderr or "")
            if out.strip():
                return out
            last = out
        except Exception as e:                      # noqa: BLE001 -- any transport failure
            last = f"{type(e).__name__}: {e}"
        time.sleep(20 * (attempt + 1))
    return "NETFAIL " + last


def main():
    # 1. wait until the pusher has actually created the kernel
    for i in range(60):
        out = cli("kernels", "status", KID)
        low = out.lower()
        if "netfail" in low:
            print(f"[{time.strftime('%H:%M:%S')}] network down ({i+1}/60), retrying", flush=True)
            time.sleep(240)
            continue
        if "kernelwork" in low or "complete" in low:
            break
        if "not found" in low or "404" in low or "permission" in low:
            print(f"[{time.strftime('%H:%M:%S')}] v9ft not pushed yet ({i+1}/60)", flush=True)
            time.sleep(240)
            continue
        print(f"[{time.strftime('%H:%M:%S')}] status: {out.strip().splitlines()[-1]}", flush=True)
        break
    else:
        print("v9ft never appeared; nothing to watch", flush=True)
        return

    # 2. poll to a terminal state
    for i in range(80):
        out = cli("kernels", "status", KID)
        low = out.lower()
        if "netfail" in low:
            print(f"[{time.strftime('%H:%M:%S')}] network down ({i+1}/80), retrying", flush=True)
            time.sleep(240)
            continue
        tail = out.strip().splitlines()[-1] if out.strip() else "(no output)"
        print(f"[{time.strftime('%H:%M:%S')}] {i+1}/80  {tail}", flush=True)
        if "complete" in low:
            break
        if "error" in low or "fail" in low or "cancel" in low:
            print("v9ft did NOT complete; read the log, do not guess", flush=True)
            return
        time.sleep(240)
    else:
        print("still running after the poll budget; re-run this watcher", flush=True)
        return

    # 3. collect and read
    OUT.mkdir(parents=True, exist_ok=True)
    print(cli("kernels", "output", KID, "-p", str(OUT)).strip()[-800:], flush=True)

    best = None
    for f in sorted(OUT.rglob("raptor_ft_v9ft0.json")):
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"unreadable {f.name}: {e}", flush=True)
            continue
        best = j.get("best_gold_auc")
        print(f"  {f.name}: best_gold_auc={best} epochs={j.get('epochs')} "
              f"n_train={j.get('n_train')} n_gold={j.get('n_gold')} src={j.get('src')}", flush=True)
    if best is None:
        print("NO RECEIPT (raptor_ft_v9ft0.json). Check the log for the zero-tensor guard -- a "
              "SystemExit naming '0 backbone tensors' means the warm start did not load.",
              flush=True)
        return

    best = float(best)
    v_start = best > START_AUC
    v_msft2 = best >= MSFT2_AUC
    print("\n" + "=" * 88)
    print("READ AGAINST THE PREREG")
    print(f"  v9ft0 fine-tuned gold_auc = {best:.4f}")
    print(f"  >  own start  {START_AUC}  -> {'YES' if v_start else 'NO'}")
    print(f"  >= msft2      {MSFT2_AUC}  -> {'YES' if v_msft2 else 'NO'}")
    print()
    print("  *** THIS IS NOT THE CLEAN A/B THE BUILD SCRIPT CLAIMED -- READ BEFORE CONCLUDING ***")
    print("  The 3-line diff makes the STARTING CHECKPOINT the only changed *code* input, but the")
    print("  two starts are not geometry-matched to this corpus (FACTS A172, scripts/dump_arms.py):")
    print("      maxspan-v5     native geometry = 336 / 64 slots  <- MATCHES knee-raptor-corpus")
    print("      finespacing-v9 native geometry = 336 / 80 slots  <- does NOT match")
    print("  scripts/corpus64_manifest.py pins the corpus at SLOTS=64 / RES=336 and verifies the")
    print("  shards against [*, 64, 336, 336], so 80 slots CANNOT be built from it.")
    print("  => msft2 was a geometry-MATCHED warm start. v9ft is a geometry-CHANGED one: the")
    print("     attention head must relearn slot statistics over 64 slots instead of 80.")
    print("  => so a LOW v9ft number is confounded -- it may be the geometry change, not v9")
    print("     being a worse starting point. Do NOT record it as 'v9 is a weaker start'.")
    print("  => and the same limit is why the eval-only v9 number was invalid, not pessimistic.")
    print("  => a geometry-fair v9 test needs an 80-slot corpus, which the community-stack path")
    print("     (rendering slots from DICOM) can do and the Raptor trainer path cannot.")
    if v_start and v_msft2:
        print("  => the v9 warm start BEATS both bars. Next: is it worth a submission slot?")
        print("     The member-level delta is NOT locally measurable (870-OOF CI is +/-0.004 and")
        print("     the 11x decay law puts the expected gain near +0.0002), so this is a belief")
        print("     bet on the board, not a measurement. Say so when proposing it.")
    else:
        print("  => below the bar. The v5 warm start (msft2) stays the only measured gain and")
        print("     this arm line stops here. Record it and do not re-open.")
    (ROOT / "state" / "v9ft_result.json").write_text(json.dumps(
        {"tag": "v9ft0", "best_gold_auc": best, "start": START_AUC, "msft2": MSFT2_AUC,
         "beats_start": v_start, "beats_msft2": v_msft2}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
