# -*- coding: utf-8 -*-
"""Build the TPU training notebook from the baseline (torch_xla, 336x12 + DINOv2-large).

What changes vs the GPU training version (build_train_notebook.py):

  1. The device path moves entirely inside a spawned `_train_fn` and `_train_fn` is the
     FIRST thing that touches the XLA device (torch_xla forbids touching xla_device before
     xmp.spawn; the diagnostic confirmed the failure mode). The decode/cache build stays in
     the parent (pure numpy) and is handed to the workers through the module-level `G` via
     fork, which shares it copy-on-write - decode runs once, four workers read the same
     bytes.
  2. Data parallelism: each worker trains its own shard of the batch list (stride = world),
     and xm.optimizer_step all-reduces the gradients, so the four cores act as one optimizer
     with an effective batch of world * BATCH_STUDIES. `steps_per_epoch` is a floor so every
     core runs the same number of collectives and never desyncs.
  3. CUDA coupling -> XLA: `torch.autocast("cuda", enabled=...)` -> `("xla", enabled=False)`
     i.e. fp32 everywhere (fp32 so TPU-trained weights load on a T4 without WeightsError).
     No GradScaler, no torch.cuda.empty_cache (xm.mark_step). The per-step loss .item() is
     REMOVED entirely: it native-segfaulted even in fp32 by re-executing the large forward
     after backward freed it (leaking ~10G/step). Loss logging is gone; holdout AUC is the
     monitor.

     Device -> host transfers now happen ONE FORWARD AT A TIME. Every run from v8 to v16
     native-aborted inside `predict`, always at the single `.cpu()` that follows a
     device-side accumulation of all N_GROUP logits: that one transfer materialises
     N_GROUP DINOv2-large forwards in a single execution, and a v3 chip's ~16G does not
     hold them (the heartbeat file stopped at `b0_cpu` with no further progress).
     `predict` and `predict_member` now mark_step + `.cpu()` each group separately and sum
     the logits on the host; `fingerprint`, the best-state snapshot and save_member
     mark_step before their transfer for the same reason. Training was never affected
     because a training step materialises exactly one forward - which is also why
     "one forward per execution" is the fix rather than a retry.
  4. Fingerprint stays fp32 (no autocast) and is computed on the TPU, so train+infer both on
     TPU are self-consistent. The inference side is ported separately; the training run does
     not call it (no weights attached -> find_weights() is None).
  5. Config: CACHE_IMG stays 336, RUNS -> single r336, N_GROUP_MAX -> 4, and - the one that
     actually matters on the bigger machine - CACHE_BUDGET_MAX_GB 24 -> 60. That constant is
     a HARD ceiling in plan_cache regardless of reported RAM, so without it the TPU silently
     clamps to 6 slices exactly like a T4.

Run with enable_internet=True, enable_tpu=True, machine_shape Tpu1VmV38, and no weights
dataset attached. The first run is a validation of the port; N_FOLDS/EPOCHS default to a
conservative 3/3 and should be tuned up once the first run reveals the real epoch time.
"""

# --- GBK CONSOLE GUARD (see work/patch_gbk_guard.py) -------------------------------------------------
# This console is cp936, which cannot encode the U+26A0 warning sign used below. An unguarded
# `print("... \u26a0 ...")` raises UnicodeEncodeError at that line and kills the script MID-REPORT, while
# everything printed earlier is already on screen -- so the run looks complete. Self-contained on purpose:
# it imports `sys` itself and depends on nothing else in this tree.
import sys as _sys

try:
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001  -- non-reconfigurable stream: leave it alone, prints still degrade
    pass
# ----------------------------------------------------------------------------------------------------
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=2026,
                help="per-run seed; re-run with another seed to grow the ensemble")
ap.add_argument("--epochs", type=int, default=3,
                help="target epochs per fold")
ap.add_argument("--folds", type=int, default=3,
                help="report-md5 CV folds (one trained member each); must be >= 2 - "
                     "with a single fold every study lands in the holdout and the "
                     "training set is empty (v18 built with --folds 1 died at si0 with "
                     "an empty batch, after 33 minutes of decoding)")
ap.add_argument("--batch", type=int, default=2,
                help="studies per step; 8 OOMs the 16G/chip HBM at large-336x12 "
                     "(peak ~2.9G/study)")
ARGS = ap.parse_args()
assert ARGS.folds >= 2, f"--folds {ARGS.folds} leaves no training studies; use >= 2"

SRC = Path("notebooks/rsna-knee-baseline-v1.ipynb")
DST = Path("notebooks/rsna-knee-tpu-train-v1.ipynb")

nb = json.loads(SRC.read_text(encoding="utf-8"))
cells = nb["cells"]


def cell_text(c):
    s = c["source"]
    return "".join(s) if isinstance(s, list) else s


def set_cell_text(c, text):
    c["source"] = text


# ---- config edits (constants cell) ----------------------------------------- #
RUNS_OLD = '''# Resolution is the axis under test. A feature of width d mm survives resampling only if
# the pixel pitch is at most d/2, and the pitch here is set by the crop above rather than
# by the acquired field of view: CROP_MM / P. At 224 px that is 0.58 mm, above the 0.5 mm
# a 1 mm tear needs; at 336 px it is 0.39 mm and clears it. Both configurations read the
# same cache, so the comparison isolates the resize.
RUNS = [
    {"name": "r224", "img": 224},
    {"name": "r336", "img": 336},
]'''

RUNS_NEW = '''# Resolution. The weights package identifies a member by one resolution where the pixel
# cache and the model input must agree, so this run trains a single 336 px config (the
# cache CACHE_IMG below is already 336). The TPU has enough RAM for the full 12-slice
# decode, so the only axis left to push is the backbone (large below) and the member count
# (N_FOLDS x seeds).
RUNS = [
    {"name": "r336", "img": 336},
]'''

CONFIG_EDITS = [
    ("SEED = 2026",
     f"SEED = int(os.environ.get(\"SEED\", \"{ARGS.seed}\"))  # per-run seed; re-run with another SEED to grow the ensemble",
     "SEED"),
    ("EPOCHS = 10",
     f"EPOCHS = int(os.environ.get(\"EPOCHS\", \"{ARGS.epochs}\"))   # target epochs per fold (large-336x12 is slow; tune after the first run shows the real epoch time)\nN_FOLDS = int(os.environ.get(\"N_FOLDS\", \"{ARGS.folds}\"))  # report-md5 CV folds; one trained member per fold",
     "EPOCHS/N_FOLDS"),
    ("BATCH_STUDIES = 8",
     f"BATCH_STUDIES = {ARGS.batch}",
     "BATCH_STUDIES"),
    ("EVAL_BATCH = 8",
     f"EVAL_BATCH = {ARGS.batch}   # inference batch; 8 = same 24-slice peak as BATCH_STUDIES=8 that OOM'd the 16G/chip, tie to training batch",
     "EVAL_BATCH"),
    ("N_GROUP_MAX = 1",
     "N_GROUP_MAX = 4         # slice windows per study (slices = GROUP * N_GROUP); TPU RAM fits all 12",
     "N_GROUP_MAX"),
    ("CACHE_BUDGET_MAX_GB = 24.0",
     "CACHE_BUDGET_MAX_GB = 60.0 # hard ceiling in plan_cache: 24 caps the TPU to 6 slices, raise it to fit 12",
     "CACHE_BUDGET_MAX_GB"),
    ("def find_dinov2(variant=\"small\"):",
     "def find_dinov2(variant=\"large\"):",
     "find_dinov2 variant"),
    ("variant=\"small\", pool=\"cls_mean\"",
     "variant=\"large\", pool=\"cls_mean\"",
     "build_model variant"),
]

# ---- main() tail replacement: decode stays in the parent, training is spawned - #
NEW_MAIN_TAIL = '''    # Grouped on report text: assign each study a fold by an md5 of its report so
    # byte-identical reports never straddle a split. The va/tr split itself happens
    # inside _train_fn (one fold at a time); here we only fix the assignment and the
    # study -> index map, then hand the decode + label state to the spawned workers.
    rep = train_df.set_index("StudyInstanceUID")["Report"].fillna("")
    grp = np.array([int(hashlib.md5(rep.get(s, s).encode()).hexdigest()[:8], 16) % N_FOLDS
                    for s in st_tr])
    gpos = {s: i for i, s in enumerate(st_tr)}

    # fork shares these arrays copy-on-write, so the decode (done above, in this parent
    # process) runs once and the four workers read the same bytes.
    G.update(dict(
        st_tr=st_tr, Ctr=Ctr, Mtr=Mtr,
        st_te=st_te, Cte=Cte, Mte=Mte,
        Y=Y, W=W, keep=keep, grp=grp, gpos=gpos, gold=gold,
        test_df=test_df,
    ))

    # Drain the decode's worker state before the fork: the decode ran a ThreadPoolExecutor
    # and (even with OMP/MKL pinned to 1) leaves malloc arenas and idle pools that fork
    # inherits. A gc + thread reset + short settle gives the cleanest fork we can get
    # without moving the decode into a separate process.
    import gc as _gc
    _gc.collect()
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    import time as _time
    _time.sleep(2)

    # Record the parent's RSS right before the fork so we can compare against each
    # worker's "fork_after" sample: if workers' RSS balloons far above the parent's,
    # the COW share of the 33GB decode cache is breaking and we're heading for host OOM.
    try:
        rss = -1
        with open("/proc/self/status") as _f:
            for _l in _f:
                if _l.startswith("VmRSS:"):
                    rss = int(_l.split()[1]); break
        with open("/kaggle/working/parent_mem.txt", "w") as _pf:
            _pf.write(f"parent_before_fork rss_kb={rss}\\n")
    except Exception:
        pass

    # Spawn first: _train_fn is the first thing that touches the XLA device.
    xmp.spawn(_train_fn, start_method="fork")
'''

# ---- the spawned training entry (inserted before try: main()) --------------- #
TPU_CELL = '''# --- TPU training entry (spawned over the v3-8's cores) ---------------------- #
#
# torch_xla forbids touching the XLA device before xmp.spawn, so the whole device path
# lives inside _train_fn and _train_fn is the first thing that touches it. The decode
# cache (Ctr/Mtr/Cte/Mte) is built in the parent (main) as pure numpy and handed to the
# workers through the module-level G via fork, which shares it copy-on-write: the decode
# runs once, the four workers read the same bytes.
#
# Data parallelism: each worker trains its own shard of the batch list and
# xm.optimizer_step all-reduces the gradients, so the four cores act as one optimizer
# with an effective batch of world * BATCH_STUDIES. Every worker runs the identical
# forward over the whole holdout (eval) so selection and the saved checkpoint agree
# across cores; only the master logs, saves members and writes submissions.

# --- Kaggle TPU fix ----------------------------------------------------------- #
# The official Kaggle TPU notebook (distributed-pytorch-xla-basics-with-pjrt) pops
# these two vars BEFORE importing the torch_xla submodules and BEFORE xmp.spawn.
# Kaggle's image pre-sets TPU_PROCESS_ADDRESSES / CLOUD_TPU_TASK_ID for a single
# worker, so PJRT's slice builder sees 1 worker address instead of the 8 the v3-8
# slice needs and xmp.spawn dies with "Expected 8 worker addresses, got 1". The pop
# must happen before the submodule imports (they cache the topology env).
#
# Do NOT force TPU_PROCESS_BOUNDS: on Kaggle the v3-8 is PCI-enumerated as 8
# single-core "chips" (not 4 chips x 2 cores), so num_available_chips() = 8 and the
# correct spawn is 8 processes x 1 core x ~16 GB HBM each. This is exactly what v3
# ran and it reached training (then hit HBM OOM at BATCH_STUDIES=8, fixed by --batch
# 4). Forcing 4 processes ("2,2,1") makes each process try to drive 2 PCI devices and
# crashes XLA init with "terminated abruptly" (v6). Leave the bounds unset -> 8 procs.
import os
os.environ.pop("TPU_PROCESS_ADDRESSES", None)
os.environ.pop("CLOUD_TPU_TASK_ID", None)

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.xla_backend
import torch_xla.experimental.pjrt_backend  # required for torch.distributed on TPU v2/v3
import torch.distributed as dist

try:
    from torch_xla._internal import tpu as _tpu
    _probe = (f"[tpu probe] chips={_tpu.num_available_chips()} "
              f"proc_bounds={_tpu.process_bounds_size()} "
              f"n_procs={_tpu.num_local_processes()}")
    print(_probe, flush=True)
    with open("/kaggle/working/tpu_probe.txt", "w") as _pf:
        _pf.write(_probe + "\\n")
except Exception as _e:
    print(f"[tpu probe] failed: {_e!r}", flush=True)

G = {}

# Holdout studies scored each epoch. The full 1415 studies over four groups is 2832
# forward executions per epoch - about two hours on top of the training, and a Kaggle
# session stops at 12h with the folds and the test predict still to run. 256 studies puts
# the AUC standard error near 0.018, well inside the differences being read off it, and
# scoring the SAME subsample every epoch is what makes an epoch curve comparable.
AUC_EVAL_MAX = int(os.environ.get("AUC_EVAL_MAX", "256"))
AUC_EVAL_SEED = 1234

# Which epochs get the holdout eval. 0 - the default - evaluates only the last epoch of
# each fold: that is the one that can produce the saved member, and the eval costs about
# as much as a third of the epoch it follows. Set it to 1 for the old every-epoch curve
# when diagnosing, and budget the session for it.
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "0"))


def _eval_now(ep, epochs, every):
    return ep == epochs - 1 or (every > 0 and ep % every == 0)


def _phb(msg):
    """Predict-phase heartbeat: one append-only file per process.

    v17 wrote a single truncated file shared by all eight workers, so the last write won
    and the file was left holding `b0_g3_xfer` - with no way to tell whether that was the
    real predict's first batch or the smoke test's only one. Appending cannot lose the
    position and a file per PID cannot race; the whole predict phase is ~2800 markers,
    under 40 KB.
    """
    try:
        with open(f"/kaggle/working/predict_hb_{os.getpid()}.txt", "a") as _f:
            print(msg, file=_f)
    except Exception:
        pass


def _train_fn(index):
    try:
        return _train_fn_inner(index)
    except BaseException:
        import traceback as _tb
        try:
            with open(f"/kaggle/working/worker_{index}_crash.txt", "w") as _cf:
                _cf.write(_tb.format_exc())
        except Exception:
            pass
        raise


def _train_fn_inner(index):
    dev = xm.xla_device()
    import faulthandler
    faulthandler.enable()
    dist.init_process_group("xla", init_method="xla://")
    world = dist.get_world_size()
    rank = index

    def _mem(tag):
        # Append one line of (process RSS, system MemAvailable) to a per-worker file so a
        # later native crash can be told apart: RSS climbing toward MemAvailable=0 is host
        # OOM (OOM-killer SIGKILL, no Python trace), a flat RSS with a sudden crash is the
        # fork-corruption class instead. Ridden on every _mark so all existing heartbeat
        # points get memory samples for free.
        try:
            rss = -1
            with open("/proc/self/status") as _f:
                for _l in _f:
                    if _l.startswith("VmRSS:"):
                        rss = int(_l.split()[1]); break
            avail = -1
            with open("/proc/meminfo") as _f:
                for _l in _f:
                    if _l.startswith("MemAvailable:"):
                        avail = int(_l.split()[1]); break
            with open(f"/kaggle/working/worker_{index}_mem.txt", "a") as _mf:
                _mf.write(f"{tag} rss_kb={rss} avail_kb={avail}\\n")
        except Exception:
            pass

    def _mark(tag):
        try:
            with open(f"/kaggle/working/worker_{index}_hb.txt", "w") as _h:
                _h.write(tag)
        except Exception:
            pass
        _mem(tag)

    def _devmem(tag):
        # HOST memory is already covered by _mem, and v19 ruled it out: RSS was identical
        # to the byte between the last training step and the crash, with 265 GB still
        # available. The remaining suspect is the DEVICE, which _mem cannot see - each v3
        # chip has its own 16 GB of HBM. torch_xla exposes device memory differently
        # across versions, so try each accessor and record whichever answers. A native
        # abort is not catchable, which is exactly why this is written BEFORE the predict
        # rather than after it: if the run dies again, the numbers are already on disk.
        val = None
        try:
            import torch_xla.debug.metrics as _met
            val = _met.memory_info()
        except Exception:
            pass
        if val is None:
            try:
                import torch_xla.core.xla_model as _xm2
                val = _xm2.get_memory_info(dev)
            except Exception:
                pass
        try:
            with open(f"/kaggle/working/worker_{index}_devmem.txt", "a") as _df:
                _df.write(f"{tag} {val}\\n")
        except Exception:
            pass

    _mem("fork_after")

    # Warm up the XLA->host transfer path. The first REAL .cpu() in the whole run is the
    # end-of-epoch predict read, and it crashes natively (a worker dies, the executor
    # SIGTERMs the rest -> BrokenProcessPool). If the transfer manager is broken by the
    # fork-from-dirty-process, this warm-up crashes HERE (right after device init, before
    # the model even loads) and the marker localizes it; if it survives, the transfer path
    # is sound and the predict/best_state/save_member .cpu() transfers below are safe.
    _mark("warmup_pre")
    _ = float(torch.zeros((), device=dev).cpu())
    _mark("warmup_done")

    st_tr = G["st_tr"]; Ctr = G["Ctr"]; Mtr = G["Mtr"]
    st_te = G["st_te"]; Cte = G["Cte"]; Mte = G["Mte"]
    Y = G["Y"]; W = G["W"]; keep = G["keep"]
    grp = G["grp"]; gpos = G["gpos"]; gold = G["gold"]
    test_df = G["test_df"]

    results, test_preds = {}, {}
    _smoked = []
    _repredicted = []

    for fold in range(N_FOLDS):
        va = np.array([i for i in keep if grp[i] == fold])
        tr = np.array([i for i in keep if grp[i] != fold])
        if len(va) == 0 or len(tr) < BATCH_STUDIES:
            cut = max(1, len(keep) // N_FOLDS)
            va = keep[fold * cut:(fold + 1) * cut]
            tr = np.concatenate([keep[:fold * cut], keep[(fold + 1) * cut:]])
        if rank == 0:
            log(f"fold {fold}: train {len(tr)} / holdout {len(va)} studies")

        # The annotated studies stay in training - they are the highest-quality labels in
        # the corpus and there are too few to discard - so the honest annotation check
        # uses only the ones that fell in this fold's holdout.
        va_set = set(va.tolist())
        gi = np.array([gpos[s] for s in gold.index if s in gpos and gpos[s] in va_set])
        gold_y = gold.loc[[st_tr[i] for i in gi]].values.astype(int) if len(gi) else None
        # The AUC is read off a fixed subsample of the holdout; `va` itself stays whole
        # for the annotated check below, which is a different question and is small
        # enough to score in full either way.
        auc_idx = va
        if len(va) > AUC_EVAL_MAX:
            auc_idx = np.sort(np.random.RandomState(AUC_EVAL_SEED).choice(
                va, AUC_EVAL_MAX, replace=False))
        yv = (Y[auc_idx] > 0.5).astype(int)
        if rank == 0:
            log(f"  annotation check: {len(gi)} of {len(gold)} annotated studies in holdout")

        for cfg in RUNS:
            pitch = CROP_MM / cfg["img"]
            if rank == 0:
                log(f"=== fold {fold} {cfg['name']}: {cfg['img']} px, {pitch:.3f} mm/pixel, "
                    f"{pitch * 14:.2f} mm per patch token ===")
            torch.manual_seed(SEED)
            np.random.seed(SEED)
            model = build_model(UNFREEZE_LAST).to(dev)
            xm.broadcast_master_param(model)

            # Prove the eval + transfer path on ONE batch before spending an hour of
            # training on a run that would die at the first predict. predict() is where
            # v8-v16 all died, and it died on the first batch exactly as it would have
            # on the last - a per-execution HBM ceiling, not a slow leak - so one batch
            # here answers the question for about two minutes. Delete once a run has
            # completed end to end.
            if not _smoked:
                _smoked.append(index)
                _mark("smoke_pre")
                _s = predict(model, Ctr, Mtr, keep[:max(EVAL_BATCH, 1)], dev, cfg["img"])
                _mark(f"smoke_done shape={tuple(_s.shape)} mean={float(np.mean(_s)):.4f}")
                del _s
            opt = torch.optim.AdamW([
                {"params": [p for p in model.backbone.parameters() if p.requires_grad],
                 "lr": LR_BACKBONE},
                {"params": model.head.parameters(), "lr": LR_HEAD},
            ], weight_decay=WEIGHT_DECAY)

            # Shard the batch list across the cores. steps_per_epoch is the same on every
            # core (floor), so the all-reduce in optimizer_step never hangs and the workers
            # stay in lockstep; the <= world-1 trailing batches are dropped.
            n_batches = len(tr) // BATCH_STUDIES
            steps_per_epoch = max(n_batches // world, 1)
            steps = max(steps_per_epoch * EPOCHS, 1)
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=[LR_BACKBONE, LR_HEAD], total_steps=steps, pct_start=0.15)

            best, best_state, best_annot, done_epochs = -1.0, None, float("nan"), 0
            for ep in range(EPOCHS):
                model.train()
                perm = np.random.permutation(tr)
                nstep = 0
                for si in range(steps_per_epoch):
                    bi = si * world + rank
                    sel = perm[bi * BATCH_STUDIES:(bi + 1) * BATCH_STUDIES]
                    # The group is chosen first and sliced on the HOST, so only the six
                    # slices this step reads cross to the device - half the bytes, and the
                    # same shape the eval path has always used. The three `_mem` samples
                    # split the step: v17 showed the host leak is entirely in this segment
                    # and growing faster every step (52 MB/step over si1-30, 172 MB/step
                    # over si130-187), while backward frees and the optimizer step costs
                    # exactly nothing. These name the operation instead of the region.
                    g = int(torch.randint(N_GROUP, (1,)).item())
                    rows = torch.from_numpy(np.ascontiguousarray(
                        Ctr[sel, :, g * GROUP:(g + 1) * GROUP])).to(dev)
                    _mem(f"ep{ep} si{si} a_rows")
                    imgs = augment(rows)
                    _mem(f"ep{ep} si{si} b_aug")
                    m = torch.from_numpy(Mtr[sel]).to(dev)
                    y = torch.from_numpy(Y[sel]).to(dev)
                    w = torch.from_numpy(W[sel]).to(dev)
                    _mem(f"ep{ep} si{si} c_ymw")
                    _mark(f"fold{fold} {cfg['name']} ep{ep} si{si} fwd g={g} shape={tuple(imgs.shape)}")
                    loss = (F.binary_cross_entropy_with_logits(
                        model(imgs, m, cfg["img"]), y, reduction="none") * w).mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    _mark(f"fold{fold} {cfg['name']} ep{ep} si{si} bwd")
                    xm.optimizer_step(opt)
                    _mark(f"fold{fold} {cfg['name']} ep{ep} si{si} opt")
                    sched.step()
                    nstep += 1

                # No per-step loss .item(): v15 proved it native-segfaults even in fp32 (it
                # re-executes the large forward after backward freed it, leaking ~10G/step).
                # Loss logging dropped; the holdout AUC (predict) is the monitor. v8 ran 187
                # steps fine with no per-step .item().
                # The holdout eval is both the monitor and the member-selection criterion,
                # and it is not free: one predict over AUC_EVAL_MAX studies costs
                # ~4 * AUC_EVAL_MAX / BATCH_STUDIES forward executions, about twenty
                # minutes at 336x12 on large. Run every epoch and the eval alone outgrows
                # the training it measures (v17's budget: 15 evals over 5 folds x 3 epochs
                # is 16h against a 12h session). It therefore runs on the LAST epoch of
                # each fold, which is the only one that can produce the saved member;
                # EVAL_EVERY > 0 adds intermediate evals when an epoch curve is wanted.
                d, g_auc = float("nan"), float("nan")
                if _eval_now(ep, EPOCHS, EVAL_EVERY):
                    # A diagnostic, not a monitor: it distinguishes "predict fails on the
                    # trained state" from "predict fails on accumulated batches", and the
                    # answer is the same on every fold. One batch, once per process.
                    if not _repredicted:
                        _repredicted.append(ep)
                        _mark(f"ep{ep} repredict_pre")
                        # v20: every run from v8 to v19 died HERE - at the first mark_step()
                        # of the first predict AFTER training. The marker ordering in predict
                        # (mark_step, then _phb("b0_g0_xfer"), then .cpu()) means the fatal
                        # operation is the EXECUTION of that first forward, not the transfer.
                        # Host memory is ruled out (v19: RSS identical to the byte at the
                        # crash, 265 GB available) and so is the forward shape - training
                        # runs (2, 6, 3, 336, 336) 440 times without trouble. What differs is
                        # the STATE: the training graph's buffers are still live when predict
                        # builds a new graph. So measure the device, drop what training left
                        # behind, measure again, then predict. All three lines are written
                        # before the crash point, so a repeat crash still leaves the evidence.
                        _devmem(f"ep{ep} devmem_before")
                        model.zero_grad(set_to_none=True)
                        import gc as _gc
                        _gc.collect()
                        xm.mark_step()
                        _devmem(f"ep{ep} devmem_after_free")
                        _r = predict(model, Ctr, Mtr, auc_idx[:max(EVAL_BATCH, 1)], dev,
                                     cfg["img"])
                        _mark(f"ep{ep} repredict_done mean={float(np.mean(_r)):.4f}")
                        del _r

                    _mark(f"ep{ep} predict_pre")
                    pv = predict(model, Ctr, Mtr, auc_idx, dev, cfg["img"])
                    _mark(f"ep{ep} predict_done")
                    d = macro_auc(yv, pv)
                    if gold_y is not None and len(gi):
                        g_auc = macro_auc(gold_y, predict(model, Ctr, Mtr, gi, dev,
                                                          cfg["img"]))
                    if rank == 0:
                        log(f"  epoch {ep + 1}/{EPOCHS}  holdout {d:.4f}"
                            f"  annot(n={len(gi)}) {g_auc:.4f}")

                # Selection reads the holdout alone. The annotation check is reported
                # because it measures something different, but only a handful of annotated
                # studies land in any one holdout, so it cannot arbitrate between epochs.
                if d > best:
                    _mark(f"ep{ep} best_state_pre")
                    best, best_annot = d, g_auc
                    # Execute the pending graph before the transfer: a .cpu() taken
                    # straight off the lazy parameters pulls whatever is still queued
                    # into the same execution as the copy (see predict).
                    xm.mark_step()
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
                    _mark(f"ep{ep} best_state_done")
                done_epochs = ep + 1

            if best_state is not None:
                model.load_state_dict(best_state)
            key = f"fold{fold}_{cfg['name']}"
            results[key] = (best, best_annot)
            test_preds[key] = predict(model, Cte, Mte, np.arange(len(st_te)), dev,
                                      cfg["img"])
            if rank == 0:
                log(f"  {key}: best holdout {best:.4f} (annot {best_annot:.4f})")
                if best_state is not None:
                    save_member(model, dev, cfg, fold=fold, holdout=best, annot=best_annot,
                                epochs_done=done_epochs, seed=SEED,
                                source_run=f"tpu-train-{cfg['name']}-seed{SEED}")
            del model, opt, sched, best_state
            gc.collect()
            xm.mark_step()

    # ---- summary + submissions, master only ---- #
    if rank == 0:
        log("---- summary ----")
        for n, (d, g_auc) in results.items():
            log(f"  {n:12s} holdout {d:.4f}   annot {g_auc:.4f}")
        pick = max(results, key=lambda k: results[k][0])
        log(f"best on the holdout: {pick} ({results[pick][0]:.4f})")

        for name, pred in test_preds.items():
            sub = write_submission(pred, st_te, test_df, f"submission_{name}.csv")
            log(f"  submission_{name}.csv {sub.shape}; "
                f"nulls {int(sub[TARGETS].isna().sum().sum())}")

        ens = np.mean([pd.DataFrame(p).rank(pct=True).values for p in test_preds.values()],
                      axis=0)
        write_submission(ens, st_te, test_df, "submission_rankmean.csv")
        log(f"  submission_rankmean.csv (rank mean of {len(test_preds)})")

        sub = write_submission(test_preds[pick], st_te, test_df, "submission.csv")
        log(f"submission.csv = {pick}; {sub.shape}; "
            f"nulls {int(sub[TARGETS].isna().sum().sum())}")
        print(sub.head().to_string())
'''

# ---- deps missing from the TPU image (installed as the notebook's first cell) #
PIP_CELL = '''# --- Install deps missing from the TPU image ---------------------------------- #
# The GPU image ships pydicom but the torch-xla (TPU) image does not - the import cell
# raises ModuleNotFoundError: No module named 'pydicom' without this. scikit-learn
# supplies roc_auc_score (macro_auc) and is installed defensively; both are
# pure-python/numpy wheels with no torch dependency, and pip leaves the already-present
# numpy alone (sklearn only needs numpy>=1.22, which torch 2.8 already guarantees), so
# this is a fast no-op where they already exist and cannot destabilise the torch build.
#
# Pin BLAS/OpenMP threads to 1 BEFORE numpy is imported (this cell runs first). The
# decode wakes numpy's process-level OpenMP/MKL worker pool; those workers survive the
# decode and, at the later xmp.spawn(fork), their inherited locks corrupt the 8 forked
# children -> the intermittent native crash at the first host transfer (SIGTERM cascade,
# no Python trace). Forcing 1 BLAS thread keeps the fork from a process with live
# worker threads. Cost: the decode is ~1.5x slower (8-12 min instead of 6-8); the TPU
# training is unaffected (device ops, no CPU BLAS).
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import sys, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "pydicom", "scikit-learn"], check=True)
print("pydicom", "ok")
'''

# ---- save_member + manifest builder (same as GPU, cpu'd state_dict) --------- #
SAVE_CELL = '''# --- Save trained members as a reusable weights package --------------------- #
# A model is only useful if a later inference-only notebook can load it and trust that
# it computes what it computed here. The package written below is exactly what
# `find_weights` / `infer_from_package` read back. The member id is a hash of decode
# config + fold + seed, so re-running the same training overwrites the same file, while
# a different SEED (or fold) adds a distinct member. The state_dict is moved to CPU
# before torch.save - XLA tensors are lazy and torch.save must not see them.

WEIGHTS_OUT = Path("weights")
_PIXEL_KEYS = ("band", "crop_mm", "group", "img", "rules", "slices", "slots")


def _member_config(cfg):
    """The config a member records; matches what `adopt_config_globals` / `build_model`
    read back at inference. `img` must equal the cache resolution the member was fitted
    on, which is 336 here (RUNS r336 and CACHE_IMG 336 agree).
    """
    return {
        "img": int(cfg["img"]),
        "slices": int(CACHE_SLICES),
        "group": int(GROUP),
        "crop_mm": float(CROP_MM),
        "band": [float(x) for x in SLICE_BAND],
        "slots": [s[0] for s in SLOTS],
        "rules": dict(RULES),
        "variant": "large",
        "backbone": "facebook/dinov2-large",
        "unfreeze_last": int(UNFREEZE_LAST),
        "pool": "cls_mean",
        "prior": False,
    }


def save_member(model, dev, cfg, fold=0, holdout=None, annot=None,
                epochs_done=None, seed=None, source_run="tpu-train-v1"):
    """Write one trained member and append it to the package manifest."""
    WEIGHTS_OUT.mkdir(exist_ok=True)
    config = _member_config(cfg)
    pixel_group = json.dumps({k: config[k] for k in _PIXEL_KEYS}, sort_keys=True)
    ident = hashlib.md5(f"{pixel_group}|{fold}|{seed}".encode()).hexdigest()[:10]
    fname = f"m_{ident}.pt"

    model.eval()
    fp = fingerprint(model, dev, int(cfg["img"]))  # (2, N_TARGETS) logits, fp32
    xm.mark_step()   # settle the graph before the transfer (see predict)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"model": state, "fingerprint": fp}, WEIGHTS_OUT / fname)

    member = {
        "id": ident,
        "file": fname,
        "fold": int(fold),
        "tag": "",
        "config": config,
        "seed": int(seed) if seed is not None else SEED,
        "epochs_done": int(epochs_done) if epochs_done is not None else EPOCHS,
        "holdout": float(holdout) if holdout is not None else None,
        "annot": float(annot) if annot is not None and not np.isnan(annot) else None,
        "fingerprint": np.asarray(fp).tolist(),
        "pixel_group": pixel_group,
        "source_run": source_run,
    }

    man_path = WEIGHTS_OUT / "manifest.json"
    man = {"format": 2, "members": [], "history": [], "targets": TARGETS}
    if man_path.is_file():
        try:
            man = json.loads(man_path.read_text())
        except ValueError:
            man = {"format": 2, "members": [], "history": [], "targets": TARGETS}
    man.setdefault("targets", TARGETS)
    man.setdefault("history", [])
    if all(m["id"] != ident for m in man["members"]):
        man["members"].append(member)
    man_path.write_text(json.dumps(man, indent=2))
    log(f"  saved member {ident} -> {fname} (holdout {holdout:.4f})")
    return member
'''

# ---- one forward per device->host transfer (the v8-v16 predict crash) -------- #
# `acc = z if acc is None else acc + z` keeps N_GROUP DINOv2-large forwards alive in one
# lazy graph; the single `.cpu()` that follows executes all of them at once and the v3
# chip native-aborts. These replacements move the accumulation to the host and force one
# execution per group, which is the same shape the training step already proves fits.
#
# The sigmoid moves to the host with it and is still applied ONCE, to the summed logits:
# sigmoid per group then summed would average N_GROUP probabilities instead of one mean
# logit, which is a different number. `acc` is fp32 numpy either way, so nothing is lost.
PREDICT_OLD = '''        acc = None
        for g in range(N_GROUP):
            # Gathered a group at a time rather than whole and then sliced. The two are
            # the same pixels, but taking the whole of a study out of the cache allocates
            # every slice it holds - most of which this pass will not look at until a
            # later iteration, by which time they have been fetched again. Measured over
            # a cache of twelve slices, the difference between the two is the difference
            # between the step being bound by memory and being bound by the encoder.
            rows = torch.from_numpy(np.ascontiguousarray(
                cache[sel, :, g * GROUP:(g + 1) * GROUP])).to(dev)
            with torch.autocast("xla", enabled=False):
                z = model(rows, m, img_size).float()
            acc = z if acc is None else acc + z
        out.append(torch.sigmoid(acc / N_GROUP).cpu().numpy())'''

PREDICT_NEW = '''        acc = None
        for g in range(N_GROUP):
            # Gathered a group at a time rather than whole and then sliced. The two are
            # the same pixels, but taking the whole of a study out of the cache allocates
            # every slice it holds - most of which this pass will not look at until a
            # later iteration, by which time they have been fetched again. Measured over
            # a cache of twelve slices, the difference between the two is the difference
            # between the step being bound by memory and being bound by the encoder.
            rows = torch.from_numpy(np.ascontiguousarray(
                cache[sel, :, g * GROUP:(g + 1) * GROUP])).to(dev)
            with torch.autocast("xla", enabled=False):
                z = model(rows, m, img_size).float()
            # Execute THIS group's forward and take it to the host before building the
            # next one. Letting the logits pile up on the device keeps a graph of
            # N_GROUP DINOv2-large forwards alive, and the single .cpu() below then
            # materialises all of them at once - more HBM than a v3 chip has, which
            # native-aborts the worker. That is the crash every run from v8 to v16 died
            # on, here and nowhere else. One forward per execution is exactly the shape
            # the training step already proves fits, and the sum moves to numpy, which is
            # exact for a sum of fp32 logits. The sigmoid is applied once, at the end:
            # averaging four probabilities is not averaging four logits.
            xm.mark_step()
            _phb(f"b{b}_g{g}_xfer")
            zg = z.cpu().numpy()
            acc = zg if acc is None else acc + zg
        out.append((1.0 / (1.0 + np.exp(-(acc / N_GROUP)))).astype(np.float32))'''

PREDICT_MEMBER_OLD = '''        acc = None
        for st in starts:
            rows = torch.from_numpy(
                np.ascontiguousarray(cache[sel, :, st:st + group])).to(dev)
            with torch.autocast("xla", enabled=False):
                z = model(rows, m, img_size).float()
            v = z if pool == "logit" else torch.sigmoid(z)
            acc = v if acc is None else acc + v
        v = acc / len(starts)
        out.append((torch.sigmoid(v) if pool == "logit" else v).cpu().numpy())'''

PREDICT_MEMBER_NEW = '''        acc = None
        for st in starts:
            rows = torch.from_numpy(
                np.ascontiguousarray(cache[sel, :, st:st + group])).to(dev)
            with torch.autocast("xla", enabled=False):
                z = model(rows, m, img_size).float()
            # Same hazard as predict, and worse here: the TTA window count reaches ten
            # for a twelve-slice cache, so the device-side accumulation would hold ten
            # forwards. One execution and one transfer per window; the pooling and the
            # final sigmoid both move to numpy. `pool` is honoured per window exactly as
            # before, so the number this returns is unchanged.
            xm.mark_step()
            _phb(f"b{b}_w{st}_xfer")
            zg = z.cpu().numpy()
            v = zg if pool == "logit" else (1.0 / (1.0 + np.exp(-zg))).astype(np.float32)
            acc = v if acc is None else acc + v
        v = acc / len(starts)
        out.append((1.0 / (1.0 + np.exp(-v)) if pool == "logit" else v).astype(np.float32))'''

FINGERPRINT_OLD = '''        out = model(imgs, mask, img_size).float().cpu().numpy()'''

FINGERPRINT_NEW = '''        z = model(imgs, mask, img_size).float()
        # Execute the forward before reading it back: a .cpu() straight off the lazy
        # tensor is the transfer shape that aborts (see predict), and a fingerprint that
        # crashes is indistinguishable from a fingerprint that mismatches.
        xm.mark_step()
        out = z.cpu().numpy()'''

# --------------------------------------------------------------------------- #
applied = {"runs": False, "train": False, "tpu": False, "save": False,
           "pip": False, "autocast": 0, "config": 0, "predict": 0, "augment": 0}

for c in cells:
    if c["cell_type"] != "code":
        continue
    t = cell_text(c)

    if "RUNS = [" in t and RUNS_OLD in t and not applied["runs"]:
        t = t.replace(RUNS_OLD, RUNS_NEW)
        applied["runs"] = True
    for old, new, tag in CONFIG_EDITS:
        if old in t and t.count(old) == 1:
            t = t.replace(old, new)
            applied["config"] += 1

    # main() tail: grouping + hand-off + spawn replaces the old fine-tune + summary +
    # submissions (everything from "Grouped on report text" to the end of main).
    if "def main():" in t and "    # Grouped on report text:" in t:
        start = t.index("    # Grouped on report text:")
        end_marker = "print(sub.head().to_string())"
        end = t.index(end_marker) + len(end_marker)
        t = t[:start] + NEW_MAIN_TAIL + t[end:]
        applied["train"] = True

    # remaining cuda autocast (predict + predict_member) -> xla fp32 (no autocast)
    old_ac = 'torch.autocast("cuda", enabled=dev.type == "cuda")'
    new_ac = 'torch.autocast("xla", enabled=False)'
    applied["autocast"] += t.count(old_ac)
    t = t.replace(old_ac, new_ac)

    # One forward per device->host transfer (see the module docstring). Every run v8-v16
    # native-aborted at the single .cpu() after all N_GROUP logits had piled up on the
    # device. The markers stay in, now one per group, so a further crash still reports
    # which group and which batch it died on instead of needing another queue cycle.
    if PREDICT_OLD in t:
        assert t.count(PREDICT_OLD) == 1, "predict block matched more than once"
        t = t.replace(PREDICT_OLD, PREDICT_NEW)
        applied["predict"] += 1
    if PREDICT_MEMBER_OLD in t:
        assert t.count(PREDICT_MEMBER_OLD) == 1, "predict_member block matched more than once"
        t = t.replace(PREDICT_MEMBER_OLD, PREDICT_MEMBER_NEW)
        applied["predict"] += 1
    if FINGERPRINT_OLD in t:
        assert t.count(FINGERPRINT_OLD) == 1, "fingerprint transfer matched more than once"
        t = t.replace(FINGERPRINT_OLD, FINGERPRINT_NEW)
        applied["predict"] += 1

    # One line at predict entry recording what it was asked to score, so the append-only
    # heartbeat can be read against the size of the job rather than guessed at.
    if '    two coincide.\n    """\n    model.eval()\n    out = []' in t:
        assert t.count('    two coincide.\n    """\n    model.eval()\n    out = []') == 1
        t = t.replace(
            '    two coincide.\n    """\n    model.eval()\n    out = []',
            '    two coincide.\n    """\n    model.eval()\n'
            '    _phb(f"predict_start n={len(idx)} batch={EVAL_BATCH} ngroup={N_GROUP}")\n'
            '    out = []')
        applied["predict"] += 1

    # Device-side RNG out of the augmentation. XLA's device RNG carries its state forward
    # from draw to draw, and every draw in `augment` happens inside the one segment v17
    # measured as leaking (and leaking faster each step). These are 36 numbers per step:
    # drawing them on the CPU and transferring them costs nothing and leaves the device
    # graph with no RNG state to chain through.
    n_rand = t.count("torch.rand(n, device=dev)")
    assert n_rand in (0, 4), f"augment: expected 4 device draws, found {n_rand}"
    if n_rand:
        t = t.replace("torch.rand(n, device=dev)", "torch.rand(n).to(dev)")
        t = t.replace("torch.rand(n, 1, 1, 1, device=dev)", "torch.rand(n, 1, 1, 1).to(dev)")
        applied["augment"] = n_rand

    set_cell_text(c, t)

# Install-missing-deps cell goes first so every later import finds pydicom.
cells.insert(0, {"cell_type": "code", "execution_count": None,
                 "metadata": {}, "outputs": [], "source": PIP_CELL})
applied["pip"] = True

# Insert the save_member + TPU cells immediately before the "try: main()" cell.
insert_at = None
for i, c in enumerate(cells):
    if c["cell_type"] == "code":
        t = cell_text(c)
        if t.strip().startswith("try:") and "main()" in t:
            insert_at = i
            break

assert insert_at is not None, "could not find the try: main() cell"
for src in (SAVE_CELL, TPU_CELL):
    cells.insert(insert_at, {"cell_type": "code", "execution_count": None,
                             "metadata": {}, "outputs": [], "source": src})
applied["save"] = True
applied["tpu"] = True

# ---- verify ---------------------------------------------------------------- #
out = json.dumps(nb, ensure_ascii=False, indent=1)
flat = "\n".join(cell_text(c) for c in cells if c["cell_type"] == "code")
assert applied["runs"], "RUNS not replaced"
assert applied["train"], "main() tail not replaced"
assert applied["tpu"] and applied["save"], "cells not inserted"
assert applied["pip"], "pip cell not inserted"
assert '"pydicom"' in flat and "scikit-learn" in flat, "pip cell missing deps"
assert applied["config"] == 8, f"config edits: expected 8, applied {applied['config']}"
assert applied["autocast"] == 2, f"autocast edits: expected 2, applied {applied['autocast']}"
assert applied["predict"] == 4, f"transfer edits: expected 4, applied {applied['predict']}"
assert applied["augment"] == 4, f"augment edits: expected 4, applied {applied['augment']}"

assert '"name": "r224"' not in flat, "r224 still present"
assert '"name": "r336"' in flat, "r336 missing"
assert "CACHE_BUDGET_MAX_GB = 60.0" in flat, "cache budget not raised"
assert "xmp.spawn(_train_fn, start_method=\"fork\")" in flat, "spawn missing"
assert "def _train_fn(index):" in flat, "_train_fn missing"
assert "def save_member(" in flat, "save_member missing"
assert "torch.amp.GradScaler" not in flat, "GradScaler still present"
assert 'torch.autocast("cuda"' not in flat, "cuda autocast still present"
assert 'torch.autocast("xla"' in flat, "xla autocast missing"
assert "xm.optimizer_step(opt)" in flat, "optimizer_step missing"
assert "save_member(model, dev, cfg, fold=fold" in flat, "save_member call missing"
# the crash this version exists to fix: no device-side accumulation left to transfer
assert "torch.sigmoid(acc / N_GROUP).cpu().numpy()" not in flat, "predict still accumulates on device"
assert "(torch.sigmoid(v) if pool == \"logit\" else v).cpu().numpy()" not in flat, \
    "predict_member still accumulates on device"
assert flat.count("xm.mark_step()") >= 6, "transfer guards missing"
assert "1.0 / (1.0 + np.exp(-(acc / N_GROUP)))" in flat, "host-side sigmoid missing"
assert "_smoked" in flat, "predict smoke test missing"
assert "predict_hb_{os.getpid()}.txt" in flat, "predict heartbeat still shared/truncated"
assert "repredict_pre" in flat, "post-training smoke predict missing"
assert "_devmem(" in flat, "device-memory probe missing"
assert "devmem_before" in flat and "devmem_after_free" in flat, \
    "the predict handoff must report device memory on BOTH sides of the release"
assert "model.zero_grad(set_to_none=True)" in flat, \
    "training-state release before the post-training predict is missing - that is the " \
    "v20 hypothesis, and without it the experiment tests nothing"
assert "AUC_EVAL_MAX" in flat and "predict(model, Ctr, Mtr, auc_idx" in flat, \
    "holdout eval not subsampled"
assert "torch.rand(n, device=dev)" not in flat, "device RNG still in augment"
assert "np.ascontiguousarray(\n                        Ctr[sel, :, g * GROUP:(g + 1) * GROUP])" in flat, \
    "training group still sliced on the device"
assert "def _eval_now(" in flat and "_eval_now(ep, EPOCHS, EVAL_EVERY)" in flat, \
    "holdout eval not gated by EVAL_EVERY"
assert "if _eval_now(ep, EPOCHS, EVAL_EVERY):" in flat, "eval gate not wrapping the predict"
assert "_repredicted" in flat and "if not _repredicted:" in flat, \
    "repredict diagnostic not limited to one run"
assert ARGS.folds >= 2, "a single fold leaves the training set empty"

# compile-check every code cell so a bad escape in an injected cell (e.g. a \\n inside a
# triple-quoted source string turning into a real newline) fails HERE, not as a SyntaxError
# on the TPU after a ~1.5h queue.
for _i, _c in enumerate(cells):
    if _c["cell_type"] != "code":
        continue
    _t = cell_text(_c)
    try:
        compile(_t, f"<cell {_i}>", "exec")
    except SyntaxError as _e:
        raise SystemExit(
            f"generated notebook has SyntaxError in cell {_i} line {_e.lineno}: {_e.msg}\n"
            f"  {_e.text.strip() if _e.text else ''}")

DST.write_text(out, encoding="utf-8")
print(f"wrote {DST} ({len(cells)} cells)")
print("ok: r336 only, spawn-first TPU training, save_member per fold")
