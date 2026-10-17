# -*- coding: utf-8 -*-
"""
Build kernel-v5ft64: msft2's recipe, warm-started from v5, on the 64-SLOT corpus.

WHY THIS COMBINATION HAS NEVER BEEN RUN
    msft2  = v5 (native 64 slots) warm-started on OUR 44-slot corpus  -> geometry MISMATCH
    v8ft   = v8 (native 44 slots) warm-started on OUR 44-slot corpus  -> geometry MATCHED
    v5ft64 = v5 (native 64 slots) warm-started on the 64-SLOT corpus  -> geometry MATCHED  <-- this
v5 is our strongest checkpoint (author gold_auc 0.9214), and msft2 already produced our best
arm (solo 0.8857 on the 870 instrument) DESPITE the mismatch. So the matched version is the
one combination the user's question exposes that nobody has tried.

WHY IT IS A MERGE AND NOT A 3-LINE DIFF (unlike kernel-v8ft)
    msft2's corpus loader reads two fixed files: all_vols.npy + extra_vols.npy (44 slots).
    The 64-slot corpus is SHARDED: rsna-knee-corpus64-s<i>/vols_s<i>.npy (+ ids_s*, masks_s*).
    kernel-d64pilot already resolves that sharded layout -- and already carries the guard for
    the trap this would otherwise walk into, verbatim in its own source:

        "Without it, a kernel whose kernel_sources did not mount would fall through to the
         published 44-slot corpus and TRAIN THE WRONG ARM while every log line still looked
         healthy -- producing a delta-OOF of ~0 and a false conclusion that the dense corpus
         does not help.  MEASURED 2026-09-24: the mount probe found 0 shards."

    (SHARD_EXPECTED_ROWS = 4407, SHARD_REQUIRED = True.)

    So: base on d64pilot (correct 64-slot corpus handling), and splice in the ONLY two things
    msft2 has that d64pilot lacks -- the fixed checkpoint loader (unwrap model/student/
    state_dict + raise on zero tensors) and the 8-tensor HEAD CARRY.  d64pilot's load_raptor is
    the OLD unfixed one (`st = ck["student"] if "student" in ck else ck`), so building from it
    unchanged would hit the silent-zero trap that FACTS A170 documents.

Blocks are extracted from kernel-msft2 BY CONTENT, never retyped, and every anchor must match
exactly once or the build FATALs without writing.
"""
import hashlib
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASE = ROOT / "kernels" / "kernel-d64pilot" / "knee-mri-training-the-twelve-finding-model.py"
DON = ROOT / "kernels" / "kernel-msft2" / "knee-mri-training-the-twelve-finding-model.py"
DST_DIR = ROOT / "kernels" / "kernel-v5ft64"
DST = DST_DIR / "knee-mri-training-the-twelve-finding-model.py"


def once(text, needle, what):
    n = text.count(needle)
    if n != 1:
        raise SystemExit(f"FATAL: {what}: anchor found {n} times, expected 1")
    return text.index(needle)


def block(don, start_needle, end_needle, what):
    """Text from the line containing start_needle up to (not including) end_needle's line."""
    i = once(don, start_needle, what + " start")
    j = once(don, end_needle, what + " end")
    i0 = don.rfind("\n", 0, i) + 1
    j0 = don.rfind("\n", 0, j) + 1
    if j0 <= i0:
        raise SystemExit(f"FATAL: {what}: end precedes start")
    return don[i0:j0]


def main():
    base = BASE.read_text(encoding="utf-8")
    don = DON.read_text(encoding="utf-8")

    # ---- extract from msft2 -------------------------------------------------------------
    new_loader = block(don, "def load_raptor(bb, ckpt_path):",
                       "class RaptorClassifier", "loader")
    new_callsite = "    src, _HEAD_STATE = load_raptor(bb, a.ckpt)\n"
    head_blk = block(don, "    model = RaptorClassifier(bb, F_dim=F_dim).to(dev)",
                     "    if a.freeze_blocks > 0:", "head block")
    if "HEAD loaded" not in head_blk or "_FOREIGN_HEAD" not in head_blk:
        raise SystemExit("FATAL: extracted head block is not the msft2 head-carry block")

    # ---- 1. replace the unfixed loader ---------------------------------------------------
    old_loader = block(base, "def load_raptor(bb, ckpt_path):",
                       "class RaptorClassifier", "base loader")
    if 'ck["student"] if "student" in ck' not in old_loader:
        raise SystemExit("FATAL: base loader is not the expected unfixed one -- re-inspect")
    base = base.replace(old_loader, new_loader, 1)

    # ---- 2. call site unpacks the tuple --------------------------------------------------
    old_call = "    src = load_raptor(bb, a.ckpt)\n"
    once(base, old_call, "base call site")
    base = base.replace(old_call, new_callsite, 1)

    # ---- 3. splice the head carry after model construction -------------------------------
    old_model = "    model = RaptorClassifier(bb, F_dim=F_dim).to(dev)\n"
    once(base, old_model, "base model construction")
    base = base.replace(old_model, head_blk, 1)

    # ---- 4. --allow_bad_warmstart must exist (the head block references it) --------------
    if "--allow_bad_warmstart" not in base:
        anchor = '    ap.add_argument("--grad_ckpt", action="store_true")'
        if base.count(anchor) != 1:
            raise SystemExit("FATAL: no single --grad_ckpt anchor to add the flag beside")
        base = base.replace(anchor, anchor + '\n    ap.add_argument("--allow_bad_warmstart", '
                                          'action="store_true",\n'
                                          '                    help="train the head FROM SCRATCH '
                                          'if the warm-start head is incompatible")', 1)

    # ---- 5. prelude: the msft2 recipe, --ckpt = v5, tag v5ft64 ---------------------------
    old_pre = re.search(r"if not any\(_a\.startswith\(\"--\"\) for _a in sys\.argv\[1:\]\):.*?"
                        r"\nelse:\n    print\('\[prelude\] explicit argv supplied.*?\n",
                        base, re.S)
    if not old_pre:
        raise SystemExit("FATAL: base prelude argv block not found")
    new_pre = '''# ---- fold file: kernel-d64pilot's prelude does NOT resolve this (kernel-msft2's does), and
# the spliced head block assumes folds/fold are set.  check_names.py caught this as an
# undefined global -- py_compile does not.
_fold_cands = glob.glob('/kaggle/input/**/fold_file.json', recursive=True)
if not _fold_cands:
    raise SystemExit('fold_file.json not found under /kaggle/input -- attach ppxl16/rsna-knee-cv-folds')
_FOLD_FILE = _fold_cands[0]
print('[prelude] fold_file -> %s' % _FOLD_FILE, flush=True)

_warm_c = glob.glob('/kaggle/input/**/raptor_ft_coatnet_v5_full_swa.pt', recursive=True)
if not _warm_c:
    raise SystemExit('warm-start checkpoint raptor_ft_coatnet_v5_full_swa.pt not found under '
                     '/kaggle/input -- is dreaddevelopment/raptor-knee-maxspan attached?')
_WARM = _warm_c[0]
print('[prelude] warm start -> %s' % _WARM, flush=True)
# The 64-slot corpus arrives as kernel_sources (ppxl16/rsna-knee-corpus64-s0..s7). The base
# trainer already resolves that sharded layout and REFUSES to fall through to the 44-slot
# corpus (SHARD_REQUIRED) -- that guard is the whole reason this is based on d64pilot.
sys.argv = ['train',
            '--labels', _parq,
            '--arch', 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k',
            '--res', '384', '--epochs', '4', '--bs', '2', '--k', '12', '--k_eval', '24',
            '--grad_ckpt', '--ckpt', _WARM,
            '--seed', '42', '--tag', 'v5ft64', '--folds', '5', '--fold', '0',
            '--fold_file', _FOLD_FILE, '--workers', '0']
print('[prelude] 64-SLOT corpus recipe; warm start is v5 (native geometry 64) -> MATCHED', flush=True)
'''
    base = base[:old_pre.start()] + new_pre + base[old_pre.end():]

    DST_DIR.mkdir(parents=True, exist_ok=True)
    DST.write_text(base, encoding="utf-8")
    compile(base, str(DST), "exec")

    meta = json.loads((ROOT / "kernels" / "kernel-d64pilot" / "kernel-metadata.json")
                      .read_text(encoding="utf-8"))
    meta["id"] = "ppxl16/rsna-knee-v5ft64"
    meta["title"] = "RSNA knee v5ft64"
    meta["enable_tpu"] = False
    meta["enable_gpu"] = True
    meta.pop("machine_shape", None)                       # A168 trap 3
    ds = [d for d in meta.get("dataset_sources", [])]
    for need in ("dreaddevelopment/raptor-knee-maxspan",
                 "dreaddevelopment/knee-raptor-corpus",
                 "dreaddevelopment/knee-raptor-corpus-ext"):
        if need not in ds:
            ds.append(need)
    meta["dataset_sources"] = ds
    # kernel_sources = the 8 corpus64 shards (already present in d64pilot's metadata)
    meta.setdefault("kernel_sources", [])
    (DST_DIR / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")
    json.loads((DST_DIR / "kernel-metadata.json").read_text(encoding="utf-8"))

    print(f"built {DST.relative_to(ROOT)}  bytes={len(base.encode())}  "
          f"sha256={hashlib.sha256(base.encode()).hexdigest()[:16]}")
    print("spliced from kernel-msft2 (by content, not retyped):")
    print("  - load_raptor: unwrap model/student/state_dict + raise on 0 tensors + head carry")
    print("  - call site returns (src, _HEAD_STATE)")
    print("  - head block: verifies names/shapes, refuses a foreign head unless overridden")
    print(f"  - --allow_bad_warmstart present: {'--allow_bad_warmstart' in base}")
    print("based on kernel-d64pilot for the 64-slot sharded corpus + its SHARD_REQUIRED guard")
    print(f"id={meta['id']} gpu={meta['enable_gpu']} tpu={meta['enable_tpu']} "
          f"machine_shape={meta.get('machine_shape')}")
    print(f"datasets={len(ds)} kernel_sources={len(meta['kernel_sources'])}")
    for k in meta["kernel_sources"]:
        print(f"  ks - {k}")


if __name__ == "__main__":
    main()
