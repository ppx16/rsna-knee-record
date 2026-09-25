# -*- coding: utf-8 -*-
"""Build the 64-slot density pilot as a TPU kernel.

WHY TPU, AND WHY THIS IS ONE RUN INSTEAD OF TWO.

The pilot asks one question: with everything else held fixed, does the 64-slot / max-span corpus
beat the 44-slot published one?  The trap in that sentence is "everything else": the original plan
pitted a 64-slot run against `cn0` (CoAtNet @384 on a rented 24 GB card), which cannot be reproduced
on Kaggle's 16 GB GPUs at the same config -- so the 44-slot arm would have had to be RE-RUN at a
different config, and the comparison would have carried a config change as well as a corpus change.

The TPU line has no such problem.  `nx3360` is convnext_base @336, bs 8, k 12, k_eval 24, seed 42,
12 epochs, fold 0, on the published 44-slot corpus.  So the 44-slot arm ALREADY EXISTS at exactly
the config the new arm will use:

    fixed: arch, res, bs, k, k_eval, seed, epochs, fold, AND the backend (both on TPU)
    varies: the corpus (44 slots @ span 0.152-0.847  vs  64 slots @ span 0.02-0.98)

And TPU quota is billed separately from GPU quota, so this does not compete with the Saturday
submission arms.

⚠️ THE HONEST CAVEAT: this measures the density effect on `convnext_base`, which FACTS D17 records as
a SUBSTITUTE for the frontier's CoAtNet, not an equivalent -- CoAtNet cannot train under torch_xla
(it dies on the additive relative-position bias).  The frontier's dense-corpus weights are CoAtNet.
If the density effect is architecture-dependent, this proxy can mislead.  It is still the cheapest
clean look, and it costs no GPU quota.

WHY THE BASE FILE IS THE *TPU* SCRIPT, NOT THE GENERIC TRAINER: kernel-tpu-nx336f1 is a TPU script
that actually RAN TO COMPLETION, so its X1-X7 XLA patches are known-good.  Re-deriving them from the
generic trainer just to add a corpus loader would re-open a solved problem.  Only `_open_corpus`,
`_open_ids` and the injected argv differ here.

    python scripts/build_dense_pilot_tpu.py                       # build
    python scripts/build_dense_pilot_tpu.py --push                # build + push
"""
import argparse
import ast
import json
import pathlib
import shutil
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from build_dense_pilot import NPART_SRC, patch  # noqa: E402  (one source of truth for the loader)

PROVEN_TPU = ROOT / "kernel-tpu-nx336f1" / "knee-mri-training-the-twelve-finding-model.py"
OUT = ROOT / "kernel-d64pilot-tpu"
# the exact injected argv of the proven fold-1 run; we change only tag + fold
ARGV_OLD = "'--tag', 'nx3361', '--folds', '5', '--fold', '1'"
ARGV_NEW = "'--tag', 'd64nx3360', '--folds', '5', '--fold', '0'"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--total", type=int, default=4407)
    ap.add_argument("--tag", default="d64nx3360")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="fast wiring check (1 epoch, few studies) -- use the leftover TPU quota to "
                         "prove the N-part loader works under XLA with the real mounted shards")
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.tag = "d64smoke"

    base = PROVEN_TPU.read_text(encoding="utf-8")
    for probe in ("def _open_corpus", "def _open_ids", "class _TwoPartVols"):
        if probe not in base:
            raise SystemExit("base TPU script is missing %r -- wrong file?" % probe)
    if base.count(ARGV_OLD) != 1:
        raise SystemExit("argv anchor appears %d times, expected 1: %s"
                         % (base.count(ARGV_OLD), ARGV_OLD))

    # 1) swap the corpus loader for the N-part one (the only functional change)
    code = patch(base, total=a.total, require_shards=True)

    # 2) retarget the injected argv at fold 0 / the new tag (and --smoke when building the probe)
    new_argv = ("'--tag', %r, '--folds', '5', '--fold', %r"
                % (a.tag, str(a.fold))) + (", '--smoke'" if a.smoke else "")
    code = code.replace(ARGV_OLD, new_argv)
    if a.tag not in code:
        raise SystemExit("argv retarget failed")

    # 3) keep the XLA patches: they must still be present after the corpus patch
    for probe in ("xla", "mark_step", "X6", "bf16"):
        pass
    n_xla = sum(code.count(x) for x in ("xm.mark_step", "xla_device", "_amp()"))
    if n_xla < 3:
        raise SystemExit("XLA patches look lost after the corpus patch (n=%d)" % n_xla)
    if "SHARD_REQUIRED = True" not in code:
        raise SystemExit("SHARD_REQUIRED did not get set")
    ast.parse(code)

    OUT.mkdir(exist_ok=True)
    (OUT / "knee-mri-training-the-twelve-finding-model.py").write_text(code, encoding="utf-8")

    meta = {
        "id": "ppxl16/rsna-knee-d64pilot-tpu",
        "title": "RSNA knee d64pilot tpu",
        "code_file": "knee-mri-training-the-twelve-finding-model.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": True,
        "enable_internet": True,
        # The 44-slot corpus is deliberately NOT mounted: the shard branch wins anyway, and its
        # 22 GB would only cost input IO. SHARD_REQUIRED makes a missing shard a loud failure.
        "dataset_sources": [
            "flight0234/rsna-knee-hybrid-report-labels",
            "ppxl16/rsna-knee-cv-folds",
        ],
        "kernel_sources": ["ppxl16/rsna-knee-corpus64-s%d" % i for i in range(a.shards)],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "machine_shape": "Tpu1VmV38",
    }
    (OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (OUT / "COMPARE.md").write_text(
        "# The paired comparison this run completes\n\n"
        "| | 44-slot arm | 64-slot arm |\n|---|---|---|\n"
        "| tag | `nx3360` (EXISTS) | `%s` (this kernel) |\n"
        "| corpus | published 44 slots, span 0.152-0.847 | built 64 slots, span 0.02-0.98 |\n"
        "| arch / res / bs / k / k_eval / seed / epochs | convnext_base.fb_in22k_ft_in1k / 336 / 8 /"
        " 12 / 24 / 42 / 12 | IDENTICAL |\n"
        "| backend | TPU | TPU |\n| fold | 0 | 0 |\n\n"
        "Everything is fixed except the corpus, and the backend matches, so the delta-OOF is\n"
        "attributable to the corpus. Compare with:\n\n"
        "    python scripts/cv_score.py --dir downloads/tpu --pair nx3360 %s\n\n"
        "Pre-registered reading (HANDOFF 150.2): delta > 0.004 soft = real; <= 0.004 = abandon the\n"
        "dense-corpus line.\n" % (a.tag, a.tag), encoding="utf-8")

    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_names.py"),
                        str(OUT / "knee-mri-training-the-twelve-finding-model.py")],
                       capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())
    if r.returncode != 0:
        raise SystemExit("check_names failed")
    print("built %s  tag=%s fold=%d  (%d shard sources, TPU)"
          % (OUT, a.tag, a.fold, a.shards))
    print("  XLA markers still present: %d" % n_xla)

    if a.push:
        subprocess.run(["kaggle", "kernels", "push", "-p", str(OUT)], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
