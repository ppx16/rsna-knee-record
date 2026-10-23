# -*- coding: utf-8 -*-
"""Make the mirror switch CALL-TIME so one process can run both passes.

The first patch read RAPTOR_MIRROR at import into MIRROR_ENABLED, which locks a process to
one mode. The eval driver needs BOTH passes (plain and mirrored) in one kernel session, so the
lookup moves to the point of use: `os.environ.get(...)` inside StudyWindows.__getitem__.
Then a driver can set/clear the variable between two main() calls -- which already works,
because main() re-parses sys.argv each time (that is how kernel-eval-v5v9 ran four evals).

compile() before write, as always.
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
TARGETS = [ROOT / "kernels" / "kernel-raptor-train" / "knee-mri-training-the-twelve-finding-model.py",
           ROOT / "kernels" / "kernel-raptor-train-rule" / "knee-mri-training-the-twelve-finding-model.py"]

OLD = "            if MIRROR_ENABLED:"
NEW = ('            # call-time lookup: the eval driver sets/clears RAPTOR_MIRROR between two\n'
       '            # main() calls, which is what isolates the mirror in one kernel session\n'
       '            if _os.environ.get("RAPTOR_MIRROR"):')


def main():
    for p in TARGETS:
        src = p.read_text(encoding="utf-8")
        if 'if _os.environ.get("RAPTOR_MIRROR"):' in src:
            print(f"  {p.parent.name}: already call-time")
            continue
        n = src.count(OLD)
        if n != 1:
            raise SystemExit(f"FATAL {p.parent.name}: anchor found {n} times, expected 1")
        src = src.replace(OLD, NEW, 1)
        compile(src, str(p), "exec")
        p.write_text(src, encoding="utf-8")
        print(f"  {p.parent.name}: now call-time + compile OK")


if __name__ == "__main__":
    main()
