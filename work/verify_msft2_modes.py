"""Verify the msft2 prelude flags per mode -- the field that decides gate-vs-train."""
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
NB = ROOT / "kernels" / "kernel-msft2" / "knee-mri-training-the-twelve-finding-model.py"


def prelude(text):
    i = text.find("sys.argv = [")
    j = text.find("]", i)
    return text[i:j + 1]


def build(mode):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_msft2_kernel.py"),
                        "--build", "--device", "tpu", "--mode", mode],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    ok = "PATTERN CHECK OK" in r.stdout
    t = NB.read_text(encoding="utf-8")
    p = prelude(t)
    print("mode=%-9s build_ok=%-5s | prelude has --zero_shot_only: %-5s | --limit: %-5s | gate bar 0.8963: %s"
          % (mode, ok, "--zero_shot_only" in p, "'--limit'" in p, "0.8963" in t))
    if not ok:
        print(r.stdout[-500:], r.stderr[-500:])


for m in ("gate", "finetune"):
    build(m)
print("\n(the file now holds the LAST mode built -- finetune -- which is what gets pushed)")
