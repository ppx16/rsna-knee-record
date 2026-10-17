# -*- coding: utf-8 -*-
"""Inspect a downloaded metadata.pkl WITHOUT executing arbitrary pickle opcodes.

A Kaggle dataset pickle is third-party code. We only need to READ its structure, so this
uses a restricted unpickler that refuses anything outside a small allow-list of builtins
and numpy array reconstruction. That is enough for dict/list/tuple/str/int/float/np.ndarray
and stops a hostile __reduce__ from running.
"""
import io
import pickle
import pathlib
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ALLOWED = {
    ("builtins", "dict"), ("builtins", "list"), ("builtins", "tuple"),
    ("builtins", "set"), ("builtins", "frozenset"), ("builtins", "str"),
    ("builtins", "int"), ("builtins", "float"), ("builtins", "bool"),
    ("builtins", "bytes"), ("builtins", "NoneType"), ("builtins", "slice"),
    ("builtins", "complex"), ("builtins", "object"),
    ("collections", "OrderedDict"), ("collections", "defaultdict"),
    ("numpy", "ndarray"), ("numpy", "dtype"), ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"), ("numpy", "float64"), ("numpy", "float32"),
    ("numpy", "int64"), ("numpy", "int32"), ("numpy", "uint8"), ("numpy", "bool_"),
    ("numpy._core.multiarray", "_reconstruct"),
}


class Restricted(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"BLOCKED global: {module}.{name}")


def describe(o, depth=0, maxdepth=3):
    pad = "  " * depth
    if isinstance(o, dict):
        print(f"{pad}dict[{len(o)}]")
        for i, (k, v) in enumerate(o.items()):
            if i >= 12:
                print(f"{pad}  ... ({len(o) - 12} more keys)")
                break
            print(f"{pad}  {k!r}: ", end="")
            if isinstance(v, (dict, list, tuple)) and depth < maxdepth:
                print()
                describe(v, depth + 2, maxdepth)
            else:
                try:
                    import numpy as np
                    if isinstance(v, np.ndarray):
                        print(f"ndarray shape={v.shape} dtype={v.dtype}")
                    else:
                        print(repr(v)[:160])
                except Exception:
                    print(repr(v)[:160])
    elif isinstance(o, (list, tuple)):
        print(f"{pad}{type(o).__name__}[{len(o)}]")
        for i, v in enumerate(o[:6]):
            print(f"{pad}  [{i}] ", end="")
            describe(v, depth + 2, maxdepth)
        if len(o) > 6:
            print(f"{pad}  ... ({len(o) - 6} more)")
    else:
        print(f"{pad}{type(o).__name__}: {repr(o)[:160]}")


p = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "scratch/skcache/metadata.pkl")
with open(p, "rb") as f:
    try:
        obj = Restricted(f).load()
    except Exception as e:
        print(f"restricted load failed: {e}")
        raise SystemExit(1)
print(f"{p} -> {type(obj).__name__}")
describe(obj)
