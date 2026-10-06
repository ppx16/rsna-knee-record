"""Register the re-pushed 3D TPU smoke kernel in the preflight target table.

WHY A FILE AND NOT A ONE-LINER
    The first attempt passed this as a PowerShell here-string and PowerShell stripped the quotes
    off the JSON keys, so Python saw `ref: ppxl16/...` and raised SyntaxError.  The project has
    already paid for PowerShell quoting twice (the BOM trap), so the payload goes in a file.
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
import json
import pathlib

P = pathlib.Path(__file__).resolve().parent.parent / 'state' / 'cv_targets.json'

ENTRY = {
    "ref": "ppxl16/rsna-knee-tpu-3dsmoke",
    "dir": "kernel-tpu-3dsmoke",
    "note": (
        "3D Conv3d TPU smoke, RE-PUSH. v1 got a real TPU (device xla:0, torch_xla 2.8.0) and died at "
        "~75 s on a format-string bug: 5 placeholders but 6 args, because the literal '1' in "
        "'input (1,%d,%d,%d)' was also passed as an argument. Fixed. Uses TPU quota, not the scarce GPU."
    ),
    "priority": -1,
    "attempts": 1,
}


def main() -> int:
    targets = json.loads(P.read_text(encoding='utf-8'))
    if any(t.get('dir') == ENTRY['dir'] for t in targets):
        print('already present; leaving the file alone')
    else:
        targets.insert(0, ENTRY)
        P.write_text(json.dumps(targets, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        print('inserted at the top')
    # assert the round-trip that matters: it must still parse as JSON
    reloaded = json.loads(P.read_text(encoding='utf-8'))
    assert any(t.get('dir') == ENTRY['dir'] for t in reloaded), 'entry vanished on reload'
    print('entries: %d  (round-trip OK)' % len(reloaded))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
