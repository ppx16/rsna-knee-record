# -*- coding: utf-8 -*-
"""Report (or check) the SEED and backbone baked into a generated notebook.

The GPU generator writes its output to notebooks/rsna-knee-train-v1.ipynb. Pushing is a
separate step that copies that file into kernel-train/, and forgetting the copy is a
quiet, expensive mistake: `kaggle kernels push -p kernel-train` would re-run the PREVIOUS
configuration. That matters twice over -

  * the same SEED produces member ids that hash fold+seed, so a repeat run deduplicates
    away entirely and the quota is gone;
  * the same PIXEL config produces the same ids even at a different backbone, because
    `pixel_group` is built from (band, crop_mm, group, img, rules, slices, slots) and not
    from the variant - so a run whose ids collide with banked members is silently dropped
    by the packager's first-wins merge.

Hence both are checked before a push.

    python scripts/check_notebook_seed.py kernel-train/rsna-knee-train-v1.ipynb
    python scripts/check_notebook_seed.py kernel-train/rsna-knee-train-v1.ipynb 2027
    python scripts/check_notebook_seed.py kernel-train/rsna-knee-train-v1.ipynb 2027 --variant small

Exits 0 when everything asked for matches, 1 when the push directory is stale.

⚠️ Fixed 2026-09-11: this script used to report `variant=large` for a notebook that was
actually the biomedclip build, because it scraped `variant="large"` out of the *default
argument of the now-dead DINOv2 helper* instead of reading the constant the generator
actually writes. A verification tool that gives a false green light is worse than no
tool, so BACKBONE_VARIANT/FAMILY/REPO are now the authoritative source and the old
regexes survive only as a fallback for pre-biomedclip notebooks.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

NEEDLE = re.compile(r'os\.environ\.get\(\s*"SEED"\s*,\s*"(\d+)"\s*\)')
# Fallback for a notebook that was never patched (the upstream baseline form).
PLAIN = re.compile(r"^SEED\s*=\s*(\d+)\s*$", re.M)

# Authoritative: the generator writes these constants and the run reads them.
VARIANT_CONST = re.compile(r'^BACKBONE_VARIANT\s*=\s*"([^"]+)"\s*$', re.M)
FAMILY_CONST = re.compile(r'^BACKBONE_FAMILY\s*=\s*"([^"]+)"\s*$', re.M)
REPO_CONST = re.compile(r'^BACKBONE_REPO\s*=\s*"([^"]+)"\s*$', re.M)

# Legacy signal, only meaningful for notebooks predating the BACKBONE_* constants.
VARIANT_FIND = re.compile(r'def\s+find_dinov2\s*\(\s*variant\s*=\s*"(\w+)"\s*\)')
VARIANT_BUILD = re.compile(r'variant\s*=\s*"(\w+)"\s*,\s*pool\s*=')


def cell_source(c: dict) -> str:
    s = c.get("source", "")
    return "".join(s) if isinstance(s, list) else s


def scan(path: Path) -> dict:
    """Everything the notebook says about its identity, as written by the generator."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    seed = None
    out = {"variant": [], "family": [], "repo": [], "legacy_variant": []}
    for c in nb.get("cells", []):
        if c.get("cell_type") != "code":
            continue
        src = cell_source(c)
        if seed is None:
            m = NEEDLE.search(src) or PLAIN.search(src)
            if m:
                seed = m.group(1)
        for rx, key in ((VARIANT_CONST, "variant"), (FAMILY_CONST, "family"),
                        (REPO_CONST, "repo")):
            out[key].extend(rx.findall(src))
        for rx in (VARIANT_FIND, VARIANT_BUILD):
            out["legacy_variant"].extend(rx.findall(src))
    out["seed"] = seed
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("expected_seed", nargs="?", default=None,
                    help="omit to just report what the notebook carries")
    ap.add_argument("--variant", default=None,
                    help='expected backbone, e.g. "small", "large", "biomedclip-vitb16"')
    args = ap.parse_args()

    path = Path(args.notebook)
    if not path.is_file():
        print(f"no such notebook: {path}", file=sys.stderr)
        return 2
    try:
        info = scan(path)
    except (ValueError, KeyError) as exc:
        print(f"{path}: not a readable notebook ({exc})", file=sys.stderr)
        return 2

    seed = info["seed"]
    if seed is None:
        print(f"{path}: no SEED assignment found", file=sys.stderr)
        return 1

    const_variants = sorted(set(info["variant"]))
    if len(const_variants) > 1:
        print(f"{path}: BACKBONE_VARIANT is inconsistent across the notebook: "
              f"{const_variants} - the generator only half-applied its edits", file=sys.stderr)
        return 1

    if const_variants:
        variant = const_variants[0]
        family = ",".join(sorted(set(info["family"]))) or "?"
        repo = ",".join(sorted(set(info["repo"]))) or "?"
        source = "BACKBONE_VARIANT"
    else:
        # Pre-biomedclip notebook: the DINOv2 call sites are still live code.
        legacy = sorted(set(info["legacy_variant"]))
        if len(legacy) > 1:
            print(f"{path}: variant is inconsistent across the notebook: {legacy} - the "
                  f"generator only half-applied its edits", file=sys.stderr)
            return 1
        variant, family, repo = (legacy[0] if legacy else None), "dinov2", "?"
        source = "legacy DINOv2 call sites"

    if args.expected_seed is None and args.variant is None:
        print(f"seed={seed} variant={variant} family={family} repo={repo} (from {source})")
        return 0

    bad = []
    if args.expected_seed is not None and seed != args.expected_seed:
        bad.append(f"SEED={seed} but {args.expected_seed} was requested")
    if args.variant is not None and variant != args.variant:
        bad.append(f"variant={variant} but {args.variant} was requested")
    if bad:
        print(f"{path}: " + "; ".join(bad)
              + " - the push directory is stale; copy notebooks/ over it before pushing, "
                "or the run repeats the previous configuration", file=sys.stderr)
        return 1

    print(f"{path}: seed={seed} variant={variant} family={family} repo={repo} (as requested)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
