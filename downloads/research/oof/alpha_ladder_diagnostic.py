"""Map the RadImageNet blend weight past the retired 0.25 grid ceiling.

E9/E9b bounded the search at ``alpha=0.25`` and deployed a fixed ``0.20`` inherited from
the public mechanism. Four of five outer folds selected the boundary, so the optimum was
never observed. This script re-runs the same nested, dual-source, grouped-fold protocol on
an extended ladder and emits the E10 deployment contract.

The public OOF path stays an explicit input and is never copied into ``solution/``. Only
aggregate per-target scores, hashes, and the resulting decision are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from evaluation.metric import roc_auc

CFG = json.loads((ROOT / "evaluation/task_config.json").read_text(encoding="utf-8"))
TARGETS = CFG["targets"]
UID = CFG["id"]
TRAIN_PATH = ROOT / "data/train.csv"
BASE_OOF_PATH = ROOT / "baseline/public_0899/weights/oof.npz"
REMOTE_OOF_PATH = HERE / "artifacts/e9-t4-v15/v52_oof.csv"
DIAGNOSTIC_PATH = HERE / "state/alpha_ladder_diagnostic.json"
CONTRACT_PATH = HERE / "artifacts/e10/e10_contract.json"

EXPECTED_PUBLIC_OOF_SHA256 = "43bc2cf07ce924e2d02e3240c31638130c3275882caaed77e4368d39dbf465ff"
EXPECTED_REMOTE_OOF_SHA256 = "f5b1f02029af98740565bec5b6a76d8471b4b6a6039d09ea410368230cb1f03f"
EXPECTED_BASE_GOLD_MACRO = 0.8400089312435326
# E9b's pre-committed dual-OOF target set. E10 reuses it and does not re-derive it.
E9B_SELECTED = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Contusion",
]
LADDER = [0.20, 0.35, 0.50, 0.60, 0.70]
BOOTSTRAP_DRAWS = 3000
BOOTSTRAP_SEED = 20260811


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank_columns(values: np.ndarray) -> np.ndarray:
    return pd.DataFrame(np.asarray(values, dtype=np.float64)).rank(
        method="average", pct=True
    ).to_numpy(np.float64)


def target_auc(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Column-wise ROC AUC. Verified against the protected scorer before use."""
    rows, columns = truth.shape
    ranks = pd.DataFrame(prediction).rank(method="average").to_numpy(np.float64)
    positives = truth.sum(axis=0).astype(np.float64)
    negatives = rows - positives
    result = np.full(columns, np.nan)
    usable = (positives > 0) & (negatives > 0)
    rank_sum = (ranks * truth).sum(axis=0)
    result[usable] = (
        rank_sum[usable] - positives[usable] * (positives[usable] + 1.0) / 2.0
    ) / (positives[usable] * negatives[usable])
    return result


def blend(base: np.ndarray, rad: np.ndarray, alpha_map: dict[str, float]) -> np.ndarray:
    out = base.copy()
    for target, alpha in alpha_map.items():
        if alpha > 0:
            index = TARGETS.index(target)
            out[:, index] = (1.0 - alpha) * base[:, index] + alpha * rad[:, index]
    return out


# Rank basis: the blend is a mean of two rank scales, so it is NOT invariant to the
# population the ranks are taken over. At test time the ranked population and the scored
# population are the same rows, so the diagnostic ranks within the 58 scored gold studies.
# This matches the validated E9b path and keeps E10 a one-variable change from its parent.
def load_rad(path: Path, train: pd.DataFrame, expected_sha: str) -> tuple[np.ndarray, np.ndarray]:
    observed = sha256(path)
    if observed != expected_sha:
        raise RuntimeError(f"RadImageNet OOF drift for {path.name}: {observed}")
    frame = pd.read_csv(path, dtype={UID: str})
    expected_columns = [UID, *TARGETS, "fold", "is_gold"]
    if frame.columns.tolist() != expected_columns:
        raise RuntimeError(f"unexpected OOF schema: {frame.columns.tolist()}")
    if len(frame) != len(train) or frame[UID].duplicated().any():
        raise RuntimeError("OOF row identity is incomplete or duplicated")
    aligned = train[[UID]].merge(frame, on=UID, how="left", validate="one_to_one")
    if aligned[expected_columns[1:]].isna().any().any():
        raise RuntimeError("OOF does not cover every official train study")
    values = aligned[TARGETS].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("OOF contains non-finite predictions")
    return values, aligned["fold"].to_numpy(np.int64)
def per_target_rule(
    truth: np.ndarray,
    base: np.ndarray,
    sources: dict[str, np.ndarray],
    rows: np.ndarray,
) -> dict[str, float]:
    """Largest ladder weight whose per-target gain is positive in *every* OOF source."""
    reference = target_auc(truth[rows], base[rows])
    chosen: dict[str, float] = {}
    for index, target in enumerate(TARGETS):
        best = 0.0
        for alpha in LADDER:
            gains = [
                target_auc(
                    truth[rows], (1.0 - alpha) * base[rows] + alpha * rad[rows]
                )[index]
                - reference[index]
                for rad in sources.values()
            ]
            if all(gain > 0 for gain in gains):
                best = float(alpha)
        chosen[target] = best
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-oof", type=Path, required=True)
    parser.add_argument("--write-contract", action="store_true")
    args = parser.parse_args()

    started = time.time()
    train = pd.read_csv(TRAIN_PATH, dtype={UID: str})
    bundle = np.load(BASE_OOF_PATH, allow_pickle=False)
    if bundle["targets"].astype(str).tolist() != TARGETS:
        raise RuntimeError("E2 OOF target order drift")
    if not np.array_equal(bundle["ids"].astype(str), train[UID].to_numpy()):
        raise RuntimeError("E2 OOF study order drift")
    gold = bundle["gold_mask"].astype(bool)
    if not np.array_equal(gold, train[TARGETS].notna().all(axis=1).to_numpy()):
        raise RuntimeError("E2 gold mask differs from official train.csv")
    base = rank_columns(bundle["pred"].astype(np.float64)[gold])
    truth = train.loc[gold, TARGETS].to_numpy(np.int8)
    if truth.shape != (58, len(TARGETS)):
        raise RuntimeError(f"expected 58 gold rows, observed {truth.shape}")

    sources: dict[str, np.ndarray] = {}
    folds = None
    for name, path, expected in (
        ("public", args.public_oof, EXPECTED_PUBLIC_OOF_SHA256),
        ("remote_v15", REMOTE_OOF_PATH, EXPECTED_REMOTE_OOF_SHA256),
    ):
        values, fold_id = load_rad(path, train, expected)
        sources[name] = rank_columns(values[gold])
        if folds is None:
            folds = fold_id[gold]
        elif not np.array_equal(folds, fold_id[gold]):
            raise RuntimeError("OOF sources disagree on the grouped fold assignment")
    if sorted(np.unique(folds).tolist()) != [0, 1, 2, 3, 4]:
        raise RuntimeError("gold rows do not cover all five grouped folds")

    reference = target_auc(truth, base)
    protected = np.array([roc_auc(truth[:, i], base[:, i]) for i in range(len(TARGETS))])
    if not np.allclose(reference, protected, atol=1e-12):
        raise RuntimeError("fast AUC disagrees with the protected scorer")
    base_macro = float(np.mean(reference))
    if abs(base_macro - EXPECTED_BASE_GOLD_MACRO) > 1e-12:
        raise RuntimeError(f"E2 gold diagnostic drift: {base_macro}")

    ladder: dict[str, dict[str, dict[str, float]]] = {}
    for name, rad in sources.items():
        entry = {}
        for alpha in LADDER:
            scores = target_auc(truth, (1.0 - alpha) * base + alpha * rad)
            entry[f"{alpha:.2f}"] = {
                target: float(scores[index] - reference[index])
                for index, target in enumerate(TARGETS)
            }
        ladder[name] = entry

    uniform = {}
    for alpha in LADDER:
        alpha_map = {t: (alpha if t in E9B_SELECTED else 0.0) for t in TARGETS}
        uniform[f"{alpha:.2f}"] = {
            name: float(np.nanmean(target_auc(truth, blend(base, rad, alpha_map))))
            for name, rad in sources.items()
        }
        uniform[f"{alpha:.2f}"]["all_dual_positive"] = all(
            ladder[name][f"{alpha:.2f}"][t] > 0 for t in E9B_SELECTED for name in sources
        )

    # Honest out-of-fold form of the very selection this experiment performs. E9 fixed the
    # weight rather than tune it, precisely to keep the 58 gold rows untouched. E10 does use
    # them, so it must also report what the same choice looks like when the rung is picked on
    # four folds and scored on the fifth. If those held-out choices sit above the deployed
    # weight, the deployment is conservative rather than fitted.
    nested_uniform = {}
    for name, rad in sources.items():
        chosen = []
        held_out = base.copy()
        for outer in range(5):
            fit = folds != outer
            scores = [
                float(np.nanmean(target_auc(
                    truth[fit],
                    blend(base, rad, {t: (a if t in E9B_SELECTED else 0.0) for t in TARGETS})[fit],
                )))
                for a in LADDER
            ]
            alpha = float(LADDER[int(np.argmax(scores))])
            chosen.append(alpha)
            alpha_map = {t: (alpha if t in E9B_SELECTED else 0.0) for t in TARGETS}
            held = folds == outer
            held_out[held] = blend(base, rad, alpha_map)[held]
        nested_uniform[name] = {
            "outer_fold_choices": chosen,
            "min_choice": min(chosen),
            "nested_macro": float(np.nanmean(target_auc(truth, held_out))),
        }

    every = np.ones(len(truth), dtype=bool)
    descriptive_map = per_target_rule(truth, base, sources, every)
    nested_scores = {}
    nested_choices = []
    for outer in range(5):
        nested_choices.append(per_target_rule(truth, base, sources, folds != outer))
    for name, rad in sources.items():
        nested = base.copy()
        for outer in range(5):
            held = folds == outer
            nested[held] = blend(base, rad, nested_choices[outer])[held]
        nested_scores[name] = float(np.nanmean(target_auc(truth, nested)))

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(truth), (BOOTSTRAP_DRAWS, len(truth)))
    bootstrap = {}
    for name, rad in sources.items():
        wins = {f"{alpha:.2f}": [] for alpha in LADDER if alpha != 0.20}
        for draw in draws:
            y = truth[draw]
            if np.any(y.sum(axis=0) == 0) or np.any(y.sum(axis=0) == len(draw)):
                continue
            b, r = base[draw], rad[draw]
            low = np.nanmean(
                target_auc(y, blend(b, r, {t: (0.20 if t in E9B_SELECTED else 0.0) for t in TARGETS}))
            )
            for alpha in LADDER:
                if alpha == 0.20:
                    continue
                alpha_map = {t: (alpha if t in E9B_SELECTED else 0.0) for t in TARGETS}
                wins[f"{alpha:.2f}"].append(
                    float(np.nanmean(target_auc(y, blend(b, r, alpha_map))) - low)
                )
        bootstrap[name] = {
            key: {
                "draws": len(values),
                "mean_delta_vs_020": float(np.mean(values)),
                "p_beats_020": float(np.mean(np.asarray(values) > 0)),
            }
            for key, values in wins.items()
        }

    diagnostic = {
        "experiment": "E10",
        "question": "Does the retired alpha<=0.25 ceiling understate the RadImageNet vote?",
        "evidence_boundary": (
            "Every number here is a 58-study OOF diagnostic on official train labels, "
            "not a Kaggle score. Only a kernels-only submission scores the hidden test."
        ),
        "protocol": (
            "Grouped five-fold nested selection over an extended ladder, required to agree "
            "across two independently produced RadImageNet OOF runs."
        ),
        "targets": TARGETS,
        "ladder": [f"{alpha:.2f}" for alpha in LADDER],
        "e9b_selected_targets": E9B_SELECTED,
        "base_gold_macro_auc": base_macro,
        "per_target_base_auc": {t: float(reference[i]) for i, t in enumerate(TARGETS)},
        "per_target_ladder_delta": ladder,
        "uniform_alpha_on_e9b_targets": uniform,
        "uniform_alpha_nested_selection": nested_uniform,
        "selection_on_gold_rows": (
            "E10 selects the deployed weight using the 58 gold rows, which E9 deliberately "
            "avoided. uniform_alpha_nested_selection is the held-out form of that same "
            "procedure: if its per-fold choices all sit above the deployed weight, the "
            "deployment is conservative relative to what honest selection would pick."
        ),
        "per_target_rule_descriptive": descriptive_map,
        "per_target_rule_nested_macro": nested_scores,
        "per_target_rule_outer_fold_choices": [
            {t: choices[t] for t in TARGETS} for choices in nested_choices
        ],
        "bootstrap_vs_alpha_020": bootstrap,
        "public_oof_sha256": EXPECTED_PUBLIC_OOF_SHA256,
        "remote_oof_sha256": EXPECTED_REMOTE_OOF_SHA256,
        "seconds": round(time.time() - started, 2),
    }
    DIAGNOSTIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DIAGNOSTIC_PATH.write_text(
        json.dumps(diagnostic, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"base gold macro {base_macro:.6f}")
    for key, value in uniform.items():
        flag = "all-dual-positive" if value["all_dual_positive"] else "partial"
        print(
            f"  alpha={key} public={value['public']:.6f} "
            f"remote_v15={value['remote_v15']:.6f} [{flag}]"
        )
    print(f"  per-target nested {nested_scores}")
    for name, entry in nested_uniform.items():
        print(
            f"  nested uniform {name}: choices={entry['outer_fold_choices']} "
            f"min={entry['min_choice']:.2f} macro={entry['nested_macro']:.6f}"
        )
    print(f"wrote {DIAGNOSTIC_PATH.relative_to(ROOT)}")

    if args.write_contract:
        write_contract(diagnostic, descriptive_map, nested_scores)


def write_contract(
    diagnostic: dict,
    descriptive_map: dict[str, float],
    nested_scores: dict[str, float],
) -> None:
    """Emit the hash-pinned deployment contract consumed by the kernel arm."""
    uniform = diagnostic["uniform_alpha_on_e9b_targets"]
    base_auc = diagnostic["per_target_base_auc"]
    deltas = diagnostic["per_target_ladder_delta"]

    def descriptive_macro(alpha_map: dict[str, float]) -> dict[str, float]:
        """Macro AUC of this exact weight map, which the kernel re-derives before deploying.

        Distinct from ``diagnostic_macro``: for the per-target rule that headline number is
        the honest nested estimate, which by construction is not the macro of the map that
        actually ships. The kernel needs the shipped map's own value to verify against.
        """
        out = {}
        for name in deltas:
            out[name] = float(np.mean([
                base_auc[t] + (0.0 if a == 0.0 else deltas[name][f"{a:.2f}"][t])
                for t, a in alpha_map.items()
            ]))
        return out

    configurations = {}
    for alpha in LADDER:
        key = f"{alpha:.2f}"
        alpha_map = {t: (alpha if t in E9B_SELECTED else 0.0) for t in TARGETS}
        configurations[f"uniform_{int(round(100 * alpha)):03d}"] = {
            "alpha_map": alpha_map,
            "diagnostic_macro": {
                name: uniform[key][name] for name in ("public", "remote_v15")
            },
            "descriptive_macro": descriptive_macro(alpha_map),
            "all_dual_positive": bool(uniform[key]["all_dual_positive"]),
            "rationale": (
                f"E9b's pre-committed ten targets at a uniform {key} vote; "
                "the ten-target set is inherited, not re-derived."
            ),
        }
    configurations["per_target_nested"] = {
        "alpha_map": descriptive_map,
        "diagnostic_macro": nested_scores,
        "descriptive_macro": descriptive_macro(descriptive_map),
        "all_dual_positive": True,
        "rationale": (
            "Per target, the largest ladder weight positive in both independent OOF runs. "
            "Reported macro is the honest grouped-fold nested estimate, not the "
            "descriptive optimum."
        ),
    }
    contract = {
        "version": "e10-alpha-ladder-2",
        "supersedes": "e9b-dual-oof-intersection-1",
        "defect_repaired": (
            "E9's alpha grid stopped at 0.25 and four of five outer folds selected that "
            "boundary, so deployment at 0.20 was a censored search, not an optimum."
        ),
        "evidence_boundary": diagnostic["evidence_boundary"],
        "protocol": diagnostic["protocol"],
        "targets": TARGETS,
        "e9b_selected_targets": E9B_SELECTED,
        "ladder": diagnostic["ladder"],
        "base_gold_macro_auc": diagnostic["base_gold_macro_auc"],
        "per_target_base_auc": diagnostic["per_target_base_auc"],
        "per_target_ladder_delta": diagnostic["per_target_ladder_delta"],
        "uniform_alpha_nested_selection": diagnostic["uniform_alpha_nested_selection"],
        "selection_on_gold_rows": diagnostic["selection_on_gold_rows"],
        "per_target_rule_outer_fold_choices": diagnostic[
            "per_target_rule_outer_fold_choices"
        ],
        "bootstrap_vs_alpha_020": diagnostic["bootstrap_vs_alpha_020"],
        "configurations": configurations,
        "public_oof_sha256": diagnostic["public_oof_sha256"],
        "remote_oof_sha256": diagnostic["remote_oof_sha256"],
    }
    CONTRACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONTRACT_PATH.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {CONTRACT_PATH.relative_to(ROOT)} sha256={sha256(CONTRACT_PATH)}")


if __name__ == "__main__":
    main()

