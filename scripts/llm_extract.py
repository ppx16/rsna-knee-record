# -*- coding: utf-8 -*-
"""D2: LLM-based report -> 12 target extraction, vs the rule-based parser.

The rule parser (`src/report_parser.py`) reaches 0.862 macro-AUC against the 58 gold
labels. This script pilots a local LLM (Ollama qwen3:30b-a3b) on the same task to see
whether it can beat that ceiling — and, if it can, batch-runs it over the full 4407
reports to produce a replacement pseudo-label CSV.

Design notes:
  * The competition scores AUC, so we need a *ranking* per target, not a label. We ask
    the model for a probability in [0,1]; the 12 numbers become the score columns.
  * The reports are in ~9 languages; the instruction is English but the report is fed
    verbatim. Qwen3 is multilingual.
  * `format="json"` forces a parseable object, which we then validate + clamp.

Usage:
    python scripts/llm_extract.py --pilot        # 58 gold-label reports + AUC report
    python scripts/llm_extract.py --full         # all 4407 -> outputs/report_pseudo_labels_llm.csv
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import report_parser as rp  # noqa: E402

DATA = Path("D:/kaggle/rsna-knee/data/raw")
OUT = Path("D:/kaggle/rsna-knee/outputs")
OLLAMA = "http://localhost:11434/api/generate"
MODEL = "qwen3:30b-a3b"

SYSTEM = (
    "You are a radiologist reading knee MRI reports, which may be written in English, "
    "Spanish, French, German, Dutch, Turkish, Croatian, Serbian, Greek, Russian, or "
    "Bulgarian. For each report, judge the likelihood that each of 12 findings is present "
    "based ONLY on what the report states. A report that explicitly clears a finding "
    "(\"meniscus is normal\", \"no effusion\") means a LOW probability for it. A report "
    "that never mentions a finding means an uncertain probability near the middle, not "
    "zero. Answer with ONLY a flat JSON object; no markdown, no commentary."
)

FINDINGS = (
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
)


def build_prompt(report: str) -> str:
    keys = ", ".join(f'"{f}"' for f in FINDINGS)
    return (
        f"Return one probability from 0.0 (absent) to 1.0 (present) for each finding, "
        f"as a JSON object with exactly these keys: {keys}.\n\n"
        f"REPORT:\n{report}"
    )


def extract_one(report: str, timeout=300) -> dict:
    payload = {
        "model": MODEL,
        "system": SYSTEM,
        "prompt": build_prompt(report),
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0, "num_predict": 512},
    }
    r = requests.post(OLLAMA, json=payload, timeout=timeout)
    r.raise_for_status()
    d = r.json()
    # qwen3's thinking mode emits the JSON into `thinking` and leaves `response` empty
    # when format="json" is set; prefer the real answer field, then fall back.
    raw = (d.get("response") or d.get("thinking") or "").strip()
    # strip a leading ```json fence if the model added one despite format=json
    if raw.startswith("```"):
        raw = raw.split("```")[1].strip().removeprefix("json").strip()
    obj = json.loads(raw)
    return obj


def parse_row(obj: dict) -> dict:
    """Normalise an LLM answer to the 12 score columns, tolerating key-name variance."""
    out = {}
    for f in FINDINGS:
        v = obj.get(f)
        if v is None:
            # tolerate lowercased / underscored key spellings
            for k, val in obj.items():
                if k.lower().replace(" ", "_").replace("'", "") == \
                        f.lower().replace(" ", "_").replace("'", ""):
                    v = val
                    break
        if v is None:
            raise KeyError(f"missing {f!r} in {list(obj)}")
        out[f] = float(np.clip(float(v), 0.0, 1.0))
    return out


def report_auc(lab: pd.DataFrame, gold: pd.DataFrame) -> float:
    aucs = []
    for t in rp.TARGETS:
        y = gold[t].astype(int).values
        p = lab.loc[gold.index, t].values
        if y.min() == y.max():
            continue
        aucs.append(roc_auc_score(y, p))
    return float(np.nanmean(aucs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="run on the 58 gold-label reports")
    ap.add_argument("--full", action="store_true", help="run on all 4407 reports")
    ap.add_argument("--limit", type=int, default=0, help="run on first N reports (debug)")
    args = ap.parse_args()

    df = pd.read_csv(DATA / "train.csv")
    if args.pilot or args.limit:
        df = df.dropna(subset=rp.TARGETS)
    if args.limit:
        df = df.head(args.limit)

    rows = []
    t0 = time.time()
    for i, (_, row) in enumerate(df.iterrows(), 1):
        try:
            obj = extract_one(row["Report"] or "")
            rows.append({"StudyInstanceUID": row["StudyInstanceUID"],
                         **parse_row(obj)})
        except Exception as e:  # noqa: BLE001
            print(f"[{i}/{len(df)}] FAIL {row['StudyInstanceUID'][:16]}... : {e}")
            rows.append({"StudyInstanceUID": row["StudyInstanceUID"],
                         **{f: 0.5 for f in rp.TARGETS}})
        if i % 5 == 0:
            el = time.time() - t0
            print(f"[{i}/{len(df)}] {el/i:.1f}s/report, eta {el/i*(len(df)-i)/60:.1f}min")

    lab = pd.DataFrame(rows).set_index("StudyInstanceUID")
    if args.pilot:
        gold = df.set_index("StudyInstanceUID")[rp.TARGETS]
        print("\n--- per-target AUC (LLM vs rule parser) ---")
        rule = pd.DataFrame([rp.extract(r) for r in df["Report"].fillna("")],
                            index=df["StudyInstanceUID"])
        print(f'{"target":20s} {"LLM":>6s} {"rule":>6s}')
        for t in rp.TARGETS:
            y = gold[t].astype(int).values
            a_llm = roc_auc_score(y, lab.loc[gold.index, t].values) if y.min() != y.max() else float("nan")
            a_rule = roc_auc_score(y, rule.loc[gold.index, t].values) if y.min() != y.max() else float("nan")
            print(f"{t:20s} {a_llm:6.3f} {a_rule:6.3f}")
        print(f"\nLLM macro AUC:  {report_auc(lab, gold):.4f}")
        print(f"rule macro AUC: {report_auc(rule, gold):.4f}")
    else:
        cols = rp.TARGETS + [t + "__conf" for t in rp.TARGETS]
        for t in rp.TARGETS:
            lab[t + "__conf"] = 1.0  # LLM has no separate confidence; flag for downstream
        lab[cols].to_csv(OUT / "report_pseudo_labels_llm.csv")
        print(f"\nwrote {OUT/'report_pseudo_labels_llm.csv'} ({len(lab)} rows)")


if __name__ == "__main__":
    main()
