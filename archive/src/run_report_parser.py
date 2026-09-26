"""运行报告解析器：全量 extract -> 对 58 金标签验证 agreement/silence -> 产出伪标签。

用法:
    python src/run_report_parser.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import report_parser as rp

DATA = Path("D:/kaggle/rsna-knee/data/raw")
OUT = Path("D:/kaggle/rsna-knee/outputs")

TARGETS = list(rp.TARGETS)


def main() -> None:
    df = pd.read_csv(DATA / "train.csv")
    print(f"加载 train.csv: {df.shape}")

    t0 = time.time()
    LAB = pd.DataFrame([rp.extract(r) for r in df["Report"].fillna("")])
    LAB["StudyInstanceUID"] = df["StudyInstanceUID"].values
    LAB = LAB.set_index("StudyInstanceUID")
    print(f"extract {len(df)} 份报告耗时 {time.time() - t0:.1f}s")

    GOLD = df.dropna(subset=TARGETS).set_index("StudyInstanceUID")[TARGETS]
    print(f"金标签 study 数: {len(GOLD)}\n")

    rows = []
    for t in TARGETS:
        y = GOLD[t].astype(int).values
        p = LAB.loc[GOLD.index, t].values
        auc = roc_auc_score(y, p) if y.min() != y.max() else np.nan
        silence = float(((LAB[t + "__npos"] == 0) & (LAB[t + "__nneg"] == 0)).mean())
        pos_rate = float((LAB[t] > 0.5).mean())
        rows.append((t, auc, int(y.sum()), int((1 - y).sum()), pos_rate, silence))

    res = pd.DataFrame(rows, columns=["target", "auc_vs_gold", "npos", "nneg",
                                      "derived_pos_rate", "silence_rate"])
    print(res.round(3).to_string(index=False))
    print(f"\n宏观 agreement AUC: {res.auc_vs_gold.mean():.4f}")
    print(f"平均沉默率: {res.silence_rate.mean() * 100:.1f}%")

    # 保存伪标签（score + confidence）
    cols = TARGETS + [t + "__conf" for t in TARGETS]
    LAB[cols].to_csv(OUT / "report_pseudo_labels.csv")
    print(f"已保存伪标签 -> {OUT / 'report_pseudo_labels.csv'}")


if __name__ == "__main__":
    main()
