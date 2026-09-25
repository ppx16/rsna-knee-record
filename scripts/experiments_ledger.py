"""Generate state/EXPERIMENTS.md -- the ledger of every experiment tried, with its verdict.

WHY (user request 2026-09-19, restated 2026-09-21: "把所有实验结果和结论整理下，别以后忘了还重复造轮子")
    The project already has state/FACTS.md (what is TRUE) and state/NOW.md (what is HAPPENING).
    Neither answers the question a new experiment proposal has to clear first:
    **"has this exact knob already been tested, and what happened?"**

    So this generates one document from the machine-readable sources:
      * training runs      : every landed `raptor_oof_*.npz` under downloads/ (numbers measured
                             from the files themselves) + a hand-written tag -> "what changed"
                             map and verdicts (judgements cannot be derived, so they are written)
      * member candidates  : the `raptor_ft_*.json` the training script writes, whose
                             `best_gold_auc` is the SAME metric the frontier's released arms
                             report -- so this table is directly comparable to their 0.90-0.92
      * LB / fusion        : state/lb_ground_truth.csv + scripts/ens_experiment.py descriptions
      * closed directions  : the D-table of state/FACTS.md (auto-read, so it cannot drift)
      * never tried        : a hand-written candidate list, so the gaps stay visible

    2026-09-21 FIX: the first version scanned only `downloads/cv/*`, so every TPU arm -- which
    lands in `downloads/<name>/` -- was invisible, and the candidate list still offered
    `--res 336` and `--k 24` as "never tried" LONG AFTER both had been run and judged. That is
    exactly the failure this file exists to prevent, so the scan is now a recursive glob keyed
    off `cv_score.parse_name` (tag comes from the FILENAME, not the directory).

Usage:
    python scripts/experiments_ledger.py            # write state/EXPERIMENTS.md
    python scripts/experiments_ledger.py --print    # print instead of writing
"""
from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import json
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "state" / "EXPERIMENTS.md"

# ---------------------------------------------------------------- training runs
# tag -> (what changed, verdict).  Verdicts are hand-written because they encode a judgement;
# the NUMBERS are read from the npz/json files so they can never be stale.
CV_MAP = {
    "cva0": ("baseline: v4hybrid labels, 12 ep, res 224, k 12/24, ±1 window, seed 42, fold 0",
             "对照"),
    "cva1": ("same as cva0, fold 1", "对照（gate 1 的另一半）"),
    "cva2": ("same as cva0, fold 2", "对照"),
    "cvg0": ("1-epoch control (crippled)", "gate 1 的对照臂"),
    "gap0": ("window spacing ±1 -> ±2 (隔层窗口)", "⛔ 否决：−0.0119 REAL（D10）"),
    "r2880": ("--res 224 -> 288, fold 0", "⭐ 筛出：+0.0049 OOF / +0.0261 gold"),
    "r2881": ("--res 288, fold 1", "⭐ 确认：+0.0056（三折合并才勉强站住）"),
    "r2882": ("--res 288, fold 2", "⭐ 确认：+0.0092 OOF REAL"),
    "r3360": ("--res 336 (语料原生分辨率), fold 0", "⭐⭐⭐ **+0.0121 REAL（最强单折）**"),
    "r3361": ("--res 336, fold 1", "⭐⭐⭐ +0.0073 REAL"),
    "r3362": ("--res 336, fold 2", "⭐⭐⭐ +0.0103 REAL"),
    "k240": ("--k 12 -> 24 (每研究窗口数翻倍)", "⛔ 否决：−0.0035（数量没用，见 C52）"),
    "seed0": ("seed 42 -> 123（只换种子）",
              "⭐ 噪声地板：+0.0018；配对 CI ±0.0053 ⇒ 比 B12 的 ±0.0026 宽一倍（C50）"),
    "tpuprobe": ("TPU 探针：1 epoch / 160 研究（XLA 移植验证）",
                 "✅ 端到端跑通（训练→存盘→OOF），X5 修复生效（C53）"),
    "tpuprobe2": ("TPU 探针：3 epochs（量稳态）", "✅ 稳态 0.41 s/步（C56）"),
    "t2240": ("**TPU**: vit_small, res 224, 12 ep, fold 0",
              "✅ TPU 成本基线（77 min/折）；gold 0.8391"),
    "nx2240": ("**TPU**: convnext_base (89M), res 224, 12 ep, fold 0",
               "⭐⭐ **成员候选最好：gold 0.8639**（比 vit_small +0.0248）"),
    "nxl2240": ("**TPU**: convnext_large (198M), res 224, fold 0",
                "⛔ 过拟合：0.8485 < base（D15）"),
    "nx3360": ("**TPU**: convnext_base, **res 336**, fold 0",
               "⭐ res 增益在 OOF 上 REAL：+0.0078 soft（gold-58 太钝，见 D16）"),
    "nxfull": ("**TPU**: convnext_base, res 224, **--folds 0 = 全量训练**",
               "⭐ 真正的成员配置；gold-58 与前沿臂同口径"),
    # --- rented AutoDL 4090D, 44-slot published corpus, the architecture-isolation line ---------
    "cn0": ("**AUTODL 4090D**: CoAtNet-rmlp-2_rw_384 @384, k 12, 12 ep, --grad_ckpt, 44 槽, fold 0",
            "⭐⭐ OOF 0.8623 / gold 0.8779；对 nx2240 +0.0327、对 r3360 +0.0329 **都 REAL**"
            "（后端混淆只占约 1/5，见 C83）。⚠️ 成员价值低：D14 两次测到「更好的成员 → 0.941 不动」"),
    "cn1": ("**AUTODL 4090D**: 同 cn0, fold 1",
            "⭐ OOF 0.8441 / gold 0.8873（**折间不可比**：不同折不是同一个仪器，只说明离散度）"),
    "d64nx3360": ("**TPU**: convnext_base, res 336, fold 0, **64 槽密语料**（已发布数据集）",
                  "⭐⭐ **密语料线的首次真实证据**：对 `nx3360`（44 槽）soft ΔOOF **+0.0068** "
                  "[+0.0024, +0.0108]，**100% 胜出 ⇒ REAL**（阈值 0.004）；二值口径 +0.0129"
                  "（0.8368 → 0.8497）；gold-58 +0.0014 NOT RESOLVED（D16 应验）。"
                  "⚠️ 代理局限：convnext ≠ CoAtNet（D17）"),
    "nx3361": ("**TPU**: convnext_base, res 336, fold 1",
               "⚠️ **没有配对基线**：同折同架构的 res 224（`nx2241`）不存在 ⇒ 这一折本身证明不了"
               " res 增益。唯一有效的同折比较是 vs `cn1`：+0.0074 soft REAL（但混了架构+后端两个变量）"),
}

# ---------------------------------------------------------------- member candidates
# ⚠️ the script's sidecar JSON does NOT carry `arch` (only the .pt does), so the backbone has to
# come from this map. The first version read `j["arch"]` and printed `?` for every row -- a
# column that is silently empty is worse than an absent one.
ARCH_BY_TAG = {
    "t2240": "vit_small_patch16_224",
    "nx2240": "convnext_base.fb_in22k_ft_in1k",
    "nx3360": "convnext_base.fb_in22k_ft_in1k",
    "nxfull": "convnext_base.fb_in22k_ft_in1k",
    "nxl2240": "convnext_large.fb_in22k_ft_in1k",
    "cn0": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",
    "cn1": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",
    "nx3361": "convnext_base.fb_in22k_ft_in1k",
    "d64nx3360": "convnext_base.fb_in22k_ft_in1k",
}
DEFAULT_ARCH = "vit_small_patch16_224"
# only the runs that are actually member CANDIDATES -- listing all ~20 vit_small fold runs here
# would bury the comparison this table exists to make (their gold-58 is already in section A).
MEMBER_TAGS = ("t2240", "nx2240", "nx3360", "nxfull", "nxl2240")

# The frontier's own released arms report `gold_auc` from the SAME script + metric + 58 studies,
# so these are the numbers our candidates have to be read against (C16: >0.90 to be worth adding).
FRONTIER_GOLD = [
    ("前沿 `v4` widedense（89M 级）", 0.9167, "→ `dreaddevelopment/raptor-knee-widedense` 描述"),
    ("前沿 `v9` finespacing（73M coatnet）", 0.9031, "→ `raptor_ft_coatnet_v9_full.pt` 的 gold_auc"),
]

# ---------------------------------------------------------------- never tried
CANDIDATES = [
    ("**重建更密语料**：`_SLOTS44` -> `_SLOTS64` / `_SLOTS80`",
     "作者自量的**最大杠杆**（+0.006~+0.011，C14）；镜像证据在我们手上：`gap0` 更稀 = −0.012。"
     "**配方已反推出大部分**（140mm 裁剪 ✅、`linspace(lo,hi,n)` ✅）",
     "CPU 解码仅 0.2~0.7 h（C40/C44）⇒ **代价在工程不在算力**；**还差** `lo/hi` 规律 + 缺类替换规则（约 27% 研究缺类）"),
    ("更多 epoch（12 → 24）", "`nx2240` 的曲线到 ep11 仍在涨 ⇒ 欠训练",
     "convnext 稳态 283 s ⇒ ~1.9 h + 编译，会话上限内"),
    ("`--bb_lr` / `--head_lr` 调参", "训练超参；从未动过", "1 折 ~1.2 h（TPU）"),
    ("`--freeze_blocks N`", "冻结前 N 个 block", "未测；先验低（减容量，而 C65 说容量有用）"),
    ("把 `nxfull` 挂进 0.941 笔记本并提交", "**唯一还没走完的一步**：需要一个 solo >0.90 的成员（C16）",
     "1 次提交 + ~0.5 GPU-h；**但 D14 已两次测出「加视角」在公榜上是平的**"),
]


def training_rows():
    """Every landed training run, numbers read from the npz itself."""
    import numpy as np
    from cv_score import load, macro_auc, macro_soft_auc, parse_name

    found = {}
    for p in glob.glob(str(ROOT / "downloads" / "**" / "raptor_oof_*.npz"), recursive=True):
        parsed = parse_name(p)
        if not parsed:
            continue
        tag, kind, _fold = parsed
        if kind != "final":                     # base OOF only (skip swa / epens / epN)
            continue
        found.setdefault(tag, p)

    rows = []
    for tag in sorted(found):
        P, Y, _I = load(found[tag])
        gold = ""
        # prefer the JSON: the script writes best_gold_auc there and it survives a partial download
        for j in glob.glob(str(ROOT / "downloads" / "**" / ("raptor_ft_%s.json" % tag)),
                           recursive=True):
            try:
                gold = "%.4f" % json.load(open(j, encoding="utf-8"))["best_gold_auc"]
                break
            except Exception:
                pass
        if not gold:
            for g in glob.glob(str(ROOT / "downloads" / "**" / ("raptor_gold_%s.npz" % tag)),
                               recursive=True):
                Pg, Yg, _ = load(g)
                gold = "%.4f" % macro_auc(Yg, Pg)[0]
                break
        what, verdict = CV_MAP.get(tag, ("(未登记 —— 请补进 CV_MAP)", ""))
        rows.append((tag, len(Y), macro_auc(Y, P)[0], macro_soft_auc(Y, P)[0], gold, what, verdict))
    return rows


def member_rows():
    """The member-candidate sweep, from the JSONs the training script writes."""
    rows = []
    for p in sorted(glob.glob(str(ROOT / "downloads" / "**" / "raptor_ft_*.json"), recursive=True)):
        try:
            j = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        tag = j.get("tag", "?")
        if "best_gold_auc" not in j or tag not in MEMBER_TAGS:
            continue
        arch = ARCH_BY_TAG.get(tag, DEFAULT_ARCH)
        rows.append((tag, arch, int(j.get("res", 0)),
                     int(j.get("n_train", 0)), float(j["best_gold_auc"]),
                     [h.get("gold_auc") for h in (j.get("hist") or [])]))
    rows.sort(key=lambda r: -r[4])
    return rows


def ens_descriptions():
    """variant name -> first sentence of its description in ens_experiment.py."""
    spec = importlib.util.spec_from_file_location("ee", ROOT / "scripts" / "ens_experiment.py")
    ee = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(ee)
    except SystemExit:
        pass
    table = {}
    # ⚠️ the registry is named EXPERIMENTS, not EXPS -- the first version guessed EXPS and the
    # getattr default silently returned {}, so every row's "what changed" column came out "-".
    # A join that fails silently into a default is worse than one that crashes.
    reg = getattr(ee, "EXPERIMENTS", None)
    if reg is None:
        print("WARNING: ens_experiment.EXPERIMENTS not found -- descriptions will be empty")
        return table
    for name, cfg in reg.items():
        d = (cfg.get("desc") or "").strip().replace("\n", " ")
        table[name] = (d.split(". ")[0][:150] + ".") if d else ""
    return table


def lb_rows():
    p = ROOT / "state" / "lb_ground_truth.csv"
    if not p.exists():
        return []
    desc = ens_descriptions()
    out = []
    for r in csv.DictReader(open(p, encoding="utf-8")):
        m = re.search(r"rsna-knee-ens-([a-z0-9\-]+?)(?:-v\d+)? v\d+", r.get("desc") or "")
        variant = m.group(1) if m else ""
        what = ""
        for k, v in desc.items():
            if k.replace("v2src-", "") == variant or k == variant:
                what = v
                break
        out.append((r.get("ref", ""), r.get("score", "") or "-",
                    variant, what, (r.get("date") or "")[:16]))
    return out


def closed_rows():
    p = ROOT / "state" / "FACTS.md"
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").split("\n"):
        if re.match(r"^\| D\d+ \|", line):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) >= 4:
                rows.append((cells[0], cells[1], cells[2], cells[3]))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args()

    L = []
    L.append("# EXPERIMENTS —— 做过的实验台账（**生成式，不要手改**）")
    L.append("")
    L.append("> `python scripts/experiments_ledger.py` 生成。**开新实验前先来这里查一行** ——")
    L.append("> 这个文件存在的唯一目的就是「不要重复做一个实验、浪费额度」。")
    L.append("> 数值从落地产物（npz/json）与比赛账本读出，判决是手写判断 + FACTS 的 D 表。")
    L.append("")
    L.append("⚠️ **最容易被重复提的 5 件事（都已经测过并且关闭）**：")
    L.append("1. 融合层再调权重/路由/标定 → **D11/D12/D13**（公榜四角单调，全在噪声里）")
    L.append("2. 给集成加第 5 个视角 → **D14**（两次独立尝试都是 0.941 平）")
    L.append("3. 继续加大骨干容量 → **D15**（198M 比 89M 更差）")
    L.append("4. 用 `gold-58` 单折判臂间小差异 → **D16**（±0.021，无信息；用 870 条 OOF）")
    L.append("5. 在 TPU 上训 coatnet / 给 TPU 排多个任务 → **D17 / D18**")
    L.append("")

    L.append("## A. 训练实验（同一折内配对才可比 —— §B13）")
    L.append("")
    L.append("| tag | n | OOF(二值) | OOF(软) | gold-58 | 改了什么 | 判决 |")
    L.append("|---|---|---|---|---|---|---|")
    for tag, n, mb, ms, g, what, verdict in training_rows():
        L.append("| `%s` | %d | %.4f | %.4f | %s | %s | %s |"
                 % (tag, n, mb, ms, g or "-", what, verdict))
    L.append("")
    L.append("⚠️ **只有同一折内的配对差有意义**：同一配方三折 OOF 极差 **0.041**，而配对 CI 只有 **±0.0026**。")
    L.append("⚠️ **两个口径的仪器分辨率差 7 倍**：`OOF`(870 条) 配对 ≈ **±0.004**；`gold-58`(58 条) 配对 ≈ **±0.021**（D16）。")
    L.append("")

    L.append("## B. 成员候选（`gold-58` 口径 —— **这一栏才能和前沿的臂直接比**）")
    L.append("")
    L.append("前沿发布的臂用的是**同一个脚本、同一个指标、同一批 58 条**，所以它们的 `gold_auc` 与下表同口径。")
    L.append("**C16 的门槛：solo >0.90 才值得加进集成。**")
    L.append("")
    L.append("| tag | 骨干 | 参数 | res | n_train | **gold-58** | 曲线（逐 epoch）|")
    L.append("|---|---|---|---|---|---|---|")
    for tag, arch, res, ntr, gold, hist in member_rows():
        short = arch.split(".")[0]
        curve = " ".join("%.4f" % h for h in hist[:12]) if hist else "-"
        L.append("| `%s` | `%s` | %s | %d | %d | **%.4f** | %s |"
                 % (tag, short, _params(short), res, ntr, gold, curve))
    for name, g, src in FRONTIER_GOLD:
        L.append("| *前沿* | %s | — | — | 全量 | **%.4f** | %s |" % (name, g, src))
    L.append("")

    L.append("## C. 公榜提交（融合侧；5 次/天）")
    L.append("")
    L.append("| ref | 公榜 | 变体 | 改了什么（取自 ens_experiment 描述） | 提交时间 |")
    L.append("|---|---|---|---|---|")
    for ref, score, variant, what, date in lb_rows():
        L.append("| %s | **%s** | `%s` | %s | %s |" % (ref, score, variant or "-", what or "-", date))
    L.append("")
    L.append("**当前最优 = 规范配方（`56166132`，0.941）。** 四角见 §116.1：")
    L.append("路由开+标定 .40 = **0.941** ／ 标定 0 = 0.940 ／ 路由关 = 0.939 ／ 两个都关 = 0.938。")
    L.append("⚠️ **公榜 SE ≈ 0.008**（B11）⇒ **≤0.003 的差异在这里判不出来**，别拿公榜数字做判决。")
    L.append("")

    L.append("## D. 已关闭的方向（**别再花额度**）")
    L.append("")
    L.append("| # | 方向 | 为什么关掉 | 出处 |")
    L.append("|---|---|---|---|")
    for num, what, why, src in closed_rows():
        L.append("| %s | %s | %s | %s |" % (num, what, why, src))
    L.append("")

    L.append("## E. 还没做过的（候选空间 —— 这是「还能往哪走」的清单）")
    L.append("")
    L.append("| 候选 | 为什么值得 | 代价 / 前提 |")
    L.append("|---|---|---|")
    for what, why, cost in CANDIDATES:
        L.append("| %s | %s | %s |" % (what, why, cost))
    L.append("")

    text = "\n".join(L) + "\n"
    if a.print:
        print(text)
    else:
        OUT.write_text(text, encoding="utf-8")
        print("wrote %s (%d lines)" % (OUT.relative_to(ROOT), text.count("\n")))
    return 0


_PARAMS = {"vit_small_patch16_224": "22 M", "convnext_base": "89 M", "convnext_large": "198 M",
           "coatnet_rmlp_2_rw_384": "73 M"}


def _params(short: str) -> str:
    """Exact map first. The first version also did a bare `startswith` fallback, which matched
    `convnext_large` against `convnext_base` (shared prefix) and printed 89 M for a 198 M model --
    a wrong number is worse than a missing one."""
    if short in _PARAMS:
        return _PARAMS[short]
    for k, v in _PARAMS.items():
        if short and k == short:
            return v
    return "—"


if __name__ == "__main__":
    sys.exit(main())
