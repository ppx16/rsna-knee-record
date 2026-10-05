# 预注册：半月板专科臂 `0943cfmsft2m-men`（2026-10-05）

**臂**：`56845973`（PENDING），内核 `ppxl16/rsna-knee-0943cfmsft2m-men` v2。
**= `0943cfmsft2m`（cn0 + 骨折修复 + msft2 + MCL）加 `public0033` 半月板专科，固定 90:10，只打在
Medial + Lateral Meniscus 两个标签上，其余十个标签一字不动。**

构建器：`scripts/build_meniscus_arm.py`（构建期编译该 cell，并断言**结果**：几何常量在、还原在、只加一格）。
本地预检：`work/preflight_meniscus_bundle.py` —— runtime 自己的 `verify_bundle` **PASS**、`self_test` **PASS**。

---

## 为什么这一条与之前所有臂**不是同一类**

前面每一条都是"往集成里再加一个成员 / 再叠一个已实测的改动"。**这一条不是。**

`public0033` 是**只打两个标签的专科**，按作者**预先注册**的配方加上去：
`parent_weight 0.90 / bag_weight 0.10`、`targets: [Medial Meniscus, Lateral Meniscus]`、
`tie_method: average`、`rerank_after_blend: false`，并把 `alternate_weights: [0.05,0.15,0.20,1.00]`
与 `retry_after_public: true` 列为**禁止项**（作者不许自己看分后调权重）。

而基座自己的日志说过：**"Medial Meniscus / Medial OA / Lateral OA / Effusion / Synovitis / Contusion
是各臂真正不同、外层权重只在这里起作用的地方。"** ⇒ 专科打的正是这里，而不是已经饱和的地方。

⚠️ **它的 `runtime:` 块全是 `false`** —— `notebook_created` / `private_kernel_pushed` /
`competition_submission_created` 全部未执行。**这条配方设计完整、从未跑过。**

## 可见集运行：整条链路通过

```
[men] cache (3, 6, 12, 336, 336) uint8 | mask (3, 6) float32 | studies 3
[men] inference 12.6s for 3 studies -> /kaggle/working/_men0033/public0033_bag_raw.csv
[men] receipt status: passed
    Medial Meniscus    mean=0.3093 std=0.1544 min=0.1629 max=0.5228
    Lateral Meniscus   mean=0.2903 std=0.0414 min=0.2323 max=0.3264
[men] BLEND APPLIED at w=0.10 on ['Medial Meniscus','Lateral Meniscus']
```

**合理性检查（这些是真的过了）：** 概率非退化（`std>0`）、落在合理区间（0.16–0.52）、
cache/mask 契约在 cell 内被**自己断言**而不是让它深埋在 runtime 里才炸、schema 与研究对象顺序断言全过、
receipt `passed`、且**十个未触及标签被断言与基座逐值相同**。

⚠️ **跑失败的那一版（v1）值得记**：它死在 `Public0033InferenceError: bag_raw_csv filename drift` ——
runtime 把输出**文件名和父目录都钉死**（`public0033_bag_raw.csv` / `public0033_cached_inference_receipt.json`，
且必须是 `work_dir` 直接子级、拒绝覆盖）。**而 fail-closed 按设计工作**：日志明写
`base submission left untouched`，可见集 SHA 仍是基座的 `095bcf70…`，**没有把坏文件交出去**。

## 读数规则（先定，后看）

**提交前基线（快照 2026-10-05T07:22:52Z）**：

| | |
|---|---|
| 我们 | **rank 518** |
| 铜牌线（前 10%） | **518** |
| `0.943` 格 | 376..1385（1010 队） |
| 余量 | **+0**（正好压在线上） |
| 格内百分位 | **14.07%** |

| 观察 | 判定 |
|---|---|
| 格内百分位 **优于 14.07%** | **真实提升** → 进终局候选，与 `0943cfmsft2`（38.89%→13.66%）并列比较 |
| 格内百分位**持平或更差** | **这一条也无效** —— 那么"专科"和"堆成员"一样到头，记录为负结果 |
| 公榜 ≤ 0.940 | 真回归，丢掉 |

⚠️ **提交前后必须各取一次快照**（漂移实测 ≈ −34 名/天，16 小时就能吃掉 23 名）。
⚠️ **可见集的 `rho` 不能当证据**：n=3 时 Spearman 只能取 `{-1, ±0.5, +1}`（本仓给 M448 记过这个量化陷阱）。
日志里那个 `rho=-1.0000` **是三个点上的必然取值之一，不是"强烈分歧"的证据。**

## 我预期的方向（写下来便于对账）

**持平或略优，概率不高。** 理由：它是 10% 权重、2/12 个标签；且它的父是 **0.936**，我们是 0.943 且已叠了
cn0 + msft2 + MCL。**但它是本轮唯一"结构上不同"的杠杆，而且它的目标是基座自己点名的十个标签里最该动的两个。**

⇒ **若它有效，说明"标签级专科"这条轴还活着；若无效，则本轮所有轴（公开产物 / 新成员 / 叠加 / 专科）
都被实测否证，应该转入收尾。** 两种结果都有信息量。

## 未解决的真实风险（不埋）

**几何是按 YAML 重建的。** 缓解是复用我们自己的渲染器（`read_slot` 的联合 p1–p99 恰好就是
`joint_12_slice_p1_p99`）而不是新写一套，但它**不是**逐字节复现原作者的缓存 ——
原作者的缓存文件哈希（`group0.336.uint8.memmap` 等）钉在 `run_contract.json` 里，而那是他们自己的研究集合，
**我们无法复现，也无法比对**。⇒ **若几何有细微偏差，成员会静默产出"看着合理但错的"预测，而没有任何本地仪器能看出来。**
