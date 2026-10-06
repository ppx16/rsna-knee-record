# 预注册：五折 `msft2` — 2026-10-05

> ## ⛔ **NOT PROCEEDED — and fold 1 has now closed the question with data.** (2026-10-06)
>
> **Fold 1 completed on TPU v5e-8. Its result says the plan was not worth doing, and the quota says it
> was not even possible.**
>
> ```
> [cv] fold 1/5: train 3527 | OOF held-out 822
> [raptor] head tensors carried: 8            <- msft2's critical fix, verified again
> [msft2] HEAD loaded: 8/8 tensors, names and shapes verified
> [zeroshot] VERDICT: PASS (parent bar 0.8963, got 0.8961)
> SWA over epochs [3, 2, 1] | gold 0.9082 (best-epoch 0.9088) no gain
> ```
>
> **Fold 1 = 0.9088 (best) / 0.9082 (SWA); fold 0, the one already in the stack, = 0.9069.** The two are
> the same model to within noise, so a five-fold average would land near **0.907–0.910** — **no material
> gain over the single fold 0**, for the price below.
>
> ⚠️ **And the price was never affordable.** Measured from the quota log:
>
> | | TPU used |
> |---|---|
> | 2026-10-05 08:00 | 3.79 h |
> | 2026-10-06 05:55 | 6.99 h → **+3.20 h** |
> | 2026-10-06 14:20 | 7.34 h → **+0.35 h tail** |
>
> ⇒ **one fold costs ≈ 3.2–3.6 h, not the inherited 2.6 h.** Five folds would have been **16–18 h**,
> against a remaining budget of **16.21 h at the time**. **The plan could not have completed.** The
> "~2.6 h per fold" figure came from `PREREG-2026-10-04-convnext-member.md`, i.e. it was sized for the
> ConvNeXt member — and the CoAtNet fold cost had never been measured. Now it has.
>
> **What fold 1 did buy:** three things that were only assumptions are now verified — the head-carrying
> fix works on a second fold, the zero-shot gate passes, and the CoAtNet recipe reproduces across folds.
> That is worth the ~3.4 TPU-hours; the arm is not.

**状态：构建器 `scripts/build_msft2_folds.py` 已就绪（`--check` 通过）。开跑前写死读数规则。**

---

## 0. 为什么是这一条

`msft2` 是本仓**唯一有正面实测**的东西：它的 SWA 检查点作为第 6 个 Raptor 视角把我们从
rank 708 推到 487（格内 38.89% → 13.66%）。**而产生它的那次是五折设计里的第 0 折** ——
`--folds 5 --fold 0` —— **folds 1–4 从未训过**。

⇒ 这不是"再加一个同类成员"（止损规则禁止的那个），而是**同一个成员做完整**：折平均降方差、
不新增假设，而且**产出我们今天没有的 4407 研究完整 OOF**。

## 1. 设备：TPU（不是 GPU）

`FACTS` **A31**（用户 2026-09-21 定的规矩）：*"GPU 留给提交用，实验走 TPU"*。已核实
`kernel-msft2` 元数据本身就是 `enable_tpu=True, enable_gpu=False` —— fold 0 当年就在 TPU 上跑。
0943 提交内核跑在 `NvidiaTeslaT4`，**训练不动 GPU 额度**。

| | 剩余 | 并发上限 | 单会话上限 |
|---|---|---|---|
| GPU | 24.40 h | 2 | 9 h |
| **TPU** | **16.21 h** | **1**（排队中也占名额，A28/A136） | **≈9 h**（A29） |

⚠️ **13 h > 9 h 单会话上限 ⇒ 一个内核循环跑完五折【不可能】，必须逐折独立推送、串行执行。**
⚠️ 排队不消耗额度（A33），所以串行等待是**墙钟**代价而非预算代价。

## 2. 只训 folds 1–4（fold 0 复用）

fold 0 的产物已经是 `raptor_ft_msft0_swa.pt`（在 `ppxl16/rsna-knee-msft2-ckpt` 里）。
重跑它要烧 ~2.6 TPU-h 去复现一个我们已有的产物。**⇒ 只训 1、2、3、4，共 4 次推送。**

## 3. ⚠️ fold 1 就是成本测量（不是浪费的一次运行）

*"每折 ≈ 2.6 h"* 是从 `PREREG-2026-10-04-convnext-member.md` 继承的，**那是给 ConvNeXt 估的**；
而磁盘上 `msft2` 的日志被截断到 3.5 KB，**CoAtNet 的单折成本从未被直接测过**。

**⇒ fold 1 跑完，从它自己的日志读出真实单折成本，再决定后面几折怎么排：**

| 实测单折 | 决定 |
|---|---|
| ≤ 3.2 h | 全推 folds 2、3、4（总 ≤ 12.8 h，在 16.21 h 内） |
| 3.2–3.9 h | 只再推 folds 2、3（总 ≤ 11.7 h） |
| > 3.9 h | **停**，只用 folds 0+1，并如实记录"五折在预算内放不下" |

## 4. 融合设计：方案 (a)，先注册

今天 `msft2-ours` 是**一个视角、w = 0.20**（Raptor 子融合的 14.3%）。五个检查点不能各占 0.20 ——
那会一次改两件事，结果就不可解释了。

**⇒ 方案 (a)：五折的预测做 rank-mean 合成【一个】成员，插进现有槽位、保持 w = 0.20。**
只改一件事：*"折平均是否优于第 0 折"*。

⚠️ 实现上若要落到 `ARMS`，等价形式是五条臂各 `w = 0.04`（家族份额仍为 0.20）——**但那是近似**：
子融合是按 rank 平均的，逐臂加权与先 rank-mean 再加权不完全等价。这一点在读数时如实标注。

## 5. 读数规则（先定，后看）

| 关卡 | 判据 |
|---|---|
| **本地门（零 GPU）** | 五折拼出的 **4407 研究 OOF macro-AUC** 必须**优于 fold 0 的 0.8847**（one-convention 表口径）。**不优于 ⇒ 不接入、不提交。** |
| **提交门** | 提交前**必须取一次快照**，提交后与新鲜快照比：**格内百分位**要移 **≥3 个百分点**（止损规则的噪声量级）才记为真实提升 |
| **余量** | 每次都要**同时报余量** —— 奖牌看的是绝对前 10% 线，不是格内百分位 |

**基线快照（`state/lb_snapshot_2026-10-05T160516Z.json`）**：teams 5226，rank **551**，铜牌线 **522**，
**margin −29**，`0.943` 格 414..1419（1006），格内 **13.63%**。

## 6. ⚠️ 天花板，说在前面

**即使五折 msft2 完美，它也在 0.000x–0.001 量级** —— 名次仪器能分辨，显示分不能。
**这是巩固，不是突破。** 它不会越过策略文档那条结构性边界（要到 0.95 需要结构上不同的东西）。
本仓其余每条轴都已实测关闭：专科（meniscus 0.940 门掉）、外部/更多数据（已证明关闭）、
Kaggle Models（0.7992 实测关闭）、第三个同类 CoAtNet（止损规则）、自训 ConvNeXt（已实测失败）。

**⇒ 这个计划的真实替代方案是什么都不做。** 它的独立价值在于：**产出完整 OOF**，
把"msft2 有用"从一次性的名次观测变成一个可测量、可复用的成员。

## 7. 止损

> 若 fold 1 实测成本 > 3.9 h，或五折 OOF 不优于 fold 0 ⇒ **停止，不接入，不提交**，
> 按 `RESULT-2026-10-05-two-arms-flat-or-worse.md` 转入收尾，锁定终局配对
> **`56814081`（primary）+ `56599932`（second）**。
