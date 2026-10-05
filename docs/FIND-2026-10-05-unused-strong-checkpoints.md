# FIND 2026-10-05：两个从未用过的强检查点，以及 C48 点名却一直没补的缺口

从"自训练那条线唯一还活着的支路"（`msft2` 式微调：**公开强检查点 + head 能迁移 + 4 epoch + SWA**）
出发，枚举了 Kaggle 上**全部 94 个膝盖相关数据集**，筛出我们没用过的检查点候选，并**用检查点自己的
元数据**判定 head 是否可迁移 —— 不靠任何人的说法。

## 1. ⭐ `dreaddevelopment/raptor-knee-arms` —— 三个异构骨干，同源训练器

顶层键是 `['model', 'gold_auc', 'aucs', 'src', 'res', 'arch', 'lab', 'epoch']`，
头结构是 `clsW / clsb / norm.weight / norm.bias / att.0.* / att.3.*`（8 个张量）——
**与我们 notebook 里 `RaptorClassifier` 的头部定义逐项对应**。

| 文件 | 架构 | **gold_auc** | res | epoch | 头维度 |
|---|---|---|---|---|---|
| `raptor_ft_cnv2b336.pt` | **convnextv2_base.fcmae_ft_in22k_in1k_384** | **0.8901** | 336 | 7 | 1024 |
| `raptor_ft_coatnet384.pt` | coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k | **0.9025** | 384 | 8 | 1024 |
| `raptor_ft_effv2l480.pt` | tf_efficientnetv2_l.in21k_ft_in1k | 0.8716 | 480 | 12 | 1280 |

⭐ **`coatnet384` 的 gold 0.9025 越过 A167 的成员门槛（solo ≥ ~0.90）**，而我们从未用过它。
⭐ **`cnv2b336` 是 ConvNeXtV2 家族、gold 0.8901、头可迁移。**

### 为什么 `cnv2b336` 比我们上次的 ConvNeXt 尝试强得多

上次用的是 `apoorvadixit/rsna-knee-weights-v4cx`，它的 head 是**外来架构**（11 个张量一个都不匹配
`RaptorClassifier`）⇒ 只能 `--allow_bad_warmstart` 从零训 ⇒ **cx0 的 870 研究 OOF 只有 0.8493**。

**⇒ `cnv2b336` 高 0.041，而且它的头结构同源。** 这正是 `msft2` 成功的那条路的形状：
**不同架构族（去相关）+ 同源头（不用从零训）。**

⚠️ **一个尚未解决的细节**：`coatnet384` 与 `cnv2b336` 的头维度都是 **1024**，而 v5/v8/v10
（同为 `coatnet_rmlp_2_rw_384`）在我们栈里走的是 `F_dim=768` 的加载路径。这两者不是同一套配置，
**接入前必须把这个维度对齐搞清楚，否则会静默加载失败或错配。** 这一点我没有查完，如实记下。

## 2. ⭐ C48 点名、却一直没补上的缺口：v9 finespacing

C48 原文：

> *"One of a family of slice sampling variants meant to be ensembled, so **pair it with the wide span
> and wide dense checkpoints for a diversity boost**."* —— 我们的集合已有 wide span（v5）与 wide dense（v10），
> **唯独缺 v9**。单模型公榜 **0.932**（该家族最强单模型，无集成无 TTA）。

**核对结果（逐字查我们自己的 notebook）：**

```
finespacing            False
raptor_ft_coatnet_v9   False
_SLOTS80               False
v9_full                False
ARMS 实际挂载的 file： v5, v10, v5(reverse), v8, cn0, msft0_swa   ← 没有 v9
```

**⇒ 缺口是真的，至今未补。** 而 `dreaddevelopment/raptor-knee-finespacing` 就在那里：
`raptor_ft_coatnet_v9_full.pt`（293 MB）。

**它需要自己的几何（C48 已记全）**：`_SLOTS80` = Sagittal-FS 22 / Sagittal 18 / Coronal-FS 15 /
Coronal 10 / Axial 15，`k_eval = 78`，span **2%–98%**，crop **140mm**，img 336，res 384。

⚠️ **而且 C48 附带一条必须遵守的警告**：*"Every number quoted… goes with a window count, and the two
are not separable… When comparing checkpoints, **hold the window count fixed**."*
⇒ v9 若接进来，必须用 `k_eval_fixed: 78`（我们的 per-arm 覆盖机制**已经存在**，为 cn0 建的）。

**⇒ 我们已经有能力渲染它**：`build_cache` + `globals().update(SLICE_BAND=..., CROP_MM=..., CACHE_SLICES=...)`
这套覆盖模式，在半月板专科那一次已经被验证跑通（缓存 `(N,6,12,336,336)` 成功产出）。

## 3. 其它筛出来但优先级较低的

- `dreaddevelopment/raptor-knee-fullspan` → `v7_full_swa`；`raptor-knee-widefov` → `v6_full_swa`（同一家族）
- `dreaddevelopment/raptor-knee-widedense`（518 MB，82 票）—— ⚠️ **D7 记过我们的 `add-widedense-arm` = 0.941 平**，但那是**作集成成员**；**作 msft2 的 warm start 是另一回事**，未测过
- `kawaharataishi/rsna-knee-coatnet-distilled-checkpoint`（518 MB）—— 早前 403，**slug 已变成单数形式**，值得重试
- `stevenleehans/rsna-knee-champ-members-only`（394 MB）、`stevenleehans/rsna-knee-llm199-e30-members`（394 MB）
- `myominhtet/rsna-knee-coat5fold`（1.3 GB）、`prvsiyan/rsna-knee-pulse-v6-checkpoints-public-20260813`（453 MB）
- `stefanoblando/rsna-knee-stage1/2/3-checkpoints`、`tannguyenngoc/rsna-knee-coatnet-a100-best`
- `tbongc/rsna-knee-raptor-v8geo-*` / `v10geo-*`（早前 403，现已可见）

## 4. 下一步的建议顺序

1. **先接 `raptor_ft_cnv2b336.pt`（gold 0.8901，ConvNeXtV2，头同源）** —— 它是"不同架构族"这条轴上
   目前**最强的、且从未用过**的候选，而且形状与我们上次失败的 ConvNeXt 尝试完全相同、只是头能迁移。
2. **同时补 v9 finespacing** —— C48 指出的缺口，作者本人推荐它配 v5/v10 用，且渲染机制已验证。
3. **`coatnet384`（0.9025）** 先解决那个 1024 vs 768 的维度问题再谈。
