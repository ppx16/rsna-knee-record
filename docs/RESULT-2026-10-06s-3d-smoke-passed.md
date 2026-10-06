# 结果：3D 体数据 smoke **PASS**（但顺带抓出一个会让真臂白训的标签 bug）— 2026-10-06

**内核 `ppxl16/rsna-knee-tpu-3dsmoke` COMPLETE**（16:58Z，排队约 **2 小时**）。
`state/smoke3d_result.json`。预注册在 `docs/PREREG-2026-10-06q-3d-smoke.md`（跑之前落盘）。

## 1. 判定：**PASS**（按预注册那一支）

```
[3d] torch_xla 2.8.0 ; device xla:0 ; world_size 1
[3d] corpus : all_vols.npy (15.90 GB)
[3d] vols shape (3200, 44, 336, 336) uint8
[3d] model  : Conv3d stack width=24, 0.41M params, input (1,32,160,160)
[3d] first batch (2,1,32,160,160) built in 1.2s (CPU, memmap+trilinear)
[3d] median step 0.03s over 11 warm steps ; total 12.9s
[3d] VERDICT: PASS -- Conv3d compiled and stepped on TPU without OOM
```

⇒ **Conv3d + BatchNorm3d + adaptive pool + BCE 在 PJRT 上能编译、能 step、不 OOM。**
这正是这一步要买的东西 —— 而本仓有过"新东西死在 XLA 上"的历史（`A133` 关掉了 CoAtNet-on-XLA）。

**体数据来源也确认了**：`knee-raptor-corpus` = **(3200, 44, 336, 336) uint8, 15.90 GB** —— **每个研究一个真 44 层叠**。
⚠️ 而 A160 建议的那个公开 3D 数据集（`barun2104/...-3d-volumes`）**只有 20 个文件 / 0.01 GB，是样本不是数据集**，
这一点在预注册里已经排除，此处不再重复。

## 2. ⚠️⚠️ 但 `loss nan` 每一步都是 NaN —— 抓出一个会让真臂**白训**的 bug

```
[3d] step  1/12  loss nan  0.34s
[3d] step  2/12  loss nan  0.03s
...
[3d] step 12/12  loss nan  0.02s
```

**机制 PASS 是真的，训练是坏的。** 追下去发现原因在**标签解析**：

`kernel_main.py` 的 `load_labels` 原本只接受 `YES / NO / TRUE / FALSE`，其余一律 `nan`。
而它读的那份表 `report_labels_v4hybrid.csv`（4407 行）**根本不是那种格式**：

```
StudyInstanceUID, ACL, MCL, Medial Meniscus, ..., ACL__conf, MCL__conf, ...
1.2.826...7260,  0.25, 0.25, 0.975, ...
```

⇒ **值是 0.25 / 0.975 / 0.034 这样的【软概率】，而且根本没有 `__verdict` 列。**
⇒ 旧解析器把**每一个格子都变成 NaN** ⇒ 掩码全 False ⇒ **空张量送进 `BCEWithLogitsLoss`** ⇒ 每步 NaN。

**这不是本轮才有的**：`report_labels_v4hybrid.csv` 一直是我们记录里"标签质量 0.8991"的那一份
（见 `report_labels_v5.csv` 等于官方 58 标签、所以我们用 v4hybrid 做标签质量工作）。
**它一直是软标签，只是从来没有代码把它当软标签读。**

### 已修并在真实数据上验证

解析器现在：`YES/TRUE→1.0`、`NO/FALSE→0.0`、**其余尝试 `float()`（软标签这一支）**、失败才 `nan`。

在**真实的 4407 行**上实测（`work/_verify_labels.py`，跑完即删）：

| | |
|---|---|
| 解析行数 | **4407** |
| 单元格 | 52884 |
| **NaN 单元格** | **0（0.0%）** |
| 值域 / 均值 | **0.005 .. 1.000** / 0.343 |
| 每一行都有至少一个可用标签 | **True** |
| 完全有标签的行 | **4407** |

⇒ **旧解析器：0 个浮点、全 NaN。新解析器：52,884 个可用单元格。**

## 3. ⚠️ 这条 PASS **不证明**什么

* **0.03 s/step 是给一个 0.41M 参数、32×160×160 的玩具模型的**。
  预注册要求记录 `median_step_s` 是因为"真臂可行性由它决定"，而**这个数不能外推**：
  把它乘 4407 得到的 "0.02 小时" **没有意义**。
  真臂的每步成本取决于真实的宽度/深度/分辨率，**尚未测量**。
* 本次**没有跑真臂、没有检查点、没有精度结论**。`kernel_main.py` 自己就是这么写的。
* smoke 用的标签是"找得到就用真的、找不到就用随机"，**修复后的解析器尚未在 TPU 上再跑一次**。

## 4. 下一步（按预注册写死的那一支）

预注册的下一条是：**把同一个文件按 `width/depth/hw/studies` 放大成真臂**，
其本地门是 **solo ≥ 0.90，且必须在【大 n 仪器】上量**（gold-58 的 CI 是 ±0.021，**判不动 0.90 这条线**），
对照 `msft2` 在同一仪器上的 **0.8847**。

⚠️ **但先要做的是一件更便宜的事**：修复后的解析器**重跑一次 smoke**，
确认 `loss` 不再是 NaN。**在确认训练真的在学之前，放大模型只是把一个坏掉的东西做大。**
