# 预注册：3D 体数据 **机制 smoke** — 2026-10-06

**内核 `ppxl16/rsna-knee-tpu-3dsmoke` v1 已推**（kernelId 137323723，QUEUED）。本文件在读数前落盘。

## 1. 为什么是这条轴（不是"再训一个更强的同族"）

**A160 ③** 记着一条实测出来的**门偏差**：成员门的基座 = A5(DINOv3) + public20(DINOv2) + Rad(ResNet50)
—— **它完全没有 CoAtNet/Raptor 这一支**。所以 `cn0` 的 +0.0028 里相当一部分是"**补上了一整个缺失的族**"，
不是"这个成员本身强"。

⇒ **修正后的判据**：要找的不是"更强的臂"，而是**归纳偏置与 {DINOv2, DINOv3, ResNet50, CoAtNet}
都不同的臂**。而**前沿整栈都是 2D 切片/窗口**（DINOv2 槽注意力、A5 的 16 片当通道、RadImageNet
ResNet-50、四个 CoAtNet raptor 视角）⇒ **唯一没人试过的归纳偏置是一个真正的 3D 体积。**

**A167 ③** 给出独立的门槛佐证（来自 `fuadmuhammad/rsna-knee-fast-parent-0-957-v2` 的实测记录）：
*"Below ~0.90 solo, a member costs more than its diversity buys."*
⇒ **我们成员线失败的根因是臂在 0.85–0.88，不在 0.90 以上。**

**A163** 同时警告：**"训一个更强的【同族】臂这条路的尽头是冗余，不是分数"**（A5 族四条臂过门全部拿到权重 0）。

⇒ **3D 是唯一同时满足"新归纳偏置"与"未被否证"的方向。**

## 2. ⚠️ 这一步【不是臂】，它只回答一个问题

`kernel_main.py` 自述：

> **does a Conv3d stack compile under PJRT, step, and stay inside HBM?**
> This kernel answers ONLY that. It is not an arm, produces no checkpoint, and is not meant to learn
> anything: labels are real when the label file is found and random otherwise.

**为什么先问这个**：本仓有"新东西死在 XLA 上"的历史（A133 关掉了 CoAtNet-on-XLA）。

**体数据来源**：**我们已经挂载的语料** —— `knee-raptor-corpus` 是 **(4407, 44, 336, 336) uint8**，
每个研究一个**真 44 层叠**。
⚠️ 而 A160 建议的那个公开 3D 数据集 **`barun2104/rsna-knee-mri-processed-3d-volumes` 已经查过：
只有 20 个文件 / 0.01 GB，是样本不是数据集**，所以**不用它**。这条更正写在这里，免得下一个人再去找它。

**成本**：默认 `--studies 64 --steps 12 --bs 2 --depth 32 --hw 160 --width 24`，几分钟级。
**它打印自判**：`[3d] VERDICT: PASS ...` 并写 `/kaggle/working/smoke3d.json`。

## 3. 读数规则（先定，后看）

| 结果 | 判定 | 下一步 |
|---|---|---|
| **PASS**（`VERDICT: PASS` + 写出 `smoke3d.json`） | Conv3d + BN + adaptive pool + BCE 在 TPU 上**能编译、能 step、不 OOM** | **进入下一步：把同一个文件按 `--width/--depth/--hw/--studies` 放大成一条真臂**，预注册它的读数（见 §4） |
| **FAIL / ERROR** | 3D 在本仓的 TPU 路径上跑不起来 | **这条轴以最小代价关闭**，如实记录，**不再投入**；回到"成员质量"线 |
| 排队很久没落地 | **没读到**，不得当成 FAIL | 等，或改用其它设备（A168 ⑤：`machine_shape` 会盖过 enable 开关，改设备时必须删掉它） |

**同时必须记下**：`median_step_s`（中位每步秒数）与 HBM 余量 —— 因为**下一步的可行性完全由它决定**
（若一步要几十秒，4407 研究的真臂就装不进 9 h 会话上限）。

## 4. 若 PASS，真臂的读数规则（现在就先写，免得事后挪门槛）

* **本地门**：3D 臂的 **solo** 必须 **≥ 0.90**（A167 那条独立门槛），**在紧仪器（大 n OOF）上量**，
  不是在 gold-58 上量 —— gold-58 的 CI 是 ±0.021（A83），**判不动 0.90 这条线**。
* **对照**：同一个仪器上 `msft2` = **0.8847**。**3D 臂必须超过它**才有意义。
* **不达标 ⇒ 不接入、不提交**，如实记"3D 在预算内达不到成员门槛"。

## 5. 未验证的（如实记下）

* 该 smoke **从未运行过**：A168 曾记它"占着 TPU 并发位"，但**今天核实：它不在内核列表里，
  且按 slug 查询被拒 ⇒ v1 是刚才才建出来的**。A168 那句话就此更正。
* 从零训练一个 3D CNN 在 4407 研究上**达到 solo 0.90 是很高的要求**；本步只买"能不能跑"这个事实，
  **不保证后面一定能成**。
* smoke 的标签是"找得到就用真的、找不到就用随机"，所以**它的 loss 曲线不含任何精度信息**。
