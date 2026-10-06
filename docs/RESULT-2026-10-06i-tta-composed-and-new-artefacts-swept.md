# 结果：TTA 臂合成成功 + 新一轮公开产物清点 — 2026-10-06

## 一、TTA 臂：**合成成功**（预注册的管线验证通过）

`ppxl16/rsna-knee-0943cfmsft2-rdrtta` **COMPLETE**（2026-10-06T08:10:35Z 读取），
`state/tta_arm_preview.json`：

```
reader leg (tta): stack submission backed up (830 bytes)
reader leg (tta): 3 checkpoints; src=/kaggle/input/datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader
                                   data=/kaggle/input/competitions/rsna-knee-abnormality-detection
reader leg (tta): free GPU memory (GB) [15.4, 15.5]
reader leg (tta): inference took 32s
reader leg (tta): BLENDED 3 ckpts, Synovitis_w=0.08 other_w=0.30, 32s
```

判定字段：`wrote_own=true`、`blended=true`、`failed=false`、`stack_complete=true`、
**`composed=true`**、`errors=[]`。

⇒ 挂载的 Stack++ 数据集被找到、reader 子进程退出 0（它要从旁边 import `knee`/`preprocess`，
接口不匹配会死在那里）、`_own.csv` 写出、**逐标签混合真的执行了**、**没有走 fallback**。
**这是一次管线验证，不是一次实验读数** —— 它证明这个臂可推，不证明它更强。

**它解锁了什么**：预注册规则里"若 reader 腿好 ≥3 个百分点 ⇒ 换主提交并**打出预注册的 TTA 臂**"
这一支，从"要先花 GPU 验证能不能跑"变成了**随时可执行**。仍然**按预注册等 reader 腿的读数**再决定，
不提前开火。

## 二、公开产物清点：5 个新数据集，逐条判定

按 `dataset_list(sort_by='updated')` 扫出的、**上次清点之后新出现**的候选：

| 数据集 | 更新 | 大小 | 判定 |
|---|---|---|---|
| `easonyanyan/rsna-knee-c12-weights` | 10-05 22:49 | 368.5 MB | ❌ 拒绝 |
| `takashimanaoya/rsna-knee-i00g-checkpoints-flat` | 10-05 05:58 | 483.9 MB | ❌ **拒绝（有硬数据）** |
| `meguminksdj/rsna-knee-distilled-models-v1` | 10-05 01:35 | 376.0 MB | ❌ 拒绝（命题本身不成立） |
| `novatron513/rsna-knee-cone12-source-public` | 10-05 03:07 | 0.03 MB | ❌ 拒绝（只是源码） |
| `kiritohayate95/rsna-knee-medgemma-grades` | 10-06 02:49 | 0.4 MB | 未评估（弱标签，非成员） |

### 2.1 `i00g` —— 用数字否掉的，不是用印象

只取了 3 个 summary JSON（不拉 484 MB）：

| seed | gold fold macro-AUC | 验证研究数 | 骨干 |
|---|---|---|---|
| 17 | **0.8106** | 15 | dinov2/small |
| 2026 | **0.8087** | 15 | dinov2/small |
| 42 | **0.7831** | 15 | **home/naoya/rsna-sparse** ← 换了模型 |

三条独立理由，任何一条都够：

1. **样本量**：**15 个**验证研究上算 12 标签 macro-AUC ⇒ CI 约 **±0.10**。0.81 与 0.90 在这条
   仪器上不可区分，而它自己文件里就写着 *"Do not compare configurations from this fold alone"*。
2. **远低于成员门**：A167 要求 solo ≥ ~0.90。
3. **轴重复**：DINOv2-Small，而我们栈里已经有 `cn0`（lixin73 的 DINOv2，874 研究上 0.8634）。
   而且 `seed42` 用的是**另一个骨干**，所以"3 seed 集成"连架构都不一致。

监督构成也说明它不是竞争轨：`train_gold=43` vs `train_pseudo=4348`（99% 是
`report_labels_v2.csv` 的 LLM 伪标签）。这是一个**做对照实验的研究轨**（RNG 对齐、ghost 干预、
哈希可重放），不是冲榜轨。

### 2.2 `meguminksdj` —— 命题不成立，不是"没测过"

文件名 `raptor_ft_coatnet_v5_distilled_swa.pt`，`raptor_ft_` 前缀 + `coatnet_v5` 与我们栈里的
检查点**同族同名**，理论上零成本换槽。

**但它没有解决我们的任何问题**：蒸馏买到的是**更小/更快**，而我们**不缺时间** ——
9 h 上限里我们只用 < 30 min。而且蒸馏学生通常 **≤ 教师**。
我们需要的槽位是**更高**，不是**更省**。

⇒ 拒绝。**唯一会改变这个判断的条件**：如果它其实是**集成蒸馏**（学生 ≥ 教师）且有分数证据。
目前**零分数声明**（0 votes、描述为空），所以不投入。

### 2.3 `c12` / `CONE-12` —— 成本高、无质量证据

读了 `novatron513/rsna-knee-cone12-source-public` 的全部源码。**CONE-12 不是新骨干**：

* `config.py`：`feature_dim = 1024`，目标是"`for the first CONE-12 comparison`"；
* `models.py`：三个**匹配的聚合头** —— `GatedMILBaseline` / `ConeBase` / `Cone12`
  （层级软序列 → 平面偏置切片 → top-k apex 稀疏池化，带可学习 `lambda_max=2.0`）；
* 输入是 `EvidenceBatch`（`payload[B,N,1024]` + `valid_mask`/`series_index`/`plane_index`/`location`）。

⇒ 它是**预计算特征上的 MIL 路由研究包**，配套 `compare.py`/`diagnostics.py`。
要用它必须**复现整套特征抽取契约**，而它**不附带任何分数**。
`easonyanyan/rsna-knee-c12-weights`（"ConvNeXt 384"，2 个文件）也没有分数声明。

⇒ 拒绝：**高集成成本 + 零质量证据**。这不是"新轴"，这是"新工作量"。

## 三、下一步（写死）

1. **等 `56872230` 的读数**，按 `state/submit_56872230.json` 里写死的漂移感知规则判：
   基准 **~16.32% 格内**（不是记录的 16.032%，差额是漂移）。
2. **等蒸馏门**（`ppxl16/rsna-knee-msft2-gate-distilled`，仍 QUEUED）的 gold-58 macro-AUC。
3. reader 腿 **≥3 个百分点**好 ⇒ 换主 + 打出 TTA 臂（**已证明可跑**）；
   在 ±3 内 ⇒ 终局配对不变；更差 ⇒ 丢掉这条腿。
4. **今天还剩 4/5 次提交**，在 (1) 出数前不动用。
