# PREREG — 2026-10-07 — 方向一(微调臂) / 方向二(配方)

> ## ⛔ REVISION 2（同日，写完本文数小时后）——第 2、3 节被推翻，先读这段
>
> 本文第 2 节（把 v9 当**冻结成员**加进集成）和第 3 节的判据**都作废**。两条独立的反证：
>
> **① 方向二的 v9 加成员路线【已经做过两次，两次都平】—— D14，逐字：**
> > ⛔ **「给 Raptor 臂加第 5 个视角」—— 两次独立尝试、两次 0.941，关闭**
> > `add-widedense-arm`（v4 widedense）**0.941**；`add-finespacing-arm`（v9 finespacing，80 槽/78 窗口，**作者亲自点名**要配 wide span + wide dense）**0.941**。两者都让其余四臂保持同比例（×0.8）⇒ 是干净的「加一个」，不是重排。⇒ 与 C14（多样性 ≈ +0.001）、C16 一致：**在公榜分辨率下这条路读不出东西，不再投入**
>
> 唯一没跑过的变体是**剂量测试** `add-finespacing-heavy`（v9 = 0.50、其余 ×0.50）：`kernels/kernel-exp/add-finespacing-heavy` **建好但从未推**（A32）。D14 判定不要再投入。
>
> **② 本文写的判据「gold-58 上 Δ ≥ +0.0005」【测不出来】—— D16 / C68：**
> > ⛔ **用 `gold-58`（单折、配对）来判「臂之间」的小差异**：**它的配对 95% CI 半宽是 ±0.021**，比要判的 +0.01 **还宽** ⇒ **无信息**。⇒ **臂间比较用 OOF（±0.004，soft 优先）；gold-58 只用于需要与前沿臂同口径的【绝对】数字。**
>
> 我写的 +0.0005 比 gold-58 的分辨率高 **42 倍**。C78 是同类实证：`nxfull` vs `nx2240` 的 **+0.0016** 落在 ±0.021 里 = 零。C81 更直接：`_RAD_EXCLUDE` 的 `'Fracture'` 就是**用 gold-58 做的决定，而它是错的**。
> ⇒ 而且 D16 记着我**之前就违反过自己定的 B12 预注册规则**（不要用 1 折判小旋钮）。这次不要重犯。
>
> ---
> ### REVISION 2 之后的真实状态
>
> | 方向 | 结论 |
> |---|---|
> | **方向二：v9 作为冻结成员** | **关闭。** 0.20 剂量已测 = 0.941 平（D14）；0.50 剂量建好未推，D14 反对。**不再投入。** |
> | **方向一：v9 warm-start 微调** | **仍然开着，而且是唯一的活杠杆。** v5 那次已做（`msft2`，solo 0.9069，唯一被公榜量到的增益）；**从 v9 起点做同一配方从未做过**，而 v9 是作者家族**最强单模**（0.932 公榜 solo vs maxspan v5 的 0.927），几何也已由数据集描述**文档化**（80 槽 22/18/15/10/15，2–98%，140mm，336px，**k_eval 78**）。 |
>
> **修正后的判据（方向一）**：用 **OOF（±0.004）**，不用 gold-58。gold-58 只用来核对"绝对水平是否在 0.90 之上"。
> **已经推出去的 `kernel-eval-v5v9` 仍然值得跑**：它的价值不在判 Δ，而在于
> (a) **v5 对照**必须复现 0.9214 —— 这是**载入器和几何是否正确的证明**；
> (b) v9 的 gold-58 是一个**绝对**数字，正是 D16 说 gold-58 **适用**的那类用途。
>
> ### ⚠️⚠️ REVISION 2 补充：读 v9 那个数【之前】必须知道的几何不匹配
>
> **v9 的文档化几何与我们语料的几何不是同一个**（数据集描述逐字）：
>
> | | v9 finespacing | 我们的语料 / 本次评测 |
> |---|---|---|
> | 槽数 | **80**（22/18/15/10/15） | **64 切片**（`_SLOTS64` 系列） |
> | 跨度 | 2–98% | 2–98%（maxspan） |
> | 裁剪 / 分辨率 | 140 mm / **336 px** | 140 mm / 336 px（语料）；本次跑 **res 384** |
> | 评分窗口 | **k_eval 78** | 本次 **k_eval 24 与 62** |
>
> ⇒ **本次对 v9 是一次【分布外】评测**：它的头（attention pooling over slots）是在 80 槽上学的，我们喂 64 槽的采样。
> **因此 v9 的数字【低于】0.9031 是预期行为，不能读成"v9 弱"。**
> 反过来说，**若 v9 在 64 槽下仍然接近或超过 0.9031，那才是强信号**（说明它对采样密度不敏感）。
>
> **⚠️ 已知先例**：`add-finespacing-arm` 那次实验（D14，读到 0.941 平）是**按文档化的 80 槽 / k_eval 78 配好几何**才跑的（其 experiment.json 逐字记着 "80 slots (22/18/15/10/15) over 2-98 percent, 140 mm crop, 336 px, and k_eval 78"）。
> ⇒ **本次 eval-only 用的是 maxspan 的 64 槽配置，所以它比 D14 那次更不"公平"于 v9。**
> **真要公平评 v9，必须先复刻 80 槽 / k_eval 78 的采样** —— 那是本次没有做的一步，**在下结论前要补**。
>
> #### 补：80 槽配置在我们仓库里是【现成的】，但有一个未解的疑点
>
> **现成**（`scripts/ens_experiment.py`）：
> · `SLOTS80_SRC`（line 801）给出逐槽定义 `_SLOTS80 = [("Sagittal", 1, 22), ("Sagittal", 0, 18), ...]`；
> · `finespacing-v9` 臂：`"slots": "_SLOTS80"`（line 792）；
> · **两条硬断言**（line 885-888）：槽数必须 **5 条且合计 80**，否则 ABORT；**`k_eval` 必须是 78**，否则 ABORT；
> · `scripts/dump_arms.py`（line 14）独立列出 `finespacing-v9 (336, 80 slots, span .02-.98) -> 78`。
>
> ⚠️ **疑点（必须先验证，别跳过）**：`scripts/corpus64_manifest.py` 写 `SLOTS = 64`，并校验语料数组形状
> `[*, 64, RES, RES]`，span `[0.02, 0.98]`、crop 140 mm。**若语料是把每个研究预计算成 64 个切片，那 80 槽就凑不出来。**
> 而 D14 的 `add-finespacing-arm` 确实跑成过（读到 0.941）—— 所以二者必有一处我理解错了：要么语料实际可提供 >64 切片
> （`_SLOTS44/64/80` 只是**从中挑哪些**的采样表，不是上限），要么那次用的是别的渲染路径（community-stack 的 `SLOTS` 从 DICOM 现场渲染）。
> **⇒ 在主张"对 v9 公平了"之前，先实测语料每个研究的可用切片数。** 这一步没做，就不要下"v9 在本语料下只有 X"的结论。
> 复现入口：`python scripts/dump_arms.py` 与 `python scripts/corpus64_manifest.py`。
>
> ---

> 本文 = 用户指示"就主攻微调臂和配方两个方向"之后的作战计划与预登记。
> 所有数字来自**产物本身**，不是笔记。复现命令写在每段末尾。
> ⚠️ **第 2 节与第 3 节的判据已被 REVISION 2 作废，保留原文仅作审计轨迹。**

---

## 0. 本轮先纠正了两个已流传的说法（都用真权重验过）

复现：`python work/verify_maxspan_ckpt.py`

| 检查项 | 实测 |
|---|---|
| `raptor_ft_coatnet_v5_full_swa.pt` 顶层键 | `model, gold_auc, aucs, src, res, arch, lab, epoch, swa_over` |
| v5 `gold_auc` | **0.9214481792** —— 与 A168 记的 0.9214 逐位一致 ✅ |
| v5 `arch` / `res` | `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` / **384** |
| v5 `src` | **`timm-pretrained`** ← A168 未记：它自己也是从 timm 训的，不是 SSL |
| **旧加载器（只解包 `student`）** | **0 个 backbone 张量 —— 静默零** ⚠️ |
| 新加载器 | **478 backbone + 8 head**，head 名 `clsW/clsb/norm.weight/norm.bias/att.0.*/att.3.*`，与 `RaptorClassifier` **逐字相同** |
| backbone 参数量 | 72.89 M |

**⚠️ 更正 1：`kernels/kernel-raptor-train` 与 `kernel-raptor-train-rule` 从来没有过 A168 说的那个修复。**
两者都只解包 `student`，且没有 0 张量断言 ⇒ 用 maxspan v5 当 `--ckpt` 会**解出空字典、`load_state_dict({}, strict=False)` 静默通过、打印 "loaded 0 tensors"、然后从 timm 从头训**。
已修（`work/patch_raptor_train_member.py`，两处都改，用 `compile()` 而非 `ast.parse` 验证——上一次打补丁就是 `ast.parse` 没抓住的）：

1. `load_raptor` 依次尝试 `model / student / state_dict / net / backbone / ema`，**0 张量直接 raise**；
2. 新增 `--init_ckpt`：加载**完整** state（backbone **+ head**），**匹配不到一半张量就拒绝**（静默重初始化是这条线最贵的失败模式）；
3. 新增 `--eval_only`：只跑 `evaluate()` 然后退出 —— 这是"先花分钟级代价量出这个成员加不加分"的入口；
4. 两份 metadata 删掉 `machine_shape`（A168 坑③：继承来的 `machine_shape` 会让 `enable_gpu:true` 仍被算作 TPU 会话）；
5. 两份 metadata 的 `id` 未被覆盖（A168 坑②的另一半：push 会按 `id` 改名）。

**⚠️ 更正 2（重要）：方向一那次实验【已经执行过，而且成功了】。**
`kernels/kernel-msft2/knee-mri-training-the-twelve-finding-model.py` 的 `load_raptor`（line 389-423）**就是修好的那版**——解包 + 0 张量 `raise` + **额外携带 8 张 head 张量**（"msft2 patch"），并且自带一条实测记录：

> *"Dropping them is what made the 2026-09-28 run start from an untrained head and score **0.8795 at epoch 0 against a parent at 0.9214**."*

它的预置 argv（line 92-98）就是"从 maxspan v5 微调"：

```
--arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k  --res 384  --epochs 4  --bs 2
--k 12  --k_eval 24  --grad_ckpt  --ckpt <raptor_ft_coatnet_v5_full_swa.pt>
--seed 42  --tag msft0  --folds 5  --fold 0  --workers 0
```

⇒ **`msft2` = 正确的 v5(0.9214) backbone+head warm-start 微调，solo 0.9069，是我们唯一被公榜量到的增益。**
所以 A168 的"已修"对 `kernel-msft2` 为真，只是**没有传播**到 shipped 训练器。**方向一的核心不是"还没做"，而是"做过了、已收获"。**

---

## 1. 两条线汇合到同一个产物：`raptor_ft_coatnet_v9_full.pt`

同一个目录里的 **v9 finespacing**（本地 `downloads/raptor_v9/`，数据集 `dreaddevelopment/raptor-knee-finespacing`）：

| | v5 maxspan | **v9 finespacing** |
|---|---|---|
| `gold_auc` | 0.9214482 | **0.9031125** |
| arch / res | coatnet_rmlp_2_rw_384 / 384 | 同 |
| head 8 张、名字 | 与 `RaptorClassifier` 逐字相同 | 同 |
| 旧加载器 | 0 张量 | **0 张量** |

⭐ **v9 的 0.9031 高于 A167 的 ~0.90 成员门槛**，而 **C42 明确记着"我们的集合只用了 v5/v8/v10(+v5-reverse)"——v9 从未挂过**。

⭐ 而且 v9 正好站在**唯一被测出大效应的那条轴**上：

- **C47（作者自己的表，逐字）**：`44 层@42窗 → 0.926`；`64 层@42窗 → 0.924`；同权重 `62 窗 → 0.927`；`64 层宽跨度@62窗 → 0.928`；**`80 层@78窗 → 0.932`**。原话：*"Widening the span only pays if slices are added to keep the sampling dense"*。
- **我们自己的 `gap0`**：±1→±2 隔层 = **−0.012**，与作者"44 层摊到宽跨度 = 0.917 vs 0.926 = −0.009"**同号同量级** —— 两个独立仪器互证。
- ⇒ 好方向是**更密**（finespacing），v9 就是这个方向上的现成权重。

**数据集描述里一条 A168 没记的关键信息**（逐字）：
> *"The corpus behind them samples **64 slices** per study across **2 to 98 percent** of each series, cropped to a **140 mm box** ... resized to **336 pixels**. Scoring uses **62 windows** per study, which is every window position a 64-slice volume holds."*
> *"...they score **0.928 on the public leaderboard as a single model**, with no ensembling."*

⚠️ **我们训练器默认 `--k_eval 24`，作者评分用 62。** 所以"复现作者"与"和已有臂可比"是两个不同的设置，必须分开跑、分开读。

---

## 2. 实验 A —— 方向二(配方)：v9 作为**冻结成员**

**为什么先做这个**：不需要训练，只要一次 eval-only，代价分钟级；而且它回答的是"这个成员加不加分"，也就是最终唯一重要的问题。

**做法**
1. 跑一次 eval-only，拿到 v9 在**同一个 58 研究 gold 集**上的预测（`--folds 0` ⇒ `gold_ids` = competition `train.csv` 里 12 标签全非空的研究，**不需要标签文件**，与之前每个 raptor 臂同一个仪器）。
   - 主设置 `--k_eval 24`（与已有臂可比）；副设置 `--k_eval 62`（复现作者的 0.928 协议）。
2. 用 `scripts/oof_fusion_probe.py` 把 v9 按其**视图权重**混进现有集合，在本地 gold-58 仪器上量 Δ。

**预登记判据**
- 本地 Δ **≥ +0.0005** ⇒ 值得占一次提交额度；否则**关闭**，不提交。
- v9 solo 必须落在 **0.9031 ± 0.005**。若明显更低，说明几何/窗口数不对，**先修设置再谈混**，不许把设置错误读成"没有多样性"。
- ⚠️ 已知反例：**v4 widedense 加进去 = 0.941 平**。所以先验并不高，Δ 门槛就是这个原因。

**若要提交**：只在与 `56878092`(0.944 primary) 相同的配方上做**单变量**改动，其余逐字不动；读规则沿用分数制（0.944 = 跨过铜线 = 换 primary 并停止花额度）。

---

## 3. 实验 B —— 方向一(微调臂)：从 v9 warm-start 微调

**为什么是 v9 而不是再来一次 v5**：v5 已经被 `msft2` 做过了。v9 是**同一配方、不同起点**的干净 A/B，而且起点更低（0.9031 < 0.9214）——所以它必须真的赚回差距。

**做法**：照抄 `msft2` 配方逐字，只换 `--ckpt`
```
--arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 --epochs 4 --bs 2
--k 12 --k_eval 24 --grad_ckpt --ckpt <raptor_ft_coatnet_v9_full.pt>
--seed 42 --tag v9ft0 --folds 5 --fold 0 --workers 0
```
⚠️ **必须携带那 8 张 head 张量**（msft2 patch 的实测：丢掉 head ⇒ epoch 0 = 0.8795 vs parent 0.9214）。用 `--init_ckpt` 走完整 state，或沿用 `kernel-msft2` 的 `load_raptor`（返回 `(src, head_state)`）。

**预登记判据**
- `gold_auc` **> 0.9031**（微调必须比自己的起点高），**且**
- 与 `msft2` 的 0.9069 比：**必须 ≥ 0.9069**，否则这条臂还不如已有的那个 v5 微调，不值得占槽位；
- 最终仍以"A 实验的本地 Δ ≥ +0.0005"为准——**solo 高不等于成员有用**（A167 的 11× 衰减律：`cn0` solo 0.8769→+0.0028，`msft2` 0.9069→+0.00025）。

**槽位**：TPU 优先（GPU 稀有）。⚠️ GPU 模式下**不要带 `machine_shape`**。

---

## 4. 执行顺序与成本

| # | 动作 | 代价 | 决定什么 |
|---|---|---|---|
| 1 | v9 eval-only @ k_eval 24（+62） | 分钟级 1 次 GPU/TPU 会话 | 方向二的前提；同时给出方向一的起点基线 |
| 2 | 本地 gold-58 混 v9，量 Δ | 免费 | 方向二成立/关闭 |
| 3 | v9 warm-start 微调（仅在 1 给的数字可信后） | 1 次训练会话 | 方向一是否还有一枪 |
| 4 | 只有 Δ ≥ +0.0005 才提交 | 1/5 额度 | —— |

**不要重开**（都已测过关闭）：五折 msft2、OWN_W 扫描、specialist/meniscus、外部数据、Kaggle Models、加视图、自训 ConvNeXt、路由/权重/标定、蒸馏检查点(0.8895)、3D 两个起点。

---

## 5. 本轮遗留的诚实边界

- `msft2` 的 0.9069 是**公榜**数字；它的本地 gold-58 OOF 是否与之一致**尚未核对**。若不一致，A 实验的 Δ 门槛要按本地仪器重新定，而不是照搬公榜。
- v9 的 `0.9031` 是**作者自己的 gold_auc**，不是我们的仪器量出来的。实验 A 的第 1 步就是这个数字的**本地复现**。
- `--k_eval 24` 与作者的 62 不同，所以"本地复现 = 0.9031"**不应期待精确相等**；判据写成 ±0.005 是这个原因。
