# RESULT — 2026-10-07 — 微调臂/配方两个方向的审计：三个发现，两个改判

> 用户指示"就主攻微调臂和配方两个方向"。本轮先做**前提审计**而不是直接烧算力，结果两条方向的判断都被改写。
> 所有结论都从产物或已落盘的实测记录里读出，不是推演。

---

## 发现 1 —— shipped 训练器从来没有过那个修复（已修并验证）

`kernels/kernel-raptor-train` 与 `-rule` 的 `load_raptor` 只解包 `student`，没有 0 张量断言。
用**真权重**验证（`python work/verify_maxspan_ckpt.py`）：

| | v5 maxspan | v9 finespacing |
|---|---|---|
| 顶层键 | `model, gold_auc, aucs, src, res, arch, lab, epoch, swa_over` | 同（无 `swa_over`） |
| `gold_auc` | **0.9214481792** | **0.9031125207** |
| `arch` / `res` | `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` / 384 | 同 |
| `src` | `timm-pretrained` | 同 |
| **旧加载器（只解包 `student`）** | **0 backbone 张量 · 静默零** | **0 backbone 张量 · 静默零** |
| 新加载器 | **478 backbone + 8 head** | 同 |

head 8 张的名字：`clsW, clsb, norm.weight, norm.bias, att.0.weight, att.0.bias, att.3.weight, att.3.bias`
—— 与我们的 `RaptorClassifier` **逐字相同** ⇒ 那个 `model` 字典**就是**我们的 state_dict。

**已修**（`work/patch_raptor_train_member.py`，两份都改，`compile()` 验证）：
解包 `model/student/state_dict/net/backbone/ema` + **0 张量 raise**；新增 `--init_ckpt`（载入**完整** state 含 head，匹配不足一半即拒）；新增 `--eval_only`；两份 metadata 删掉 `machine_shape`。

---

## 发现 2 —— 方向一那次实验【已经执行过并成功了】，不是"还没做"

`kernels/kernel-msft2` 的 `load_raptor`（line 389-423）**就是修好的那版**，且额外携带那 8 张 head，并自带实测记录：
> *"Dropping them is what made the 2026-09-28 run start from an untrained head and score **0.8795 at epoch 0 against a parent at 0.9214**."*

其预置 argv（line 92-98）就是"从 maxspan v5 微调"：`--ckpt raptor_ft_coatnet_v5_full_swa.pt`、`coatnet_rmlp_2_rw_384 @384`、4 epoch、bs 2、`--k_eval 24`、`--grad_ckpt`、`seed 42`、`tag msft0`、`folds 5 fold 0`。

⇒ **`msft2` = 正确的 v5(0.9214) backbone+head warm-start 微调，solo 0.9069，是我们唯一被公榜量到的增益。**

---

## 发现 3 —— gold-58 仪器建成，且对 msft2 **精确等于公榜**

`work/gold58_instrument.py`（零成本，只用已在本地的 npz）：**173 个 `raptor_gold_*.npz` 全部落在同一批 58 个研究上**（同 ids，d-truth 0.00000）。

| 臂 | gold-58 macroAUC | 公榜 solo |
|---|---|---|
| `msft0_swa`（msft2） | **0.9069** | **0.9069** ← 四位小数完全相同 |
| `msft0`（best，非 SWA） | 0.9048 | — |
| `cn0` | 0.8779 | 0.8769（记录） |
| `nx2240` | 0.8639 | — |

⇒ 本地仪器对 msft2 **给出公榜那个数**。此前标记的"0.9069 是公榜数字、本地未必一致"这条边界**解除**。

---

## 改判 1 —— 方向二（v9 当冻结成员）**关闭**

**D14，逐字**：
> ⛔ **「给 Raptor 臂加第 5 个视角」—— 两次独立尝试、两次 0.941，关闭**
> `add-widedense-arm`（v4）**0.941**；`add-finespacing-arm`（v9 finespacing，80 槽/78 窗口，**作者亲自点名**）**0.941**。两者都让其余四臂同比例（×0.8）⇒ 干净的「加一个」。⇒ 与 C14（多样性 ≈ +0.001）、C16 一致：**公榜分辨率下读不出东西，不再投入**

唯一没跑过的变体是**剂量测试** `add-finespacing-heavy`（v9 = 0.50）：`kernels/kernel-exp/add-finespacing-heavy` **建好但从未推**（A32）。**D14 反对投入。**

---

## 改判 2 —— 我自己写的判据「gold-58 上 Δ ≥ +0.0005」**测不出来**

**D16 / C68**：gold-58 的**配对 95% CI 半宽 = ±0.021**，比要判的 +0.01 还宽 ⇒ 无信息。
⇒ 我写的 **+0.0005 比该仪器的分辨率高 42 倍**。
同类实证：**C78** `nxfull` vs `nx2240` = **+0.0016**，落在 ±0.021 里 = 零；**C81** `_RAD_EXCLUDE` 的 `'Fracture'` 是**用 gold-58 做的决定，而它是错的**。
⇒ 而且 D16 记着我**此前就违反过自己定的 B12 规则**（不要用 1 折判小旋钮）。

**规则修正**：臂间比较用 **OOF（±0.004，soft 优先）**；gold-58 只用于**需要与前沿臂同口径的绝对数字**。

---

## 因此：两个方向的真实状态

| 方向 | 状态 |
|---|---|
| **配方 · v9 加成员** | **关闭**（D14 两次独立零结果；剂量测试未推且 D14 反对） |
| **配方 · reader 腿权重** | 只有 **w=0.15 越过铜线 → 0.944**；w=0.30 / TTA / community-stack 全 0.943。**0.944 就是公开面天花板**，再调权重不可能更高 |
| **微调臂 · v5 warm-start** | **已做已收获**（`msft2` 0.9069） |
| **微调臂 · v9 warm-start** | **唯一未做的变体**：从 v9(0.9031) 起点走同一配方。v9 是作者家族**最强单模**（0.932 公榜 solo vs maxspan v5 的 0.927），几何已文档化（80 槽 22/18/15/10/15，2–98%，140mm，336px，**k_eval 78**） |

⚠️ **诚实的先验警告**：按 **A167 的 11× 衰减律**，`msft2` solo 0.9069 只换来 **+0.00025** 的成员增益。
即使 v9 warm-start 成功，预期成员增益也在这个量级 —— **低于公榜分辨率**。
⇒ 在 0.944 已经是公开天花板的前提下，**继续追公榜分数本身已接近无收益**。

---

## 已推出去的算力

`kernel-eval-v5v9`（`ppxl16/rsna-knee-eval-v5v9`，v1 **RUNNING**），一次 GPU 会话评四个组合：

| tag | 检查点 | k_eval | 用途 |
|---|---|---|---|
| `ev_v5maxspan` | v5 maxspan | 24 | **对照**：必须复现 0.9214±0.005 ⇒ 证明载入器与几何正确 |
| `ev_v5maxspan62` | v5 maxspan | 62 | 作者评分协议 |
| `ev_v9fine` | v9 finespacing | 24 | v9 的**绝对**数字（D16 说 gold-58 适用此用途） |
| `ev_v9fine62` | v9 finespacing | 62 | 作者评分协议 |

`--folds 0` ⇒ `gold_ids` = competition `train.csv` 里 12 标签全非空的研究（58 个），不需要标签文件，与之前每个 raptor 臂**同一个仪器**。
监视：`python work/watch_eval_v5v9.py`。预检：0 blockers（那条 AUC-without-n 是误报，JSON 里已写 `n_gold`）。

**读法定死**：**若 v5 对照没复现，v9 那个数不许读成"v9 弱"——先修设置。**

---

## ⛔ 第 (2) 步不是"选错仪器"，而是【任何本地仪器都测不出来】—— 这一步作废，无法修复

我先把它的错法判成"gold-58 分辨率不够，改用 OOF 就行"。**这个自我修正本身也是错的**，量一下就清楚：

| 仪器 | 分辨率（配对 CI 半宽） | 本仓库依据 |
|---|---|---|
| **gold-58**（58 研究，单折） | **±0.021** | D16 / C68（`nx2240` vs `nx3360`：Δ −0.0012，CI [−0.0222, +0.0190]） |
| **870-OOF**（fold-0 held-out） | **±0.004** | D16（"臂间比较用 OOF(±0.004, soft 优先)"） |
| **要判的效应量** | **≈ +0.0002** | A167 的 11× 衰减律，用 `msft2` 实测外推：solo 0.9069 → 成员 **+0.00025** |

⇒ 即使换成 **870 条 OOF（±0.004）**，也**仍然比要判的 +0.0002 宽 20 倍**。
**我们把所有本地仪器都用了，还是够不着这个尺度。** 这不是仪器选型问题，是**效应量本身低于可测下限**。

**因此第 (2) 步的判据（`Δ ≥ +0.0005`）无论用哪个仪器都不可执行**，与 D20 无关的那部分也成立。
**⇒ 关于"某个成员加进去到底有没有用"的决策，在本地做不出来；** 要么在**公榜**上做（分辨率 0.001，仍偏粗），要么按**机制/多样性**先验做，并**明说这是一次信念下注，不是测量**。

⚠️ 这条与用户既定方针直接相关：**"别只在意公榜分数"是对的，但在本任务上，本地仪器恰好无法替代公榜去判成员级效应** —— 两者都不能判 +0.0002。
⇒ 结论不是"回到只刷公榜"，而是：**不要再把算力投在成员级微调上**（D22 的量化依据），把力气放在**能产生 ≥0.005 量级差异的东西**上。

---

## 附：公开面重扫（2026-10-07）—— 干净的否定结果

重扫 `kernels_list(competition, sort_by=dateCreated)` 最新 40 条，逐条核对了任何自称 > 0.944 的：

| ref | 日期 | 结论 |
|---|---|---|
| `ctogaurav/rsna-knee-pure-coatnet-0946` | 10-06 19:22 | **标题的 0946 不是它的分数**。它是 `ryokucha/rsna-knee-d4-blend-0946-ours10` 的**逐字拷贝**（作者自述 "I did not write any of the code"），而后者**自称 0.943**（badge "Public LB: 0.943"，进度表止于 0.943）；"0946" 指它**融合进去的公开 0.946 管线**（其代码注释：`0.946 public submission unchanged before adding our explicit 10% leg`）。 |
| `sameerk2004/rsna-knee-sil-d3-dino-w35 / w40 / dino-perlabel / cnx10 / tri-super` | 10-07 07:30–07:54 | 同一个 `goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944` 基座的**一串 fork**，自述"keep that core and only tighten the independent reader/inference leg" —— **正是我们已做过的 reader 腿**（w=0.15 → 0.944）。无新机制。 |
| `haideptry/rsna-knee-0944-sota-2xt4-fast-stack-convnext-blend` | 10-07 03:34 | 同一 0.944 基座。 |
| `rogersjohnson/rsna-e1b-eff-rad-standalone`、`kaijunlin404/rsna-knee-fast-2xt4-inference` | 10-07 | 效率/EfficientNet 支线，无分数声明。 |

⇒ **公开 notebook 面确实顶在 0.944，这次扫描没有找到任何高于它的配方或机制。**（注意口径：这是**公开 notebook** 的天花板；**榜单**最高 0.959、139 队 ≥0.946 —— 见 C89，榜上比我们高的队伍用的是未公开方法。）

⇒ **"去公开面找一个更好的配方"这条捷径，本次扫描未找到入口。** 方向二若还要走，只能走**未公开的机制**，而那需要新证据才能立项。



---

## 已建成待推：`kernel-v9ft`（方向一唯一未做的变体）

`work/build_v9ft.py` 从 `kernels/kernel-msft2` 的**同一份训练器**生成 `kernels/kernel-v9ft`，
并**机械证明** diff 只有 3 行（脚本会在行数变化或行数 ≠ 3 时直接 FATAL）：

| 行 | msft2 | v9ft |
|---|---|---|
| 87 | `glob('**/raptor_ft_coatnet_v5_full_swa.pt')` | `glob('**/raptor_ft_coatnet_v9_full.pt')` |
| 89 | 报错文案 | 报错文案（指向 finespacing 数据集） |
| 97 | `--tag msft0` | `--tag v9ft0` |

其余逐字相同：`coatnet_rmlp_2_rw_384 @384`、4 epoch、bs 2、k 12、k_eval 24、`--grad_ckpt`、
seed 42、`folds 5 fold 0`、workers 0 ⇒ **唯一的变量就是起始检查点**。

⚠️ **改用 GPU 而非 TPU**：msft2 的 metadata 是 `enable_tpu: true`，但 **D17** 记录 torch_xla 上训练
`coatnet_rmlp_2_rw_384` 死在 timm 的加性相对位置偏置那次加法上（关 `fused_attn` 只是换个地方撞）。
metadata 里**不写 `machine_shape`**（A168 坑③）。数据集 = msft2 的 5 个 + `dreaddevelopment/raptor-knee-finespacing`。
预检 **0 blockers**；非 ASCII 6 字节且 GBK 可解（与 msft2 同款字节，msft2 推成功过）。

**推送被 `Maximum batch GPU session count of 2 reached` 挡住**（eval 内核占 1，ARC clone `arc-agi-3-duck-18-1gc` QUEUED 占 1）。
不取消 ARC（它在跑）。改用**有界**轮询推送器 `work/push_v9ft_when_free.py`（每 4 分钟重试，最多 25 次，成功即退）——
这是对**一个具体阻塞**的一次有界等待，不是自动定时任务。

