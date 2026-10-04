# FIND 2026-10-04：比我们分高的 356 个队，用的是什么方法

**问题**：榜上在我们之上的队用了什么额外方法。

**方法**：把 `0.943` 之上的队映射到用户名（快照 `2026-10-04T12:44`），再逐个查他们的**公开 kernel 与 dataset**，
和我们的栈做差集 —— 就是找出 M448 那个漏子用的同一招。

⚠️ **先说清这条调查的性质**：**这是一个"没有公开记录"的调查。** 下面的第 1 节是**测量**，
第 2 节是**推断**，我把两者分开写。

---

## 1. 测量（可复核）

### 1.1 榜的结构

| | |
|---|---|
| 队伍总数 | 5107 |
| 我们 | **rank 487 · 0.943** |
| **严格高于我们的队** | **356** |
| 其中**单人队** | **290 / 356** |
| 分数分布（0.944→0.963） | 65 · 62 · 47 · 33 · 26 · 17 · 12 · 16 · 10 · 12 · 7 · 13 · 9 · 7 · 9 · 6 · 1 · 1 · 2 · 1 |
| 榜首 | **0.963** |

### 1.2 ⭐ **前 27 名里，没有一个人为这个比赛发过任何东西**

用 API 逐个查了前 27 名的 **39 个用户名**（`kernels_list(user=)` + `dataset_list(user=)`）：

- **0 个膝盖相关的 kernel**
- **0 个膝盖相关的 dataset**
- 唯一被我过滤器匹配到的"rsna"产物，全都是**别的 RSNA 比赛**的：
  * `wadakoki`（rank 3）的 `rsna-infer-pipeline-public` 是 **`rsna-2024-lumbar-spine-degenerative-classification`**（腰椎）；
    他另有 `rsna-spine-final-models` = **21,468 MB**。
  * `vaillant`（rank 13）：`RSNA L-Spine Models V4/V5`（3.3 / 4.1 GB）、`RSNA-STR PE Checkpoints`（5.1 GB）、
    `RSNA-STR PE Detection Train JPEGs`（**53.6 GB**）。
  * `quan0095`（rank 19）：42 kernels / 20 datasets，`rsna_2024_sagittal_t2_3d_models`（815 MB）、
    `rsna_2024_axialt2_3d_models`（812 MB）、`rsna_2024_sagittal_t1_3d_models`（816 MB）、
    `rsna_2024_axial_classify_model` / `saggital_classify_model` / `detection_model`。

**⇒ 结论不是"他们用了什么方法我们不知道"，而是"他们把方法藏起来了"。**

### 1.3 公开区确实封顶 —— 而且这次是**配方作者本人**证的

`prvsiyan/the-bee-s-knees-final-rsna-push`（2026-10-02，83 票）逐字自述：

> The public score of **0.940 belongs to that recipe, not to this notebook.** Nothing added here
> improved it: **five controlled variations were submitted and none beat it**
> （three tied at 0.940, two regressed to 0.922）.

`evgendvorkin/rsna-versia-5`（291 票）是一份公开谱系编年史，标题就是
**"Evolution of Public Solutions: 0.937 → 0.94 → 0.941 → 0.942 → 0.943"**。

⇒ **公开区的终点就是 0.943，两次独立确认。**

### 1.4 提交次数的差距（这是一条真实的、结构性的差异）

| 队 | 提交数 |
|---|---|
| 榜首 rhoskeri | **92** |
| knee-deep（rank 9） | **298** |
| YOKKOISHO（rank 5） | **257** |
| dalab（rank 7） | 174 |
| …顶部普遍 | **26 – 298** |
| **我们** | **41** |

⇒ **顶部队伍的迭代次数是我们的 2–7 倍。** 这在 ±0.008 的显示分噪声下只有一种解释：
他们要么有比我们大的效应，要么在用**比显示分更细的仪器**（我们昨天才发现名次就是那台仪器）。

---

## 2. 推断（说明依据，不当结论用）

把上面三条拼起来，最自洽的解释是：

### 2.1 他们的"额外方法"就是**自己训模型，然后私下集成**

证据链：
- 0.943 之上**没有任何公开配方**（1.2、1.3）；
- 唯一两条被证明能到 0.946 的公开 notebook（`d4-blend-0946`、`ours10`），**唯一差异都是私有数据集**（我们两次独立验证过）；
- 这批人在**其他 RSNA 比赛**里的公开产物，全都是**自己训的模型集**（21 GB、53 GB、整套 3D T1/T2 模型）。

### 2.2 ⭐ 一个具体的、我们没在用的方向：**3D 体数据模型**

`quan0095`（rank 19）在腰椎比赛里发的是 `sagittal_t2_3d` / `axialt2_3d` / `sagittal_t1_3d` —— **3D 模型**。

**而膝盖这套公开配方自始至终是 2.5D**（3 层窗口沿深度堆叠，靠 attention 池化）。
我们自己的 A5 也是 2.5D（`n_slice` 取 6 或 16 层）。

⚠️ 但这条**只是旁证**，不是证据：他们在腰椎用 3D，不代表他们在膝盖也用 3D。
而且 3D 在这个比赛里要自己训、要 24 GB 以上的卡，**成本和我们刚起步的 ConvNeXt 不在一个量级**。

### 2.3 一条更便宜的、**公开的、零训练**的候选

`romantamrazov/rsna-knee-dinosaur-v5`（128 票，2026-10-02）明确声称在 **"0.943 public V5"** 这个锚点上
**不需要额外前向、不需要训练、不需要私有权重**，方法是：

```
MAIN DINO 残差，【只用在 ACL 和 MCL】：
    92.7% 校准后的 V5 transformer
     5.5% five-fold smooth focal `soft` pooling
     1.8% weighted DINO member ensemble
免校准的 CoAt 共识：
    75% 原概率均值家族
    25% mean-of-reader-ranks 家族
```

⭐ **而这两个信号我们的栈【已经算了】**（`kernel-0943cf` 里 `states[...]['soft']` 与 `['public']` 都在，
`legacy_fold_soft_window_pool` 也在跑）—— 差别在**最后怎么合**，不在算不算。

⚠️ 但**我们的谱系和它的底不是同一条**：我们用 `maverickss26/rsna-knee-0942-restructured` 这条，
它用 `evgendvorkin/rsna-versia-5`。所以"我们是不是已经在用"**必须逐行对拍才能说**，不能推断。

---

## 3. 对我们最有用的一条结论

**不要再去公开区找"别人用的方法"了。** 这一轮是第二次穷尽核对（F45 那次是第一轮），
结论一致且更强：**0.943 之上的东西不在公开区，而且这次连"谁"都查过了 —— 356 个队，顶部 27 个零公开。**

⇒ 剩下的只有三条，和 `STRATEGY-2026-10-04` 里列的一致：

1. **自己训**（ConvNeXt 在飞；3D 是更远但更大的一步）；
2. **把公开谱系里"算了但没用上"的信号接出来**（`dinosaur-v5` 那条，零训练，但要对拍谱系）；
3. **接受 0.943 并用名次仪器守住位置**（我们目前在线内 23 名，漂移 −2/天）。

⚠️ **而 2.4 万队 / 356 队之上没有公开记录这件事，本身也说明**：想靠"抄"越过 0.943 是不可能的，
**这不是情报问题，是能力问题** —— 而我们的能力缺口已经被量化了：**一个 ≥0.90 solo、且与现栈去相关的新成员。**

---

# 补充：全量普查（464 个用户名）**推翻了我上面那句"没人公开"**

## 普查规模

| | |
|---|---|
| 我们之上的用户名 | 464 |
| 发过至少一个 public kernel | **188** |
| 发过至少一个 public dataset | **103** |
| 他们的 public kernel 总数 | 2284 |
| 他们的 public dataset 总数 | 757 |
| **其中发的是【膝盖】** | **约 21 人（kernel）/ 35 人（dataset）** |

⇒ 所以更准确的表述是：**他们发的东西绝大多数与这个比赛无关；膝盖相关的仍是公开谱系那批人**
（`mattiaangeli` · `pilkwang` · `dreaddevelopment` · `pjmathematician` · `prvsiyan` · `tonylica` …），**封顶 0.943**。

## ⭐⭐ 但有一个例外，而且它**没人注意到**

`tannguyenngoc/rsna-knee-final-locked-publicstyle`（2026-09-12 发布，**0 票**）里是一份
**完整的、带权重的公开配方**：

```
README.txt 逐字：
  RSNA Knee Final Locked Public-Style Bundle
  Locked Gold58 macro AUC:  0.961982
  Raptor:  v5 = 0.50 · v10 = 0.30 · v8 = 0.20      -> probability blend -> rank
  Base:    Raptor rank = 0.90 · DINOv3 rank = 0.10  -> rank
  E10:     10 classes: Base = 0.50 · RadImageNet = 0.50
           Baker's and Fracture: Base only          -> rank
  Final:   E10 = 0.85 · E11 RadImageNet second pass = 0.15
  IMPORTANT: Do not retune these weights on Gold58.
```

`FINAL_LOCKED_PUBLICSTYLE_recipe.json` 里同一套权重，外加 `aggregation` 字段
（`probability_then_rank` / `rank_blend_then_rank`）。

**附带的权重**：`dinov3/m_f0..m_f4.pt`（**5 折 DINOv3**，各 94 MB）、
`radimagenet/ResNet50.pt` + `v52_e11_heads.pt` + `v52_radimagenet_heads.pt`、
`raptor/v5.pt` · `v8.pt` · `v10.pt`。

### 它和我们栈的差别（逐轴）

| 轴 | 我们（`0943cf` 线） | 这份配方 |
|---|---|---|
| **DINO** | **DINOv2-small**，20 成员，**A5 @0.52** | ⭐ **DINOv3**，5 折，**@0.10** |
| Raptor 视角 | 5–6 个（含 reverse、含我们自训的 `cn0`/`msft2`） | **3 个**：v5 .50 / v10 .30 / v8 .20 |
| Rad 排除集 | `Baker's`（+ 可选 `Fracture`/`MCL`） | **`Baker's` + `Fracture`** |
| Rad 权重 | `rad_alpha 0.55` · `second_alpha 0.20` | **0.50 / 0.50，然后 E11 第二遍 0.15** |
| 归约 | `rank_of_member_probability_mean`，outer 0.65 | `probability_then_rank` / `rank_blend_then_rank` |

⭐ **最大的一条差别是 DINOv3。** 而我们全栈是 **DINOv2-small**；前沿另有一本
`mattiaangeli/bend-the-knee-to-dinov3-ensembled`（**223 票**，2026-08-22）—— 我们似乎从未分叉过它。

### ⚠️ 三条必须压住的话

1. **`0.961982` 是 gold58 上的数，不是公榜数。** 本仓自己记过 **"58 例 gold 集不能当裁判"**
   （bootstrap CI 比要追的效果还宽）。它**不能读成"这份配方公榜 0.962"**。
2. **它是 2026-09-12 的**，比我们现在的栈早；前沿后来那三处改动（第三读者 Global96、概率均值归约、
   outer 0.60→0.65）**不在这份配方里**。所以它大概率**不是**一条更高的线，而是**一条不同的线**。
3. **0 票 / 0 下载**说明没人在用，也就没有第三方验证。

**⇒ 但它是唯一一份"来自我们之上、且带权重、且可复现"的公开配方。**
**最值得先做的不是照抄，而是回答一个具体问题：DINOv3（@0.10）能不能替换掉我们的 DINOv2+A5（@0.52）那一段？**
这两者的差别不是调参，是**换了一个底座**：DINOv3 的 5 折权重就在这个数据集里，**不需要训练**。

---

# ⭐ 用这份参考做对拍的结果（2026-10-04）

这是本仓**第一次**有一份独立、逐文件 SHA 校验过的公开配方可以拿来对照。
工具：`work/crosscheck_vs_reference.py` · `work/decode_rad_cal.py` · `work/compare_calibration_numbers.py`

## 1. 常量对拍

| 项 | 参考（`rsna-baseline` v15，已验证 0.941） | 我们 | |
|---|---|---|---|
| Raptor 概率权重 | `[0.6, 0.1, 0.1, 0.2]` | v5 .60 / v10 .10 / v5-rev .10 / v8 .20 | ✅ 一致 |
| hybrid public : residual | `0.6 : 0.4` | `coat_private_alpha = 0.40` | ✅ 一致 |
| 逐标签外层权重 | ACL .75 · **MCL .6** · MedMen .8 · LatMen 1.0 · **MedOA .6** · LatOA .75 · **PF OA .6** · **Effusion .6** · **Synovitis .6** · **Baker's .6** · **Contusion .6** · Fracture .75 | 五个覆盖值**完全相同**；其余七个是**我们的默认 0.65** | ⚠️ 见下 |
| 标定表 | *"12-output, 88-input; **seven gated targets**"* | 见 §2 | ✅ |
| 成员数 | 20 DINOv2-small + 5 DINOv3-small + 5 ref Rad heads + 5 E13 Rad heads + 3 Raptor + 3 residual-CoAt epochs | 同 + 我们自训的 `cn0`/`msft2` | ✅ 结构相同 |

⚠️ **外层默认 0.65 vs 0.60 不是漂移**：C79 记着前沿那次 0.942→0.943 的三处改动之一就是
**"outer CoAtNet default 0.60 → 0.65"**，而这份参考复刻的是 **0.941 谱系**。⇒ **我们是更新的那条，不是走偏了。**
我们 notebook 的注释自己就写着 `# The default rose 0.60 -> 0.65 in this release; the five overrides are
unchanged from the 0.941 routing.` —— **与参考逐字吻合。**

## 2. ⭐ 本仓一条"来源不明"的记录，用数关闭了

本仓对 `_RAD_CAL` 的记载是：*"一个**来源不明**的嵌入式 base64/zlib 产物…… **Do not touch it**"*。

它其实**就在 notebook 里、可解码**（`_RAD_CAL_PAYLOAD` 是 base64+zlib）。解出来：

```
keys : ['mean','scale','coef','intercept','gate','protocol_columns','groups']   ← 与参考 calibration.json 键集完全相同
coef : 12 × 88        ← 正是 "12-output, 88-input"
gate : 7 个 = ACL · Medial OA · Lateral OA · PF OA · Effusion · Baker's · Contusion
```

**⇒ 然后逐系数比对 1275 个标量：**

```
scalars: ours=1275  ref=1275  common=1275
EXACT-equal scalars: 1275
differing scalars  : 0
max |difference|   : 0.000e+00
```

**⇒ 我们那张表就是公开参考的标定表，一个系数都不差。**

**这条的意义有两条：**
1. **"来源不明"结束** —— 它不是来路不明的东西，它是公开配方冻结的配方件。仓库写的 *"Do not touch it"* **结论不变，但理由升级**：从"不知道是什么"变成"知道它是前沿冻结的、且我们这份与它逐位相同"。
2. **它同时证明我们这一路没有被动过手脚** —— 标定系数若被谁改过，macro AUC 会静默偏移，而**任何门都看不出来**。

## 3. 对拍没有发现的

**在参考记载的全部常量上，没有一处无法解释的偏离。** 我们不在公开配方之下，也不在它旁边 —— **我们就在它上面，
多了两个自训视角（`cn0`、`msft2`）。** 这与本轮第三次普查的结论一致。


---

# ⛔ 更正：上面那条"换底座"的假设**是错的，我们早就在用 DINOv3**

**2026-10-04，动手做这个实验时第一件事就把它否掉了。**

## 我错在哪

我从 `kernel-0943cf/kernel-metadata.json` 的 `model_sources: ["metaresearch/dinov2/PyTorch/small/1"]`
推出"我们全栈是 DINOv2-small"。**那只对了一半。**

## 事实（两条独立证据）

**① 权重文件本身。** 把两份疑似不同的文件哈希比对：

```
tonylica bundle  knee-mri-fold-weights/m_f0.pt   sha256 b447a665a92ed65e...
tannguyenngoc    dinov3/m_f0.pt                  sha256 b447a665a92ed65e...
                            ⇒ 逐字节相同
    cfg.backbone = 'vit_small_patch16_dinov3.lvd1689m'   img=336  fold=0  180 tensors
```

**`mattiaangeli/knee-mri-fold-weights` —— 我们栈里 `a5_w=0.52` 用的那个成员 —— 骨架就是 DINOv3。**

**② 我们的加载代码。** `kernel-0943cf` 里 A5 那段：

```python
enc = timm.create_model(cfg['backbone'], pretrained=False, num_classes=0, in_chans=_in,
                        **({'img_size': cfg['img']} if 'vit_' in cfg['backbone'] else {}))
...
rsna_strict_load(m, z['state_dict'], f'A5 fold {fold}')
```

**它用的是 checkpoint 自己的 `cfg['backbone']`** —— 运行时读到 `vit_small_patch16_dinov3.lvd1689m`，
**所以我们本来就在构造 DINOv3 骨架**，并且 `strict` 载入。

⚠️ **为什么我第一遍没看见**：我在 notebook 里 grep `dinov3` 命中 **0** ——
**因为那个名字从不以字面量出现在代码里，它是运行时从 checkpoint 的 cfg 读出来的。**
又一次"字面量不在代码里，因为它藏在数据里"。**这跟"过期日志字符串"是同一类错误的镜像。**

## 所以真实结构是（两边一致）

| | 公开配方（`tonylica` README 逐字） | 我们 |
|---|---|---|
| transformer 段 | **20 DINOv2 percentile-rank votes → five DINOv3 fold-rank votes → 55/45 blend** | 同样的 20 个 DINOv2 前沿成员 + 同样的 5 折 DINOv3（`a5_w`） |
| `a5_w` | 0.941 谱系 **0.45** | 0.943 谱系 **0.52**（注释即写 "0.941 lineage used 0.45"）|
| Raptor | v5/v10/v5-reverse/v8 @ 60/10/10/20 | 同 + 我们自训的 `cn0`/`msft2` |
| 外层 | `rank((1-w)·transformer + w·hybrid)`，逐标签 w | 同（`coatnet_w`，默认 0.65）|
| 混合 | `H = rank(0.60·rank(raptor) + 0.40·rank(residual_coat))` | 同 |

**⇒ 我们不在公开配方之下，我们就在它上面（还多了两个自训视角）。"换 DINOv3 底座"这个实验不成立。**

## ⭐ 但这次动手捞到两样真东西

### 1. 一个被记为"来源不明"的产物，现在有出处了

`tonylica/rsna-knee-bend-dinov3-0917-repro-assets`（4 GB，**6830 次下载**）里有一份 `calibration.json`，
键是 **`['mean','scale','coef','intercept','gate','protocol_columns','groups']`** ——
而本仓关于我们 `_RAD_CAL` 的记录是：*"`_RAD_CAL['gate']` = [ACL, Medial OA, Lateral OA, PF OA,
Effusion, Baker's, Contusion]，一个**来源不明**的嵌入式 base64/zlib 产物"*。

**⇒ 那个"来源不明"的标定表，就是公开配方的冻结 12 输出线性标定。**
本仓写的 *"Do not touch it"* 结论**不变，而且理由从"不知道是什么"升级成"知道是前沿冻结的配方件"**。

### 2. 一份 **SHA-256 验证过的公开管线参考实现**

该包把自己描述为 *"a complete checkpoint **inference** reproduction"*：
41 个检查点成员（20 DINOv2-small + 5 DINOv3-small + 5 reference Rad heads + 5 E13 Rad heads +
3 public Raptor CoAtNet + 3 residual-gated CoAtNet epoch snapshots）+ 一个冻结标定表 +
可读的 `pipeline/`（`blend.py` / `common.py` / `public_raptor.py` / `residual_coat.py` / `transformer_rad.py`）。
**源 LB 已验证 0.941**（`56173926`，2026-09-11），作者自己写 *"0.942 is a target"*。

**用途不是提分，是对拍**：它给了我们一个**独立的、逐文件 SHA 校验的同一配方实现**，
可以用来回答"我们栈里某一段是否和前沿一致"。**这是本仓第一次有这种东西。**

### 3. 两条公开配方的权重其实不完全一样（值得记）

| | 我们的 / `tonylica`（0.941） | `tannguyenngoc`（gold58 0.962） |
|---|---|---|
| Raptor 权重 | v5 **.60** · v10 .10 · v5-rev .10 · v8 .20 | v5 **.50** · v10 **.30** · v8 .20（**无 reverse**）|
| CoAt 家族 | resgated / d4 / global96 三成员 | **没有这一层** |
| 归约 | `rank_of_member_probability_mean` | `probability_then_rank` |

⇒ 两份"公开答案"在**同一份权重**上给出**不同的组合**，且都落在 0.941–0.943。
**再一次说明：这个区间内的组合差异在公榜上看不见**（也正是名次仪器存在的理由）。


