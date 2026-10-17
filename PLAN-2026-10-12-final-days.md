# 最后几天的长期计划（写于 2026-10-12T14:00Z，关机前）

> 配套文件：`STATUS-2026-10-12.md`（本轮处境与预登记读法）·`state/FACTS.md`（权威事实，本轮新增 C140–C146 / A187–A192）·`FINAL-TICK.md`（唯一的手工勾选页）。
> 本文件只管**接下来 10 天怎么排**。

---

## 0. 关机**不会**影响正在跑的东西（先说这条）

`kaggle kernels push` 是**把任务交给 Kaggle 云端跑**，不是在本机跑。所以：
- **`ppxl16/rsna-knee-raptor-s6-fork`（现在 QUEUED）会在云端继续排队并执行**，我们关机不影响它，~8–9 小时后出结果。
- 同理 ARC 的 `57106427` 是**服务端评分**，与我们的机器无关。
- 唯一会丢的是**本机的后台看护进程**（`work/watch_kernel.py`）—— 它只写日志，丢了不损失任何东西。

**⇒ 回来第一件事不是"重跑"，是"读结果"。**

---

## 1. 硬约束（这几条决定了所有计划的形状）

| 约束 | 值 | 后果 |
|---|---|---|
| RSNA 截止 | **2026-10-22 23:59Z**（剩 **10.4 天 / 250 h**） | 最后一次 GPU 长跑必须留出 ≥12 h 余量 ⇒ **10-21 之后不再开新长跑** |
| ARC 截止 | **2026-11-02 23:59Z**（剩 **21.4 天 / 514 h**） | RSNA 结束后有 11 天可以全部投给 ARC |
| RSNA 提交 | **5 次/UTC 日** | 计划里每天最多用 1 次 |
| ARC 提交 | **1 次/日** | **每天都该用掉**（评分是对 live 游戏集重放、逐次随机，平台保留最好 ⇒ 交一次是纯上行） |
| GPU 周额度 | **30 h**，RSNA 与 ARC **共享** | 一次 RSNA 长跑 ≈ 8.5 h ⇒ **一周最多两三次** |
| 并行 GPU 会话 | **最多 2** | RSNA 排队时，ARC 只能用**另一个**槽 |

**两条已经验证过的省额度技巧**（继续用）：
- **CPU 内核不占 GPU 槽**（A6）⇒ 所有"只读、只查、只算"的活都放 CPU 内核。
- **TPU 排队慢但便宜**（用户原话："TPU排队慢做不着急的事"）⇒ 不紧急的探索性训练走 TPU。

---

## 2. RSNA（10.4 天）：只有一条主线，加三条护栏

### 2.1 主线 = 等 fork 出分，然后按预登记读

**回来先跑**：
```powershell
cd D:\kaggle\rsna-knee
kaggle kernels status ppxl16/rsna-knee-raptor-s6-fork
kaggle kernels output ppxl16/rsna-knee-raptor-s6-fork -p scratch\runs\out_fork
```
看到 `COMPLETE` 就 **下载 `submission.csv` 并按 `work/submit_kernel.py` 提交**（或直接 `kaggle competitions submit -c rsna-knee-abnormality-detection -f submission.csv -m "..."`）。

**然后照 `STATUS-2026-10-12.md` 第 4 节的表读**，不要事后编解释：

| fork 结果 | 处置 |
|---|---|
| **≥0.955** | 🎉 **进银牌区**。把它换成 tick 的①号位（第二票保持非 OAI `56878092`），**RSNA 的 GPU 工作到此为止**，剩下 9 天只做每日盯盘。 |
| **=0.954 且档内位置明显优于 534/712** | 同理按新①号位处理（**以档内位置为准，不以显示分为准**）。 |
| **=0.954 且档内仍在 534 附近** | 两个轴买不到分 ⇒ 走 2.2 的机器形状那条腿。 |
| **ERROR / 无产物** | 记日志进 FACTS，直接走 2.2。 |

### 2.2 备用腿 = 更快的机器形状（只在需要时开，最多两次）

**假设**（来自 C144④）：notebook 有 `DEADLINE_HOURS = 8.3`，8.3 h 内没被 S6 打到的研究**回落到 Raptor 的排名**，而 Raptor 单件只有 **0.949** ⇒ **跑得越慢分越低**。0.956 那次是 T4。

**动作**：复制 `kernels/kernel-raptor-s6-fork/`，**只改 `machine_shape`**（例如 `NvidiaRtxPro6000`，F49 已实测该字段在本账号生效），其余逐字不动 —— 仍然是单变量。

**⚠️ 先决条件**：那条形状上的 torch 必须支持对应架构（我们在 ARC 上吃过 sm_120 的亏）⇒ **开跑前先花 10 分钟用一个 CPU 内核或短跑内核探一次** `torch.cuda.get_device_name()`，不要直接押 8.5 h。

### 2.3 护栏（三条，任何时候都成立）

1. **tick 不许乱动**：`57059293`(0.954, OAI) + `56878092`(0.944, 非 OAI)。**永远不要两票都投 OAI**（`work/verify_final_pair.py` 会拦）。
2. **每天最多 1 次 RSNA 提交**，且只为**一个真候选**。5 次是上限不是配额。
3. **10-21 之后不再开任何 RSNA 长跑** —— 跑不完的提交等于零信息。

### 2.4 每天 15 分钟的盯盘（零 GPU）

```powershell
python work\lb_snapshot.py          # 刷公榜 -> state/lb_snapshot_*.json
python work\bronze_forecast.py      # 已修好：自己下载 + 与 JSON 对账
python work\nbscore_census.py       # 公开面真实分普查（几分钟，零 GPU）—— 看有没有新的 >0.954
```
`nbscore_census.py` 是**当前性价比最高的侦察动作**：它上次直接推翻了"公开面已探尽"。**只要它报出新的 >0.954 且无空 slug，就值得再跑一枪。**

---

## 3. ARC（21.4 天）：这是主攻，但 RSNA 结束前只能用"业余时间"

### 3.1 现状（实测）

| | |
|---|---|
| 我们最好 | **24.53**（`56883490`，是 `dfranzen/arc-agi-3-milestone-2-solution` 的副本 + W4A16 int4） |
| 铜牌线 | **top 10% = 29.92** ⇒ **差 +5.4** |
| 队伍 | 4139 · 榜首 62.96 |
| 待出分 | **`57106427` = PENDING**（高tier duck harness，NVFP4+W4A16+sglang）⇒ **回来第一件事就是读它** |

### 3.2 主线阶梯（**一次只动一个变量**）

1. **A 档（最纯）**：现有 stack + `dfranzen/albucino-qwen3-8-flash-next-drafter` 作为 `analyzer_model`。
   ⇒ **唯一变量 = harness 服务哪个模型**（F 系列已确认：ARC 生态只有 TAAF Duck 这一个方法，唯一杠杆是模型）。
2. **B 档**：NVFP4（`keithtyset/qwen3-8-flash-next-nvfp4` + 对应 runtime 数据集）。
3. **C 档**：FP8。
4. **D 档**：W4A16 on sm_120（注意：sm_120 需要专用镜像，我们在 `duck18-v*` 上已经走过一遍）。

### 3.3 每天的动作

- **用掉那 1 次提交**（纯上行）。
- 有 GPU 空槽时推**一个**新档的内核；`work/` 下的构建器**必须整份继承上游 metadata**（C139/A183：手抄环境字段必漏）。
- **F75 已否证 F58 的核心假设**（`pruned [] root …` ⇒ sgl-site 里根本没有 `numpy/`）⇒ **不要再沿着"修 numpy"那条线走**；`duck18-ve` 已证明这条链能跑通并产出 `submission.parquet`。

### 3.4 RSNA 之后的 11 天（10-23 → 11-02）

RSNA 一冻结，**全部 GPU 额度转给 ARC**，按 A→B→C→D 阶梯推进，目标先够 **29.92（铜牌）**。

---

## 4. 每次回来时的固定 checklist（照抄执行）

```powershell
cd D:\kaggle\rsna-knee
# 1) 云端的活干完了吗？
kaggle kernels status ppxl16/rsna-knee-raptor-s6-fork
kaggle kernels list --user ppxl16 --sort-by dateRun --page-size 12
# 2) 分数到账了吗？
python work\reconcile_submissions.py --append
# 3) 我们到底在哪？（权威 = 原始 CSV，不是派生工具）
python work\lb_snapshot.py ; python work\bronze_forecast.py
# 4) 公开面有新东西吗？
python work\nbscore_census.py
# 5) 配对还合法吗？
python work\verify_final_pair.py
```

ARC 那边：
```powershell
kaggle competitions submissions -c arc-prize-2026-arc-agi-3 --csv   # 看 57106427 出分没
```

---

## 5. 绝不做的事（都是踩过的坑）

| 不做 | 因为 |
|---|---|
| 两票都投 OAI 血统 | 一次裁定可以同时作废两票；平台只保最好的一个 |
| 相信**标题/名字**里的分数或模型名 | `r6cnxfsss` 与 `compact-v1` **字节相同**（C145）；`velci-kcd25-20` 的 0.955 靠**空 slug 私有数据集**（C143） |
| 相信**派生工具**打印的名次 | 它把"档内序号"当全局名次，报过 `+265 INSIDE`（真值 `−151`，C140/A190） |
| 在公开面**还存在更好资产**时就去自训模型 | 差距是**检查点质量**（C108），自训臂 solo ~0.87，补不上 |
| 10-21 之后开新长跑 | 跑不完 = 零信息 |
| 用一次提交去"试"一个公开声称 | 先跑 `nbscore_census.py`（零成本，权威） |
| 事后为结果编解释 | 预登记读法写在 `STATUS-2026-10-12.md` 第 4 节 |

---

## 6. 需要**人**做的唯一一件事

**在竞赛页 My Submissions 勾选 2 条提交**（这一步脚本做不了）。按 `FINAL-TICK.md`：
- ① `57059293`（描述含 `heliosli's restored blend-gold`）
- ② `56878092`（描述含 `w=0.15`，**看不到就不要勾**）

⚠️ **如果 fork 出了 ≥0.955，就把 ① 换成它**（到时先改 `state/final_pair.json` 的 `tick`，再重跑 `verify_final_pair.py` 确认，**不要直接改 `FINAL-TICK.md` 的字符串**）。

---

## 7. 时间线一览

| 日期 | RSNA | ARC |
|---|---|---|
| 10-12（今天） | fork 已在云端排队，关机不影响 | `57106427` 待出分 |
| 10-13 ~ 10-20 | 读 fork 分 → 决策树；每天盯盘 + 普查；**最多再开 1 次长跑**（机器形状那条腿） | 每天 1 次提交；有空槽就推 A 档内核 |
| 10-21 | **冻结**：不跑新东西，只确认产量与配对 | 开始把 GPU 转过来 |
| 10-22 | 截止 23:59Z。**勾选两条** | — |
| 10-23 ~ 11-02 | 结束 | **全力**：A→B→C→D 阶梯，目标 29.92 |
