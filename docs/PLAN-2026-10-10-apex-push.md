# PLAN 2026-10-10 — 推 0.951（blend-gold）/ 0.950（Apex）拿铜牌

> 立于此：2026-10-09T14:2xZ。依据 FACTS **C111 / C113 / C114 / C116 / C117 / A179 / A180**。
> 用户方针不变：**不开新实验**、**先确定能提分再提交**、**别过拟合公榜**（"你自己决定" ⇒ 本文件即决定）。

## 1. 为什么是这件事

| 事实 | 数字 | 出处 |
|---|---|---|
| 我们的公榜最好 | **0.944**，rank ≈1176 | C111 |
| 铜线（top 10% of 5583） | rank **558** ⇒ **0.950** | C111 |
| **0.950 够不够** | **只约五成**（并列按未舍入真分排，档内 rank 254..711） | **C113** |
| **真正的目标** | **0.951**（ranks ≤253，稳） | **C113** |
| 我们自己的改进路径 | **全部关闭或负结果**（C90/C95/C92/C96/C97/A175/C102/**C109**） | FACTS |
| 非 OAI 公开路线的天花板 | **0.944**（`sujanmajhisuzan/rsna-knee-tri-specialist-superstack`，106 票，与我们同分） | C117 调查 |

⇒ **唯一的铜牌路 = 换那个公开检查点 + 一条第二腿。而第二腿选谁，决定 0.950 还是 0.951。**

## 2. 三个资产（都已建好，都已过 GBK / JSON / 往返体检）

| 优先级 | 目录 | kernel id | 目标 | 性质 |
|---|---|---|---|---|
| ❌ **关闭** | `kernels/kernel-blendgold-0951` | `ppxl16/rsna-knee-blendgold-0951` | ~~0.951~~ | **不可复现**：第二腿需要私有的 `rsna-knee-compact-v1`（含 `manifest.json` 与 4 折 bundle），推上去当场 `ERROR — RuntimeError: Cannot resolve rsna-knee-compact-v1: []`。见 C119。 |
| **★ 主攻** | `kernels/kernel-apex-repro` | `ppxl16/rsna-knee-apex-0949` | 0.949~0.950 | Matterhorn/Apex 路线（Work 6）：CoAtNet + **goodpjw ConvNeXt MIL**，**12 个 per-target 权重在公榜上调过**。**已推 v1 → COMPLETE → 已提交 ref `57026328`（2026-10-10T00:58Z，出分 10–11 h）。** |
| ⏸ 备用 | `kernels/kernel-apex-pure0949` | `ppxl16/rsna-knee-apex-pure0949` | 0.949 | 只跑检查点、不做融合。被 apex 支配（分更低），**未推**（并发上限 2）。 |

> ⚠️ **状态更新 2026-10-10T01:0xZ（C119 / A181 / A182）**：**blend-gold 关闭**，我们能复现的上限就是 **0.950**；apex 已提交等分。
> 曾选 blend-gold 的理由在**方法论上仍然成立**（权重开跑前定死、零拟合，比 Apex 那 12 个公榜拟合权重更抗私榜回吐）—— **只是工程上够不到**。
> ⭐ **C119 的教训**：先验固定的权重是好设计，但**"可否复现"必须先用文件探测证伪，不能靠标题或票数**。上游 metadata 里那个空的数据集条目**不是垃圾，是我们看不见的私有数据集**。

**为什么不是"分更高的那条"**：0.951 与 0.950 的差别**不是靠加旋钮换来的** —— heliosli 把第二腿换成**自己独立训练的模型**，而权重是**先验固定的**。这正是"提升分数"与"过拟合公榜"的分界线；可惜那条路被私有权重锁住了。

## 3. 执行步骤（GPU 额度一解锁就跑）

```powershell
cd D:\kaggle\rsna-knee
$env:PYTHONIOENCODING='utf-8'
$env:KAGGLE_API_TOKEN = [Environment]::GetEnvironmentVariable('KAGGLE_API_TOKEN','User')

# ① 主攻
python work\push_kernel.py kernels/kernel-blendgold-0951
# ② 备用
python work\push_kernel.py kernels/kernel-apex-repro

# ③ 必须等到 COMPLETE（code competition 要真跑一遍才有分）
kaggle kernels status ppxl16/rsna-knee-blendgold-0951
kaggle kernels status ppxl16/rsna-knee-apex-0949

# ④ 提交（-k / -v 是 code competition 的路径）
kaggle competitions submit -c rsna-knee-abnormality-detection `
  -k ppxl16/rsna-knee-blendgold-0951 -v <VERSION> -f submission.csv -m "blendgold 0951"
```

⚠️ **硬约束**
1. **`kaggle kernels push` 失败时也返回 0** ⇒ 只认 `Kernel version N successfully pushed`（守卫已内置，A179）。
2. **`machine_shape` 覆盖 `enable_gpu`**（A168 陷阱③）⇒ 三个 metadata 都显式写 `NvidiaTeslaT4`。
3. **`docker_image` 是配方的一部分** ⇒ 已与上游逐字符一致（`…2757e0c7…`，qualification 记录 torch 2.11.0+cu128 / timm 1.0.29）。
4. **别把传输故障当成绩**（A173/A174）。
5. **blend-gold 自己有两层"失败就响"**：Raptor 阶段 `rc!=0` 直接 raise；>3% Raptor 失败或 Spearman<0.99 也 raise ⇒ **如果它没产出 `submission.csv`，那是真失败，不要拿半成品交。**

## 4. 判定与最终两票

- 分数落进 `state/rsna_submissions.jsonl`，并跑 `python work\rsna_daily.py` 更新 pair check。
- **候选池**（平台 keep-best ⇒ 配对的唯一职责是**私榜不要同时失败**）：
  - `56878092` **0.944** —— 我们自己的 CoAtNet/44 槽底座（**无 OAI**）
  - `56570151` **0.943** —— `0943s` 稀疏底座（**不同 substrate**）
  - 新增：blend-gold（**OAI 系**）± Apex（**OAI 系**）
- **判据**：
  - 裁定 OAI 权重**可用** ⇒ **blend-gold + `56878092`**（一高一稳，substrate 不同）。
  - 裁定**不可用** ⇒ 保持原对 `56878092 + 56570151`。
  - **不确定 ⇒ 绝不给两票都投 OAI 系**（那正是"同时失败"）。

## 5. 风险登记

| 风险 | 影响 | 处置 |
|---|---|---|
| 主办方裁定 OAI 训练权重违规 | 新提交作废 | 保留原对；不两票都押 OAI |
| blend-gold 的 0.951 是"公榜改善 ≠ 私榜改善" | 私榜回吐 | 其**权重零拟合**已把这项压到最小；且作者自己就写明了这句警告 |
| 0.950/0.951 恰落在档内下沿 | 够不到 rank 558 | C113：以 **0.951** 为目标，不再以 0.950 为目标 |
| GPU 周额度未按时刷新 | 推不上去 | 一解锁立刻重试；**TPU 无解**（两条路线都硬要求 CUDA） |
| 落后于赛场收敛速度 | 每天掉约 150 名 | 越早交出分越好，别等"更好的配方" |

## 6. 明确不做的事

- ❌ **不追 `...grandmaster-stack-v5`**：标题是 *"Target: 0.954 ~ 0.955+"*，**是目标不是成绩**，且加的是更多公榜旋钮（Dual-Res 路由 / Flip-TTA / 动态校准）。
- ❌ **不加第三腿**（把我们的 `0943cfmsft2` 也融进去）：只有在 blend-gold 的实测**够不到 0.951** 时才重新考虑。
- ❌ **不训练任何新模型**（用户明令；且 C109 已证"训更久"无余量）。
