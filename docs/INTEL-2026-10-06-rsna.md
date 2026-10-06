# RSNA Knee Abnormality Detection —— 情报汇总 2026-10-06

来源全部可复核：实时公榜 API、`kernels_list`、以及**逐字拉取的三本公开 notebook 的正文**。
每一条都标明它来自"实测"还是"作者自述"。

---

## 1. 目标：一个显示档位 = 铜牌线

原始公榜读出（`work/target_score.py`，2026-10-06 08:18Z）：

| 显示分 | 队数 | rank 区间 | 相对铜牌线（≤529） |
|---|---|---|---|
| 0.946 | 50 | 206 .. 255 | 内 |
| 0.945 | 63 | 256 .. 318 | 内 |
| **0.944** | **186** | **319 .. 504** | **整段在内** ✅ |
| **0.943** | **990** | **505 .. 1494** | **横跨**（我们在里面） |
| 0.942 | 191 | 1495 .. 1685 | 全在外 |

⇒ **`0.943 → 0.944` 这一个档位就是"进铜牌区"和"不进"的全部差别。** 0.943 段宽 990 名，线只切掉它最前面 24 名。
**5 倍队数落差说明 0.944 是真门槛，不是分布尾巴。**

**实时（14:06Z）**：teams **5310**，我们 rank **705**、`0.943`，铜线 **531**，**margin −174**，格内 **15.86%**，
`0.943` 块 n=**972**、ranks **551..1522**。榜首 **0.964**。

⚠️ **漂移**：rank 645(06:11) → 670(08:50) → 683(10:23) → **705(14:06)**。约 **+5 名/小时 ≈ 120 名/天**。

---

## 2. ⭐⭐ 配方被作者自己完整公开了——含三个我们此前不知道的数

来源：`intoxicatinglips/rsna-knee-pure-0944-verbatim`（2026-10-06 13:17 运行）**正文逐字**。

### 2.1 三个公榜数

| 组件 | 公榜 macro AUC |
|---|---|
| 社区栈（**70%**） | **0.943** |
| **reader 单独（3 折）** | **0.929** |
| **rank 混合（70/30）** | **0.944**（+0.001 over the stack） |

⭐ **reader 单独的 0.929 高于 A167 成员门（solo ≥ ~0.90）** ⇒ 它是**合格的成员**，不只是"多样性"。
⭐ 它同时解释了 `C16/D14` 那条"视角/架构多样性 ≈ +0.001"——**这条腿贡献的正是 +0.001**。

### 2.2 权重平台（作者原话）

> **Reader weight:** `OWN_W` in the last cell. **15% and 30% both scored 0.944; 45% scored 0.942.**

⇒ 与我们记录的一致。**15% 与 30% 是同平台**，45% 掉档。

### 2.3 ⭐⭐⭐ 运行时间：**6.5–8 小时**（作者原话）

> `RUN_STACK = True` (default): community stack + our reader, the 0.944 blend. Scoring the hidden test set
> takes **6.5 to 8 hours** of the 9 allowed. **It is not stuck; let it run.**
>
> `RUN_STACK = False`: our reader only. Scored in **about 20 minutes**, public LB 0.929.

⚠️⚠️ **这条更正了我自己记录里的一个错误，而且纠正的方向是"我们错怪了系统"**：
我此前把 **"<30 分钟"** 写进过 `STATUS.md` 与 `FACTS` 的口径 —— **那个 20 分钟是 `RUN_STACK=False`（只跑 reader）的时间**，
**完整栈 + reader 是 6.5–8 小时**。我把"只跑 reader"的时长当成了整条流水线。

⇒ **三条在途提交各自都需要 6.5–8 h**，预期出分：

| 提交 | 提交时刻 | 预期出分 |
|---|---|---|
| `56872230`（w=0.30） | 06:11Z | **≈14:11Z** |
| `56876044`（TTA） | 08:50Z | **≈16:50Z** |
| `56878092`（w=0.15） | 10:23Z | **≈18:23Z** |

⇒ **长时间 PENDING 是正常的，不是卡住**；服务端内核的 9 h 上限分别到 15:11 / 17:50 / 19:23Z。

---

## 3. ⚠️ 一批**硬断言**：环境不对会**硬失败**，不是静默降级

同一本正文里的代码（逐字）：

```
raise RuntimeError('D4 requires exactly two T4 GPUs')
if not all(('T4' in name for name in names)): ...
if any('T4' not in torch.cuda.get_device_name(d) for d in DEVS):
    raise RuntimeError('Select the reference T4 x2 accelerator')
```

而且正文写明：

> **Runtime contract and artifact checks** … pins checkpoint hashes, dependency versions, memory limits,
> and the D4 serving contract before expensive inference begins. **Any missing or changed required asset
> stops the run instead of silently changing predictions.**

另有**大量 SHA256 钉死**（`adapter_sha256`、`coat_resgated_ep10_top3_manifest.json`、`timm_wheel` …）。

⇒ **两条操作含义**：
1. **它要求"恰好 2×T4"**。我们的复跑拿到的正是 2×T4（已实测），所以**目前安全**；
   但**换任何别的卡都会硬失败**，不会降级。这解释了为什么这些臂是 fail-CLOSED（`FACTS` A73）。
2. **任何一个被钉死的资产变了 ⇒ 整个运行停掉**。⇒ **别人的资产更新=我们某天突然跑不起来**，这是系统性脆性。

---

## 4. ⭐⭐ 公开面**已经饱和**：最新的三本"新"notebook 是**同一本的 fork**

我逐字拉了三本 2026-10-06 的新内核：

| notebook | 运行时刻 | 正文 |
|---|---|---|
| `intoxicatinglips/rsna-knee-pure-0944-verbatim` | 13:17 | 与下同 |
| `adityakumarrai0401/rsna-knee-b03-convnext-reader-blend` | 12:57 | 与下同 |
| `sujanmajhisuzan/rsna-knee-tri-specialist-superstack` | 11:56 | 与下同 |

**三本的 markdown 首段与分数声明逐字相同**，都是：

> The community's 0.943 inference stack, run unchanged, plus one new model family… a 2.5D ConvNeXt-Tiny
> reader with a small study-level transformer… **That blend scored 0.944 on the public leaderboard.**

差别只在挂载列表（`tri` 多挂了 `goodpjw2008/rsna-knee-2-5d-convnext-reader`、`mattiaangeli/rsna-knee-coat-resgated-ep10-top3`）。

⇒ **公开面没有新东西，只有同一份 0.944 配方在被反复 fork。** 这与 `SWEEP-2026-10-06d-public-surface-converged.md`
的结论一致，而**这次的证据更强**：不是"我没找到新的"，而是"**新出现的三本，正文相同**"。

⇒ **结论（写死）：不要再期望从公开面拿到 >0.944 的东西。** 0.944 就是公开天花板。

---

## 5. 我们自己的位置

* 我们的 reader 臂 = **同一条配方，但基座是我们的变体栈 `0943cfmsft2`**（作者是在社区栈上扫的权重）。
* 在途三条：`w=0.30`、`w=0.30 + TTA`、`w=0.15`。
* 今天额度：**3 / 5**（还可再打 2 枪）。
* **GPU 剩 24.03h / 30、TPU 剩 11.97h / 20，10-10 刷新**（⚠️ 与 ARC 共用）。

---

## 6. 由这批情报**新产生**的可行动作（按价值）

1. ⭐⭐ **`RUN_STACK=False` 把"改 reader"的迭代周期从 8 小时压到 20 分钟（约 24×）。**
   作者的 `RUN_STACK` 开关是现成的：只跑 reader ⇒ ~20 min ⇒ 公榜 0.929。
   ⇒ **任何"改进 reader"的实验（换权重、换融合、加 TTA）现在都有一个 20 分钟的读数**，
   而不是 8 小时。⚠️ 但要注意：**reader-only 的分（0.929）本身低于我们的 0.943**，
   所以这条读数**只能用来比较 reader 之间的相对优劣**，不能直接当作提分。
   （且 RSNA 已有 3 次在途，提交额度 5/天，够用。）

2. ⭐ **把 `RUN_STACK=False` 当成"我们的 reader 与作者 reader 的对照"**：
   我们跑同一模式，若明显低于 0.929，说明**我们的基座/预处理与作者不同**，这个差异会解释
   为什么我们的混合可能落在 0.943 而不是 0.944。

3. **对"硬断言"做一次 preflight**：确认我们的臂里**没有**类似的卡型断言
   （否则某天 Kaggle 换环境就会静默把整条臂打死）。`work/preflight_0943.py` 已有部分断言。

4. **不要重开**：五折 `msft2`、`OWN_W` 扫描、专科、外部数据、Kaggle Models、视角多样性、
   我们自己的 ConvNeXt、蒸馏检查点（实测 0.8895）。证据见
   `docs/DECLINE-2026-10-06h-three-levers-rechecked.md` 与 `docs/RESULT-2026-10-06p-distilled-gate-rejected.md`。

---

## 7. 一句话总结

**公开天花板 = 0.944 = 铜牌线；而 0.944 的配方已经被作者完整公开、并且正在被全社区复制。**
我们手里**已经是那条配方**（三条变体在途，14:11–18:23Z 出分）。
**真正的未知只剩一个：在我们的变体栈上，它落在 0.944 还是 0.943。**
