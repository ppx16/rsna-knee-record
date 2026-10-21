# RSNA Knee Abnormality Detection — STATUS

> # ⛔ 本页已于 2026-10-21 作废（内容停在 2026-10-07 的 0.943 时代）
>
> **文件名不带日期，所以它看起来像"现役状态页" —— 它不是。** 现在的入口是：
>
> | 想看什么 | 看哪里 |
> |---|---|
> | 这次比赛的最终结果与方法 | [`README.md`](README.md) |
> | 每一条结论（含被证伪的） | [`state/FACTS.md`](state/FACTS.md) |
> | 截止日要做什么 | [`docs/DEADLINE-CHECKLIST.md`](docs/DEADLINE-CHECKLIST.md)、[`docs/FINAL-TICK.md`](docs/FINAL-TICK.md) |
> | 最近三天 | `docs/HANDOVER-2026-10-2*.md` |
>
> ⚠️ **尤其不要按本页里出现的任何 `ref` 去勾选**：那些是 0.943 时代的提交号，与当前的 tick
> （`57250245` + `56878092`，见 `state/final_pair.json`）无关。本页以下内容仅作历史保留。
>
> （这条横幅是 `work/audit_docs_tick.py` 的产物：它扫描所有文档里"tick/勾选 + ref"的组合，
> 找出与 `final_pair.json` 不一致的**指令式**行。它已经抓到过一条真的危险指令 ——
> `FINAL-TICK.md` 里那句"退回 0.954"的陈旧兜底。）

## 🟢 CURRENT STATE — 2026-10-07T03:05Z — **read this block and nothing else for the state**

*Every number below was assembled from files (`work/assemble_state.py`), not from prose. The DELTA sections
further down are a chronological audit trail and several are superseded; they are kept because this
repository has been damaged before by a STATUS whose prose had drifted.*

| | |
|---|---|
| **deadline** | **2026-10-22 23:59Z — 16.7 days** |
| **board** (`2026-10-07T03:0xZ`, `work/lb_snapshot.py`) | teams **5358**, rank **593**, **`0.944`**, bronze line **535**, **margin −58**, top **0.964** — ⚠️ **WE ARE ERODING, NOT HOLDING**: rank 545→588→593 and margin **−13 → −53 → −58** across 17:26Z→01:12Z→03:05Z. The line moved 3 places while we moved 48, so this is new submissions passing us, not the line shifting. Public **notebook** material tops out at exactly 0.944, so **0.944 cannot hold bronze** ⚠️ 口径见下方 IN FLIGHT 区（**榜单**最高 0.964、且 ≥0.946 的队伍有一百多个 —— 比我们高的用的是未公开方法）。 Re-take with `work/lb_snapshot.py`. |
| **quota** (`kaggle quota`, venv CLI — columns are `used / remaining / total`) | **GPU 5.84 h used, 24.16 h remaining** / 30 · **TPU 7.34 h used, 12.66 h remaining** / 20 · both refresh **2026-10-10** |
| **submissions today** | **0 / 5** (fresh day, 2026-10-07) |

> ### 🔄 IN FLIGHT — 2026-10-07 — 微调臂/配方两方向的审计与两个内核
>
> **审计把两条方向的判断都改写了**（全文 `docs/RESULT-2026-10-07-two-directions-audit.md`，事实进 FACTS **A170/A171/C87/C88/D20/D21/D22**）：
>
> | 方向 | 真实状态 |
> |---|---|
> | **配方 · v9 当第 5 个视角** | ⛔ **关闭**。D14：两次独立尝试（v4 widedense、**v9 finespacing**）**都是 0.941 平**。唯一没跑的 0.50 剂量测试建好未推，**判定不推**（D21）。 |
> | **配方 · reader 腿权重** | w=0.15 是唯一越过铜线的（→ **0.944**）；w=0.30 / TTA / community-stack 全 0.943。**0.944 就是公开面天花板。** |
> | **微调臂 · v5 warm-start** | ✅ **已做已收获**：`msft2` = 从 maxspan v5(0.9214) 的 backbone+head warm-start 微调，**solo 0.9069，唯一被公榜量到的增益**（C87）。 |
> | **微调臂 · v9 warm-start** | 🔄 **唯一仍开着的变体**，内核已建待推：`kernels/kernel-v9ft`（与 msft2 **只差 3 行** ⇒ 唯一变量是起始检查点）。 |
>
> ⚠️ **两条更正**：① A168 说的"已修"**只存在于 `kernel-msft2`**，`kernel-raptor-train*` **从来没有过**，本轮才补上（旧加载器对 v5/v9 两个公开检查点都解出 **0 个 backbone 张量**且静默通过 —— A170）。② 我写的判据「gold-58 上 Δ ≥ +0.0005」**作废**：gold-58 配对 CI 半宽 **±0.021**，门槛比仪器分辨率高 42 倍（**D20**，与 D16/C68/C78/C81 一致）。**臂间比较改用 OOF(±0.004)。**
>
> ### 📌 2026-10-07 收尾 —— 六条路线全部按实测关闭；今天的两个名额都已用掉
>
> | 路线 | 实测结论 | FACTS |
> |---|---|---|
> | 配方 · v9 当第 5 视角 | ⛔ 两次 0.941 平 | D14 / D21 |
> | 配方 · reader 腿权重 | 只有 **w=0.15** 越过铜线（0.944）；整条轴只有 0.002 宽 | — |
> | 微调臂 · v5/v8/v9 warm-start | **v5（msft2）是唯一成功的**；**v8 0.8899 / v9 0.8907 双双被守卫拒** | C91 / C92 / C96 |
> | **几何匹配** | ⛔ **无可检出好处**（匹配的 v8 **低于**不匹配的 v5） | **C95** |
> | 切片密度（44/64/80） | ⛔ solo +0.0065 → **成员 +0.0007，CI 跨 0** | **C90** |
> | 逐 finding 选臂 | ⛔ **事后上限只有 +0.0044**；`msft0` 已在 12 项里 8 项最好 | **C97** |
> | 3D 体积臂 | ⛔ 0.6493 / ~0.78 ≪ 0.90 门槛；且两个旧坑复发 | A175 |
>
> ⇒ **RSNA 已到公开面天花板（0.944）。我没有找到越过 0.945 的路，这一点不粉饰。**
> 剩下的差距是**模型质量**，而所有可及的新模型路线都已关闭。
>
> ### ⭐ 今天唯一期望为正的动作：最终配对改成【跨底座】
>
> `FINAL-TICK.md` 已更新，`work/verify_final_pair.py` 实测 **`pair present and COMPLETE`**：
> · **primary `56878092`**（0.944，reader 腿 w=0.15）—— 认 **`w=0.15`**，别碰 `56872230`（`30% rank leg`，0.943）
> · **second `56570151`**（**0.943**，`0943s` **稀疏 stack**，**另一支底座**）—— 从 `56613303`（0.941，同血统近邻）换过来
> ⇒ **分数更高 (+0.002) 且底座更远** = 纯改进。理由与边界见 `docs/STRATEGY-2026-10-07-final-pair-diversity.md`。
> ⚠️ private 榜**无法本地测量**，去相关是**结构判断**；但这次**分数也是顺向的**，所以即使收益为零也不亏。
>
> ### 今天用掉的名额
> · **ARC** `56909422`（PENDING）—— live 游戏复跑，零风险掷骰子
> · **RSNA** `56910273`（TTA + w=0.15，唯一没试过的配方组合）
> · **RSNA 还剩 4 个名额，我没有烧** —— 没有新东西可交，重交同一内核给同一个分。**不为"用掉名额"制造零期望动作。**
>
> ### 🔄 明天的第一件事：ARC 的 `duck-18-1gc-fix` **v5**
> boot 链已过四道关（zstandard → CUDA toolkit → 父进程 numpy → 三个 patch 阶段），
> **只剩子进程环境这一道**。v5 实施的是 F58 的**晚执行 `.pth`**（sitecustomize 里的插入被证明会被执行顺序覆盖）。
> 读数器 `work/watch_duckfix.py` 已挂，看两行：
> · `[arc_numpy_fix] pruned [...] root ...` ← `.pth` 跑了吗
> · `[sitecustomize] NUMPY_OK ...` ← 子进程 numpy 好了吗
> **若这两行正常，SGLang 就起来了** —— 那是这条重型 clone 第一次跑通，档位是 **40–55**（我们现在 24.53）。
> 台账 `arc-agi-3-ledger` 已到 **F58**（全部推送）。
>
> **一条命令读完全部**：`python work/collect_results.py`；RSNA 账本用 `python work/rsna_daily.py`。
>
> ### ⭐⭐ 为什么是 v8（FACTS **C93**，本轮最重要的结论）
> msft2（从 maxspan v5 warm-start 的 CoAtNet）是**唯一被量到越过 +0.005 的成员**：**成员 delta +0.0084，CI [+0.0060,+0.0107]**（C91）。而 A158 的模式说**只有 CoAtNet 族通过**。检查点按**几何匹配**排序（`work/rank_warmstart_ckpts.py`，从 .pt 直接读）：
> · v5 = **64 槽** ⇒ 在本语料读 **0.8963**（吃过不匹配的亏） · v9 = **80 槽** ⇒ **0.8907，被守卫拒**（掉 0.0124） · **v8 = 44 槽 ⇒ 与我们的语料【完全匹配】** ⭐
> ⇒ **预登记预测**：v8 的起始分应**高于** msft2 的 0.8963。
>
> ### ⚠️ 两条必须先读再判的纪律
> 1. **solo 分不决定任何事。** C90：同一个臂 **solo +0.0065（真实）→ 成员只剩 +0.0007**（CI [−0.0004,+0.0017] 跨 0）。⇒ 任何新臂训完，**唯一决定性的数是成员增量**：`python scripts/member_add_test.py --arm <tag>@f0 --strong`（**零 GPU**，从本地 OOF 直接算）。要打的标是 **msft2 的 +0.0084**。
> 2. **`scripts/member_add_test.py` 曾有盲区，已修**（C91）：它原来扫不到 `work/**`，**因此 A158 的"库存已榨干"漏掉了我们最好的臂 msft2**。可见臂数 22 → 31。
> 3. ⚠️ **预先记下的反证**：**v8 已经作为【视图】在栈里**（C15），v8 微调可能与它冗余（A158 的替代品陷阱）。**必须实测，不能假设。**
>
> ⚠️ **战略转向的依据（D22）**：0.944 已是公开天花板，而按 A167 的 11× 衰减律外推，v9 warm-start 的预期成员增益也只在 **+0.0002** 量级、**低于公榜分辨率**。⇒ "只在意公榜会对公榜过拟合、要做好本地 CV 增强泛化"从建议变成**主要依据**。

> ### 👉 THE ONE MANUAL ACTION: `FINAL-TICK.md` (repo root)
>
> The final pair is selected in a **browser** — there is no API for it (no `select` verb, no `selected`
> field, legacy `/api/v1` 401s; all three checked on kaggle 2.1.2 *and* 2.2.2), **Kaggle's auto-ticked
> default is NOT this pair**, and **thirteen other submissions also display 0.942–0.943**. `FINAL-TICK.md`
> gives the two description prefixes to match on, what to run before ticking, and the one condition (the
> reader leg reaching 0.944) that changes the table. Read it before touching the submission page.

> ### ⭐⭐⭐ 2026-10-06 17:26Z — **WE CROSSED TO 0.944.** Primary is now `56878092`.
>
> `56878092` (reader leg **w=0.15**) measured **0.944** and crossed the bronze step. The board moved with
> it: **rank 683 → 545**, **margin −154 → −13** (+141 places), teams 5326, line 532.
> Per the prereg the reader-variant line **stops** here. `56814081` drops out of the pair; the second slot
> is unchanged.
>
> 🚨 When ticking, the ONLY reliable distinguishing token is **`w=0.15`**: the new primary's
> description prefix CLASHES with `56872230` (both begin `0943cfmsft2 + the public 2.5D ConvNeXt reader as a
> separate leg...`). `56872230` says `30% rank leg` and scores 0.943. If you cannot see `w=0.15`, do not tick.
>
> ⚠️ **The position is thin**: margin **−13** at the public ceiling, and drift runs ~120–180
> ranks/day against us. Public material tops out at 0.944, so holding bronze needs something ABOVE it -- and
> the one unexplored axis is the 3D volume arm (FACTS A160).
>
> ⚠️ **2026-10-07 更正（FACTS C89）—— 上面那句 "Public material tops out at 0.944" 是两个不同口径混在一起，必须分开读：**
> · **公开 *notebook* 面** 确实顶在 **0.944**（本次重扫最新 40 条，无任何高于它的配方或机制；"0946" 那种标题是融合进公开 0.946 管线、不是它自己的分数）—— 这一半成立。
> · **榜单** 并没有顶在 0.944：2026-09-28 那份完整榜 **最高 0.959**，**≥0.946 有 139 队**、≥0.950 **75 队**、≥0.955 **24 队**、≥0.958 **6 队**。⇒ **榜上比我们高的队伍用的是未公开的方法，不是"公开面没有更高的"。**
> · 另：**"3D 体积臂是唯一未探索的轴" 已不再成立** —— 3D 两个起点都实测过，从零 **0.7077–0.7195**、`r3d_18` 预训练平台 **0.7808/0.7849**，**都远低于 A167 的 0.90 成员门槛**。

### ✅ FINAL PAIR — locked, and verified present + uniquely identifiable

| slot | ref | arm | public |
|---|---|---|---|
| **primary** | **`56878092`** | `0943cfmsft2` + reader leg **w=0.15** | **0.944** |
| **second** | **`56613303`** | `0943cflat` — `C + R`, **without** the public-fitted routing | 0.941 |

⚠️ **BROWSER-ONLY action.** No `select` verb, no `selected` field on `ApiSubmission`, legacy `/api/v1` 401s
(all three verified on kaggle 2.1.2 *and* the 2.2.2 wheel). **The auto-ticked default is NOT this pair**, and
**six other submissions also display 0.942–0.943** — pick by the description text, never by the score.
Verified `work/verify_final_pair.py`: both COMPLETE, description prefixes collide with nothing.

Rationale (`docs/REVISION-2026-10-06c-…`): the two candidates are **exactly equal on public numbers**, and
the competition's own **Dataset Distribution Notice** states prevalence *"is not guaranteed to be the same
across the training, public leaderboard, and final evaluation datasets"* — so the second slot should drop
dependence on the **public-fitted** per-label routing, which `cflat` does. **The switch is free.**

### 🎯 WHAT "BETTER" MEANS — quantified, so it is not re-litigated

* **One displayed step IS the bronze line.** From the raw board (`work/target_score.py`, re-run it — the line
  is a FRACTION of the field and moves): `0.944` = 186 teams at ranks **319–504, entirely inside**;
  `0.943` = 990 teams at ranks **505–1494, straddling it**, and we sit at 665. The only single step that
  reaches bronze is **+0.001**. `docs/TARGET-2026-10-06j-one-displayed-step-is-bronze.md`
* **The public ceiling is exactly 0.944.** Every newest public notebook that survives checking claims at most
  0.944 (`docs/FIND-2026-10-06k-public-ceiling-is-0944.md`). So **bronze is reachable and higher is not** on
  public material — the 63 teams from 0.945 up to 0.964 are doing something unpublished. **Do not go looking
  for another public lever**; three sweeps have already converged and this one bounds it.
* **Axes closed by measurement, do not reopen:** five-fold `msft2` (evidence, not budget),
  `OWN_W` sweep (author's own sweep shows a plateau), specialist/meniscus, external data, Kaggle Models,
  view-adding, our own ConvNeXt, routing/weights/calibration, and now **distilled checkpoints**
  (`kawaharataishi`'s scored **0.8895** on gold-58 under OUR geometry vs our `msft2`'s 0.9069 — geometry
  matched, it is just weaker; `docs/RESULT-2026-10-06p-distilled-gate-rejected.md`).
  Evidence for the earlier ones: `docs/DECLINE-2026-10-06h-three-levers-rechecked.md`.
  ⚠️ The distilled axis reopens only for a checkpoint **carrying its own CV** that scores **≥0.90** under
  our geometry. None is public today.

### ⏳ THREE READER ARMS IN FLIGHT — the gate has LANDED (rejected)

| what | id | status | how it will be read (fixed in advance) |
|---|---|---|---|
| **reader leg** ✅ LANDED | `ppxl16/rsna-knee-0943cfmsft2-rdr` → submission **`56872230`** | **COMPLETE — public score `0.943`** (15:0xZ). A TIE, not 0.944. The pre-registered branch is **pair unchanged**, and `instrument_blind` fired exactly as designed: the board carries our BEST submission, so a tie is invisible to the rank instrument. `delta_in_block_pct_raw` was +0.034 but the **drift-adjusted** delta is **0.0** (`drift_ranks_per_hour` 7.42) ⇒ verdict **no measurable change**. ⚠️ Without the drift adjustment the margin move (−118 → −179) would have read as this arm doing harm. See `docs/INTEL-2026-10-06-rsna.md` |
| **distilled zero-shot gate** ✅ LANDED | `ppxl16/rsna-knee-msft2-gate-distilled` | **COMPLETE** (12:06Z) | **gold-58 macro-AUC = 0.8895 ⇒ REJECTED as a member**, per the pre-registered 0.85–0.90 band. ⚠️ `head_loaded=8/8` and `head_tensors_carried=true`, so **the geometry MATCHES** — this is not a mismatch rejection, the checkpoint is simply weaker than our `msft2` (**0.9069** on the same gold-58) despite claiming `auc 0.9231`. `[zeroshot] VERDICT: FAIL` is NOT the criterion (it compares to the v5 parent bar 0.8963, the wrong reference). See `docs/RESULT-2026-10-06p-distilled-gate-rejected.md` |
| **TTA reader arm** ⭐ FIRED | `ppxl16/rsna-knee-0943cfmsft2-rdrtta` → submission **`56876044`** | **PENDING** (since 08:50Z) | ⚠️ **SCORE-based, because this is a THRESHOLD question, not a rank-increment one** (`docs/PREREG-2026-10-06n-tta-reader-arm.md`): **0.944 ⇒ crosses the bronze line (ranks 319–504, entirely inside) ⇒ becomes primary and STOP**; 0.943 ⇒ invisible to the rank instrument, pair unchanged; ≤0.942 ⇒ pair unchanged, reader-variant line ends. Expected **~18:50–19:50Z** |
| **reader leg w=0.15** ⭐ FIRED | `ppxl16/rsna-knee-0943cfmsft2-rdr-w15` → submission **`56878092`** | **PENDING** (since 10:23Z) | Same score-based rule (`docs/PREREG-2026-10-06o-reader-w015.md`). The author's sweep is **0.15→0.944, 0.30→0.944, 0.45→0.942**, so 0.15 is the other supported point of the same plateau; their sweep ran on the COMMUNITY stack while ours is the `0943cfmsft2` variant, so the better weight for us need not be 0.30. ⚠️ the two weights are **highly correlated**. Expected **~20:23–21:23Z** |
| **reader on COMMUNITY stack** ⭐ FIRED | `ppxl16/rsna-knee-0943-rdr` → submission **`56884048`** | **PENDING** (since 15:07Z) | **Same reader, same 70/30 weight** as `56872230`, which measured 0.943 on OUR variant stack — while the author publishes **0.944** for this configuration on the **community stack**. So the BASE is the variable this arm isolates. Verified composed before firing (`BLENDED 3 checkpoints at w=0.30`). Expected ~6.5–8 h ⇒ ~22:00Z. `docs/PREREG-2026-10-06r-reader-on-community-stack.md` |

Watchers `work/watch_rdr_submission.py` and `work/watch_distilled_gate.py` implement those rules verbatim
and write `state/reader_arm_result.json` / `state/gate_distilled_result.json`.

⚠️ **Those watchers are SESSION-SCOPED — if the session ends before a measurement lands, they die and their
result file is never written.** The results are still recoverable, and **one command does it**:

```
python work/collect_results.py
```

It prints, for both measurements, the raw evidence *and* the pre-registered verdict above; if the gate's
watcher file is absent it prints the exact fallback (fetch the kernel log and look for
`[zeroshot] zero training steps: GOLD macro-AUC <N>`). It never reports a verdict it could not earn —
transport failure prints `UNASKED`, a missing score prints `PENDING`, an unreadable log prints `UNREAD`.

### ⛔ Every other axis is CLOSED BY MEASUREMENT — do not reopen

Specialist (meniscus `56845973` = **0.940** vs a 0.943 baseline) · external / more data (the corpus **is** the
competition train set, id for id) · competition-specific Kaggle Models (best evaluable = **0.7992**, 0.10
below the gate) · Kaggle **Models** hub (rejected `yunus` head, closed) · view-adding (`FACTS` D14, two
independent attempts) · our own ConvNeXt (gold-58 0.9003 but **870-OOF 0.8493**) · routing/weights/calibration
(`FACTS` D1/D2/D3/D10–D13, all real measurements; `R` now measured too) · five-fold `msft2` (fold 1 = 0.9088
vs fold 0's 0.9069, and one fold costs **3.2–3.6 h**, so five folds = 16–18 h > the whole TPU budget) ·
`lixin73`'s DINOv2 checkpoint (**0.8634** on 874 studies = `cn0`, already in the stack) · "4-Way Consensus"
(**the 4th checkpoint has never been published** — 6 searches + full listings of all three mounted CoAt
datasets).

### ▶️ NEXT ACTION, in order

1. **Wait for the three reader arms.** Read them with `python work/collect_results.py` — one command, and it
   prints each arm's raw evidence, the drift-adjusted bar, and the branch that applies. Expected:
   `56872230` **~16:11–17:11Z** · `56876044` **~18:50–19:50Z** · `56878092` **~20:23–21:23Z**.
2. **If any reads 0.944** — it **crosses the bronze line** (ranks 319–504, entirely inside). Promote it to
   primary, keep `56613303` as second, and **stop spending slots**. Per `FINAL-TICK.md`, if several reach
   0.944 prefer the plainest (`56872230`); that is a coin-flip, not a decision worth agonising over.
3. **If all read 0.943** — that is **invisible to the rank instrument** (the board carries our best), the
   pair is unchanged, and **the reader-variant line ends here**. Do not re-fire variants on the strength of a
   tie.
4. **The distilled gate has LANDED and was REJECTED**: gold-58 macro-AUC **0.8895**, geometry matching
   (8/8 head tensors) but weaker than our `msft2` (0.9069). See
   `docs/RESULT-2026-10-06p-distilled-gate-rejected.md`. That axis is now closed.
5. **The manual tick is the last step and it is a browser action** — follow `FINAL-TICK.md`.

⚠️ **The public surface is converged and its ceiling is 0.944** — the same number as the bronze line. Three
sweeps plus a full notebook-by-notebook check found nothing above it that survives scrutiny. **Do not go
looking for another public lever**; the remaining work is to read the three arms and tick.

---

## ⭐ 2026-10-05 DELTA — the live state, and one suspicion checked and killed

### 1. Where we actually stand (snapshot 2026-10-05T07:59:16Z, persisted)

| | |
|---|---|
| teams on the board | **5192** |
| our rank / score | **518** / **0.943** |
| bronze line (top 10 %) | **519** |
| **margin** | **+1** — we are *one place* inside the bronze |
| `0.943` block | ranks **376..1389** (n=1014) |
| **in-block percentile** | **14.02 %** |
| top of board | **0.963** (2 teams), then 0.961, 0.960 |

Raw: `state/lb_snapshot_2026-10-05T075916Z.json`; appended to `state/rank_track.jsonl`.
Reproduce with `work/lb_snapshot.py` (it downloads the board, finds `ppxl16`, and persists both files).

**⚠️ The margin is not a buffer — it is a coin-flip.** The trajectory across 19 h:

| at | margin |
|---|---|
| 2026-10-04 12:44Z | **+23** (rank 487, line 510) |
| 2026-10-05 07:22Z | **+0** (rank 518, line 518) |
| 2026-10-05 07:59Z | **+1** (rank 518, line 519) |

⇒ **Without a further real gain we fall out of the bronze well before 2026-10-22.** ⚠️ The drift figure
quoted at the end of the 2026-10-04 DELTA ("≈ −2/day") was a **unit slip** and is corrected in place:
the true rate is ≈ **−40 ranks/day**. This does not change any conclusion, but it changes every
*deadline* estimate built on it.

### 2. One pending arm, and the rule it fires

`56845973` `0943cfmsft2m-men` (the author's public0033 DINov2-Base meniscus specialist at 90:10 on
Medial+Lateral Meniscus only) was submitted 07:22Z and is **still PENDING** — code reruns queue ~13 h.
`docs/PREREG-2026-10-05-meniscus-arm.md` fixes the reading in advance:

* in-block percentile **better than 14.07 %** ⇒ real gain, promote to final candidate;
* **flat or worse** ⇒ the "specialist" axis is dead too, and since the public-artifact, new-member and
  stacking axes are already falsified, **this round has no axis left and we should move to wrap-up.**

### 3. The Raptor crop was suspected mismatched, and it is NOT — checked, falsified

`CROP_MM = 130.0` is the first hit a grep finds, and `raptor-knee-native384`'s description says **140 mm**.
That would have been a silent mis-framing of every Raptor view. It is a **false alarm**: the notebook has
**per-block** preprocessing globals, and the Raptor block assigns `CROP_MM = 140.0` itself. No `ARMS`
entry carries a crop override, so all views correctly inherit 140 mm. Full text, plus the three dataset
descriptions that now independently pin our geometry: `docs/FIND-2026-10-05-raptor-geometry-confirmed.md`.

⇒ **Rule left behind: `CROP_MM`/`IMG`/`SLOTS` are not globals here — answer "what geometry do we use?"
per family, from the code path, never from a grep.** Same disease as `FIND-2026-10-04-stale-log-string.md`.

### 4. Evidence AGAINST the "add more backbones" intuition

`dreaddevelopment/raptor-knee-arms` = ConvNeXtV2-Base@336 + CoAtNet@384 + EfficientNetV2-L@480, same
2.5D window scheme and same head. Rank-mean blend = **0.915** public; **the CoAtNet arm alone = 0.914**.
⇒ the author added two architectures and an extra input resolution and bought **+0.001**, from a family
that sits ~0.028 *below* our current 0.943. **That axis is not the lever.**

### 5. ⚠️ A premise from this session that was WRONG: "v9 finespacing is an unbuilt lever"

`docs/FIND-2026-10-05-unused-strong-checkpoints.md` §2 read the **current** `0943*` `ARMS` table, found
no `finespacing-v9` entry, and concluded the gap was open and worth building. **The table observation is
correct; the conclusion was not.** `state/FACTS.md` records the prior:

* **`ref=56389731`** — `rsna-knee-ens-add-finespacing-arm` v1, submitted **2026-09-20**, ran at exactly
  80 slots / 78 windows / span 0.02–0.98 / img 336 / res 384 (the log confirms every field), and read
  **0.941 — identical to its own baseline** (**C49**, **C54**).
* **D14 closes the axis**: *"给 Raptor 臂加第 5 个视角 — 两次独立尝试、两次 0.941，关闭"* — the second was
  `add-widedense-arm` (v4 widedense), also geometry-verified.

⇒ **Re-adding v9 at the same ~0.20 dose would be a re-run of a measured-flat experiment.** The only never-run
variant is the dose test `add-finespacing-heavy` (v9 = 0.50), built and unpushed since **A32**. D14 says do
not spend on it; this file now agrees rather than treating it as the next step.
**Lesson, applied: before calling anything an "unfilled gap", grep `FACTS.md` for the artifact — not just
the current table for the string.** A gap in a table is not a gap in knowledge.

### 6. ⚠️⚠️ The GPU budget was mis-read by ~20×, and the error was in the DENOMINATOR

For this entire session I treated GPU as nearly spent and **used that to reject experiments**.
The figure I quoted was `19664 s / 21600 s` ⇒ "≈ 32 min left". Live, 2026-10-05:

| | used | remaining | total | refresh |
|---|---|---|---|---|
| **GPU** | 5.46 h | **24.54 h** | 30.00 h | 2026-10-10 |
| **TPU** | 3.79 h | 16.21 h | 20.00 h | 2026-10-10 |

```
19664 / 3600 = 5.46 h  == GPU used, live      ✓   (the numerator was right all along)
21600 / 3600 = 6.00 h  != GPU total, 30 h     ✗   per-SESSION cap, not the weekly allowance
```

⇒ remaining should have been `108000 − 19664 ≈ 24.54 h`, not 1936 s. **Same numerator, wrong total.**
⚠️ **This is why "refresh the page" would not have fixed it — a stale number and a wrong unit look
identical from outside.** It also reached `docs/PREREG-2026-10-04-convnext-member.md`, which framed the
ConvNeXt member as affordable **only as a fold-0 probe** when the full 5-fold (~13 h) was always in budget.
`docs/FIND-2026-10-05-quota-denominator-error.md`. Tool: `work/quota.py` (appends to `state/quota_log.tsv`,
which had been stale since 2026-09-26 — the log existed but nothing wrote to it).

**Rule: never record or quote an accelerator budget as a bare seconds count or single percentage. Read
`kaggle quota` at the moment of decision and always print `remaining / total` together** — the pairing is
what makes a unit error visible. ⚠️ Run it via `D:\kaggle\venv-kaggle\Scripts\kaggle.exe`; the PATH
`kaggle` failed 6/6 on the OAuth path while the venv one succeeded first try.

### 7. The "F_dim 1024 vs 768" blocker on `raptor-knee-arms` was a DEAD DEFAULT — resolved

`docs/FIND-2026-10-05-unused-strong-checkpoints.md` §1 warned that `coatnet384`/`cnv2b336` carry F_dim
**1024** while our v5/v8/v10 load through **768**, and that this had to be settled before wiring.

**Measured (`work/resolve_fdim.py`): our v5/v8/v10 are ALSO 1024.** Head tensors are identical across
every CoAtNet/ConvNeXtV2 checkpoint here — `att.0.weight (256,1024)`, `clsW (12,1024)`, `norm.weight (1024,)`
— and timm reports `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k num_features = 1024`. **Both sides were 1024.**

`768` occurs **once** in the notebook: `def __init__(self, backbone, F_dim=768, ...)` — a **default
parameter**. The only call site is `RaptorClassifier(bb, F_dim=bb.num_features)`. It is never reached, and
the load is `rsna_strict_load` (strict), so a real mismatch would have **raised**, not failed silently.

⇒ **The family is safely wireable.** What genuinely differs: `arch` must match per-arm (`load_model` raises
on arch drift, so `cnv2b336`/`effv2l480` need their own arch strings), and `effv2l480` really is **1280**
wide with an EfficientNet tail. `docs/FIND-2026-10-05-fdim-768-was-a-dead-default.md`.

⚠️ **This is the THIRD instance of one disease this session** — `CROP_MM` (per-block, not global), the
stale `[raptor-fast] ... cuda:0=[0,2]` log line, and now a dead default argument. Each wrong answer came
from reading **a declaration** instead of **the path that executes**.
**A grep finds where a value is written; it does not tell you whether that value is ever used.**

### 8. Built, NOT fired: a self-trained hard-label specialist (`msft3-spec`)

`docs/PREREG-2026-10-05-self-trained-specialist.md` + `scripts/build_msft3_specialist.py`
(`--check` passes for both tiers: anchors unique, compiles, 11 probes present). **Nothing pushed.**

**⛔ Hard entry gate: it may only be fired if the pending meniscus arm (`56845973`) reads a REAL gain**
(in-block percentile better than 14.07 %). If meniscus is flat, the label-specialist hypothesis is dead
and this is compute spent on no hypothesis.

It is not "a third same-corpus CoAtNet" (which the stop rule forbids) because it differs in three places
at once: the **loss is masked** to six labels, the **epoch selection is on that group's AUC**, and the
**blend target is that group only**. ⚠️ The second is not optional — `evaluate()` returns the all-label
macro and the loop selects on it, so a masked loss without group-aware selection would train one
objective and select on another, silently. The six labels are the ones the base's **own log** names
(*"Medial Meniscus / Medial OA / Lateral OA / Effusion / Synovitis / Contusion 是各臂真正不同…"*) — the
same sentence the meniscus prereg rests on, so the two experiments share one hypothesis.

**My stated expectation: probably flat-to-negative** (3 reasons in the prereg; the 11× decay was measured
on this corpus, and msft3-spec is still a fine-tune of it).

⚠️ **One deliberate deviation from A167, flagged:** A167's member gate (solo ≥ ~0.90) is an **all-12-label**
bar. A model trained on six labels cannot clear it, so applying it here would reject the experiment with
the wrong ruler. The gate is therefore computed **on the six-label group**. This is a real judgement call,
not an oversight.

Unverified and stated as such in the prereg: it has **never run a forward pass**; the rebuilt optimizer
(needed because the author's `AdamW([{bb},{head}])` raises on an empty group when the whole backbone is
frozen) is the likeliest place for a surprise.

### 9. ⭐ The final-selection sheet was stale by one measured result — primary changed

`docs/FINAL-SELECTION-2026-10-04.md` recommended **`56791368` (`0943cfm`) primary + `56613303` (`0943cflat`)
second**, and **never mentions `msft2` anywhere** (grep-verified). It predates the one result that matters.

⇒ **New sheet: `docs/FINAL-SELECTION-2026-10-05.md`.**

| slot | ref | arm | public |
|---|---|---|---|
| **primary** | **`56814081`** | `0943cfmsft2` — cn0 + fracture fix + **msft2 6th view** | 0.943 |
| **second** | **`56613303`** | `0943cflat` — cn0 + fracture fix + flat routing | 0.941 |

**The second slot does not change** — the 10-04 sheet's `0943cflat` argument (a strict superset of
`0943p`, and its 0.941 is not a defect) still holds and is **not** retracted. Only the primary moves.

**Why, in one line:** writing the four effects as `C` (cn0+fix), `S` (msft2), `M` (MCL), `R` (routing),
the pair values are `X=(cfmsft2,cfmsft2m)=C+S+max(0,M)`, `Y=(cfmsft2,cflat)=C+max(S,R)`,
`W=(cfm,cflat)=C+max(M,R)`. **`M` measured flat** (13.67 %→14.31 %) and **`S` measured positive**
(38.89 %→13.66 %), so **`Y ≥ X` and `Y ≥ W` always**. Both slots keep `C`; between them they hold the
one measured gain **and** the one documented public-fit hedge.

⚠️ **`0943p` (`56599159`) is dominated by `0943cflat`** (same `R`, plus `C`) and must never be ticked.
⚠️ **Still a browser-only action** — no `select` verb in the CLI, no `selected` field on `ApiSubmission`,
legacy `/api/v1` returns 401 (re-verified on 2.1.2 *and* the 2.2.2 wheel).
⚠️ **If `56845973` (meniscus) reads a real gain — in-block better than 14.07 % — this changes again.**

### 10. The rules were fetched, and one unstated assumption behind the whole selection is now verified

Every selection sheet said "two submissions may be selected" and described a tick UI — **none of them
established WHICH submissions are eligible to be ticked.** `56613303` (`0943cflat`, the recommended
second slot) was submitted **2026-09-27**; if the pool were "the most recent N" the recommendation would
be worthless and we would find out at the deadline. Put to the rules text (saved in `work/comp_pages/`):

> *"You may select up to two (2) Final Submissions for judging."* … *"A 'Final Submission' is the
> Submission **selected by the user** … used for final placement."*

**Zero hits for any pool restriction** (`most recent` / `last two` / `last 2` / `previous submission` /
`only the` / `within the last`). ⇒ **selection is from the full history; the recommendation stands.**

Four other facts the rules settled, all previously assumed or wrong:

| fact | value |
|---|---|
| Metric | `(1/12) Σ AUCᵢ` — macro ROC-AUC over the twelve targets ✓ |
| Run-time cap | **CPU ≤ 9 h, GPU ≤ 9 h**; internet **disabled**; file must be `submission.csv` |
| Entry Deadline | **2026-10-15** — must have accepted the rules to compete (satisfied) |
| Winners' Requirement Deadline | **2026-11-05** — training code, video and method description, **if** we place in a prize position |

⚠️ **The run-time cap is 9 h, so the "21600 s = 6 h" figure I had been quoting was wrong for BOTH
readings** — it is neither the 30 h weekly allowance nor this competition's session cap.
`docs/FIND-2026-10-05-rules-verified.md`.

### 11. ⚠️ WE ARE OUTSIDE THE BRONZE (+1 → **−4**), and a new arm was built, verified and submitted

**Live at 2026-10-05T09:14:38Z**: teams 5195, **rank 523**, bronze line 519, **margin −4**, `0.943`
block 382..1395 (1014). ⚠️ **The two instruments disagree on direction**: the in-block percentile
*improved* (14.02 % → **13.92 %**) while the absolute margin fell from +1 to **−4**, because the bronze
line is an absolute top-10 % line and our rank moved 518 → 523 while the block's floor moved 376 → 382.
**The medal depends on the margin, so we are currently out.**

**`0943cfmsft2flat` — `ref 56848247`, PENDING**, submitted 09:14:47Z. Built for the reason in §9: the
recommended second slot `0943cflat` sits on `0943cf`, which is no longer our best. With `C` = cn0 +
fracture fix, `S` = msft2 view, `R` = routing flattening, the pair values are
`Y = (cfmsft2, cflat) = C + max(S,R)` and `Y' = (cfmsft2, cfmsft2flat) = C + S + max(0,R)` ⇒ **`Y' ≥ Y`
for every sign of `R`**. It cannot make the pair worse, because the pair's value is the better of the two
and `cfmsft2` stays in it.

⚠️ **Verified three ways, because the `parent` preset flattens `coatnet_w` AT RUNTIME and the literal
0.65→0.75 map is still in the source** (the trap `build_0943_addmember.py` L214-216 records):

| check | result |
|---|---|
| cell-by-cell source diff vs `cfmsft2` | **exactly 1 changed line** (`PRESET … "speedy"` → `"parent"`) |
| the run's OWN log | `[config] preset=parent` · `outer CoAtNet weight per finding: flat 0.65` · all 12 findings **0.650** vs cfmsft2's non-uniform map (ACL/Fracture/LatOA 0.750, LatMen 1.000, MedMen 0.800) |
| final submission SHA | `095bcf70…` → `c1478aac…` ⇒ the edit reached the output |
| `msft2` present | cfmsft2's only delta over `0943cf` is the msft2 view, and it moved the rank ⇒ msft2 is in cfmsft2; the one-line diff ⇒ it is in the flat arm too |

⚠️ **Gap, recorded not papered over**: `[raptor-fast]` never logs the Raptor arm composition, and the
`_raptor.csv` intermediate could not be retrieved (`kernels_output`'s `file_pattern` rejects `*.csv`
with a ValueError and returns the `.log` regardless). The `msft2` conclusion therefore rests on the
diff chain above, not on a direct sighting in this run.

⚠️ Also confirmed normal, not an anomaly: the pushed run processed **3 test studies / 15 series** while
`[eda]` reports 1322 test studies. Code competitions are **re-run server-side on the full hidden test
set** — which is why submissions sit PENDING for hours. The local execution is a preview.

**Two arms are now PENDING**: `56845973` (meniscus) and `56848247` (flat). Both read against the
**13.92 %** in-block bar and the **−4** margin recorded above.

### 12. ⭐ BOTH ARMS LANDED AT OR BELOW BASELINE — the specialist is dead and the final pair changed

| ref | arm | public | vs baseline |
|---|---|---|---|
| `56814081` | `0943cfmsft2` (`C+S`) | **0.943** | baseline — **still our best** |
| `56826372` | `0943cfmsft2m` (`C+S+M`) | 0.943 | flat |
| `56599932` | `0943cf` (`C`) | 0.943 | flat |
| **`56848247`** | **`0943cfmsft2flat` (`C+S+R`)** | **0.942** | ⚠️ below |
| **`56845973`** | **`cfmsft2m-men`** | **0.940** | ⚠️ below — worst of the round |

**1. The meniscus specialist is a REGRESSION, and the prereg had already written what that means.**
`PREREG-2026-10-05-meniscus-arm.md`: *"公榜 ≤ 0.940 ⇒ 真回归，丢掉"* and *"持平或更差 ⇒ 这一条也无效 …
本轮所有轴都被否证 ⇒ 转入收尾"*. ⇒ **The label-specialist axis is falsified, and `msft3-spec` is NOT
FIRED** — its own §0 made the meniscus result a hard entry gate and the gate did not open. **That is the
prereg working, not a wasted build**; the alternative was ~2.6 GPU-hours on a measured-dead hypothesis.

**2. The routing hedge is measured unhelpful.** `cfmsft2flat` = 0.942 < 0.943, and it was **verified to
have applied exactly the intended change** (§11), so this measures the *flattening*, not a failed edit.
⇒ `R ≲ 0`, which retires the argument `FINAL-SELECTION-2026-10-04.md` §3 was built on. **`0943cflat` is
no longer the right second slot.**

**3. ⭐ FINAL PAIR REVISED** (`docs/RESULT-2026-10-05-two-arms-flat-or-worse.md` §3):

| slot | ref | arm | public |
|---|---|---|---|
| **primary** | **`56814081`** | `0943cfmsft2` — `C + S` | 0.943 |
| **second** | **`56599932`** | `0943cf` — `C`, deliberately **without** `S` | 0.943 |

With `M ≲ 0` and `R ≲ 0` both measured, `(cfmsft2, cf) = C + max(S, 0)` is the **only** candidate pair
that hedges `S` — the one effect whose evidence is a public-**rank** gain, so whose private sign is
genuinely unknown. `C` is in **both** arms; the second slot gives up `S`, and that **is** the hedge.
⚠️ **This is NOT the "delete a measured improvement to buy diversification" error** the 10-04 sheet
warned against — that was about deleting `C`.

**4. Board at 2026-10-05T16:01:59Z**: teams **5226**, rank **551**, bronze line **522**,
**margin −29**, `0.943` block 414..1419 (1006), in-block **13.63 %**.
⚠️ Margin across the day: **+1 → −4 → −29**. In-block percentile meanwhile *improved* (13.92 → 13.63).
**The instruments disagree in direction; for a medal read the margin.**

**5. Where the campaign stands.** Five axes tried, fourteen arms, **one measured gain (`S`)**. The
structural boundary in `STRATEGY-2026-10-04-how-far-can-finetuning-go.md` §4 has survived every cheap
crossing attempt. Any further arm must target a **new axis**, not a variation of a closed one.

### 13. ⭐ EVERY AXIS IS NOW MEASURED SHUT — the campaign is at its structural ceiling

Since §12, three more axes were closed **without spending an accelerator on them**, and the last one was
closed by the user's own pushback:

| axis | verdict | how it was closed |
|---|---|---|
| External / more **data** | ⛔ **closed, proven** | the raptor corpus is the competition train set **id for id** (`corpus_ids == train_ids`, 4407/4407); and no public Kaggle-mountable knee MRI labels more than ACL + meniscus. |
| Competition-specific **Kaggle Models** | ⛔ **closed, measured** | the only evaluable one scores **0.7992** macro on 4407 studies — 0.10 below the A167 gate. The rest are repackages of what we already mount, or undocumented. `RESULT-2026-10-05-models-axis-closed.md` |
| **Five-fold `msft2`** | ⛔ **not proceeded** | user: *"没必要做五折吧"* — and they are right. `msft2` gained because it went absent → present; folds 1–4 are the same recipe on overlapping data, and the sub-blend already rank-means six views. ~10–13 of the remaining 16.21 TPU-h for a gain below display resolution. `PREREG-2026-10-05-five-fold-msft2.md` |
| Specialist | ⛔ closed | meniscus `56845973` = **0.940** vs a 0.943 baseline |
| View-adding | ⛔ closed | `FACTS` D14, two independent attempts |
| Our own ConvNeXt | ⛔ closed | gold-58 0.9003 but **870-OOF 0.8493** — instruments differ by 0.051 |
| Routing / weights / calibration | ⛔ closed | `R` now measured too: `cfmsft2flat` = **0.942** < 0.943 |

**⇒ FINAL PAIR (locked):**

| slot | ref | arm | public |
|---|---|---|---|
| **primary** | **`56814081`** | `0943cfmsft2` — `C + S` | 0.943 |
| **second** | **`56613303`** | `0943cflat` — `C + R`, **without** the public-fitted routing | 0.941 |

⚠️ **The second slot was `56599932` (`0943cf`) for a few hours on 2026-10-05 and has reverted.** Reasoning
and the full case check: `docs/REVISION-2026-10-06c-second-slot-reverts-to-cflat.md`. The short version:
the two candidates are **exactly equal on public numbers**, so the choice must be made on the organisers'
**Dataset Distribution Notice** — *"prevalence … is not guaranteed to be the same across the training, public
leaderboard, and final evaluation datasets"* — and `cflat` is the one that drops the **public-fitted**
per-label routing map. **The switch is free on every measurement we hold.**

⚠️ **Still a BROWSER-ONLY action** (§9: no `select` verb, no `selected` field, `/api/v1` 401s). The
auto-ticked default is **not** this pair.

**Board at 2026-10-06T06:11Z**: teams **5278**, rank **645**, bronze line **527**, **margin −118**,
in-block **16.03 %**, top **0.964**. ⚠️ Margin across 10-05/10-06: **+1 → −4 → −29 → −118**, and the
`0.943` block is **shrinking** (1014 → 999) while its floor rises 376 → 485 — **we are in the lower part of
the block and falling through it.**

**Remaining**: 16.7 days; 4 of today's 5 submission slots; **GPU 24.31 h, TPU 12.66 h** (both refresh
2026-10-10). ⚠️ **CORRECTED 2026-10-06 14:1xZ — the "< 30 min" sentence that stood here was WRONG, and it
was wrong in the direction of blaming the system.** Two community claims conflict and I had recorded only
one of them: `qierroo`/`tensorhydra` claim *"sub-30 minute inference on the entire hidden test set"*, but the
author of **the exact recipe we are running** writes, verbatim, in
`intoxicatinglips/rsna-knee-pure-0944-verbatim`:

> `RUN_STACK = True` (default): community stack + our reader, the 0.944 blend. Scoring the hidden test set
> takes **6.5 to 8 hours** of the 9 allowed. **It is not stuck; let it run.**
> `RUN_STACK = False`: our reader only. Scored in **about 20 minutes**, public LB 0.929.

⇒ **Where the 20 minutes came from**: it is `RUN_STACK=False`, the reader ALONE. I generalised it to the
whole pipeline. **The full stack + reader is 6.5–8 h.** Our three pending submissions have been PENDING
3.7–7.9 h, which is consistent with that — though PENDING includes queueing, so it does not prove it.
**Do not declare a stall before the 9 h cap.** Full intel: `docs/INTEL-2026-10-06-rsna.md`.

---

## ⭐⭐ 2026-10-04 DELTA — READ THIS BEFORE ANYTHING BELOW

**This section supersedes several statements further down, which are now known to be WRONG.
The biggest one: `msft2` is NOT closed and its measured value is NOT zero.**

### 1. The public board has a sub-resolution instrument we were not using: the RANK

`0.943` is displayed for **953 teams**, yet they occupy **contiguous distinct ranks 357..1309**.
⇒ **Kaggle ranks on precision finer than the 3 displayed decimals.** Consequences:

* **the target is NOT "+0.001 to reach 0.944"** — it is to move up ~21 % of the block width, i.e. **≈ +0.0002**;
* the repo's premise **A96 ("the board cannot rank these arms, only disqualify") is true for the
  displayed SCORE and false for the RANK**.
* ⚠️ **The rank is confounded by field drift** — always record a snapshot BEFORE and AFTER a submission
  and compare the **in-block percentile** `(rank-lo)/(hi-lo)`, never the absolute rank.
  Measured drift with no improving arm: **−31 ranks in 16 h**.
  ⚠️ **CORRECTED 2026-10-05 — the "≈ −2/day" that used to end this line was a unit slip**
  (31/16 = 1.94 per **hour**, not per day). Two measurements now agree:
  −31 ranks in 16 h ≈ −47/day, and the 2026-10-04→10-05 margin collapse (+23 → 0 in 16.3 h) ≈ **−34/day**.
  **Working figure ≈ −40 ranks/day ≈ −1.7 ranks/hour.**

Full text: `docs/FIND-2026-10-04-rank-resolution.md`.

### 2. ⭐ `msft2` was built, submitted, and **measurably improved us** — killing §13.2's expectation

| arm | ref | result |
|---|---|---|
| `0943cfmsft2` = `0943cf` + our msft2 CoAtNet SWA as a **6th Raptor view** (w=0.20, share 14.3 %) | **`56814081`** | public **0.943** — **but rank 708 → 487** |

| | before 04:13Z | after 12:44Z |
|---|---|---|
| our rank | 708 | **487** |
| `0.943` block | 344..1280 (937) | 357..1309 (953) |
| bronze line (top 10 %) | 506 | 510 |
| **in-block percentile** | **38.89 %** | **13.66 %** |

**⇒ +221 ranks while the field grew 0.8 %; the date confound was excluded
(`pearson(rank, submission_time) = −0.13`, dates scattered at both ends).**
**⇒ rank 487 < line 510: we are currently INSIDE the bronze, by 23 places.**

⚠️ **This falsifies, in this case, HANDOFF §13.2's "a 0.910 member at 10–15 % weight → 0.000".**
msft2 is 0.9069 at a 14.3 % share and it moved us. That "0.000" was a **single noisy LB reading**,
and both I and this file treated it as a conclusion. **My own pre-submission expectation of "≈0" was
wrong.** Full text: `docs/RESULT-2026-10-04-msft2-arm-rank-move.md`.

### 3. Two other arms landed, both flat — and one of them closes a long-open question

* **`56798408` `0943cf448`** = `0943cf` + the author's PUBLIC one-fold M448 ConvNeXt fallback at w=0.10
  → **0.942**, in-block percentile 39.5 % → 38.9 % ⇒ **no measurable change.**
  ⚠️ The author's **trainer is not public** ⇒ folds 1–4 cannot be reproduced ⇒ **that line is dead.**
  `docs/PREREG-2026-10-03-m448-arm.md`.
* **`56791368` `0943cfm`** (+MCL) → **0.942**, and the pre-registered rule fired (≥0.942 ⇒ legitimate
  primary candidate). The amendment it triggered changes the **second slot** to
  **`56613303` `0943cflat`**. `docs/FINAL-SELECTION-2026-10-04.md` (amendment at the end).

### 4. What the 356 teams above us actually use — third exhaustive sweep

`docs/FIND-2026-10-04-what-is-above-us.md`. Headline: **the top 27 publish nothing for this
competition**; 188/464 of the usernames above us publish *something*, but only ~21 publish knee work,
and those are the same public-lineage authors whose ceiling is 0.943. **Top submission counts 26–298
vs our 41.** The one complete public recipe found was **a faithful replica of the pipeline we are
already on** — including the answer to a question this repo had left open:
**our `_RAD_CAL`'s "unknown provenance" is the public recipe's frozen 12-output calibration table.**

⚠️ **We are NOT missing DINOv3** — `mattiaangeli/knee-mri-fold-weights` (our `a5_w` member) is
**byte-identical** to the bundle's `dinov3/m_f0.pt`, backbone `vit_small_patch16_dinov3.lvd1689m`.
An earlier note in this session claiming "we use DINOv2, they use DINOv3" was **wrong**.

### 5. Open right now

| | |
|---|---|
| **The two final slots** | still a **browser-only** action. `0943cfmsft2` (`56814081`) is now the only arm with a **measured** gain. |
| **`ppxl16/rsna-knee-msft2-cx`** (ConvNeXt fold-0 fine-tune) | **QUEUED on TPU.** Pre-registered kill rule: **SWA gold < 0.88 ⇒ stop, do not run folds 1–4.** `docs/PREREG-2026-10-04-convnext-member.md` |
| **Drift** | ≈ **−40 ranks/day** (−34 to −47 measured twice, 2026-10-05). ⚠️ Earlier text here said "−2/day" — that was a **unit slip**, corrected. **The margin is now +1 (see the 2026-10-05 DELTA at the top), so we are one place from falling out.** |
| Strategy | `docs/STRATEGY-2026-10-04-how-far-can-finetuning-go.md` — same-corpus fine-tuning is **measured** to have decayed ~11× (cn0 +0.0028 on a 0.917 base → msft2 +0.00025 on a 0.943 stack). Stop rule proposed there. |

---

# (historical) STATUS — 2026-10-02 08:05Z

**One screen. Rewritten from scratch; the previous version was dated 2026-09-27 and had drifted.**
Deep detail lives in `FACTS.md` (A-numbers) and `docs/`; the recent session records are
`docs/RESULT-2026-10-02-msft2-finetune.md`, `docs/RESULT-2026-10-01-exp085-gate.md`,
`docs/FIND-2026-10-01-exp085.md`, `docs/FIND-2026-10-01-rsna-sweep.md`.

⚠️ **Everything from here down is the 2026-10-02 view and is retained only so the reversal is
auditable. In particular: the "msft2 is D7-closed" and "expectation ≈0.000" statements below
were REFUTED on 2026-10-04 (see the DELTA at the top).**

## ⛔ READ THIS FIRST — 2026-10-02: the msft2 / ConvNeXt plan was pursuing CLOSED directions

`state/EXPERIMENTS.md` §D is a ledger of every lever already **tested and closed**. It closes most of
what this file has been treating as open:

| closed | what it kills |
|---|---|
| **D7** | **"把我们的 raptor 成员塞进集成" is CLOSED.** The *author themselves* measured "adding a different-architecture backbone" = **+0.001** (C14), and we independently A/B'd it (`add-widedense-arm`) = **0.941 flat**. **⇒ the msft2 member is D7.** |
| **D5 / D6 / D17** | self-trained encoders with **solo <0.90 are net harmful**; the TPU route closed after predict crashed natively; TPU-trained coatnet closed. **⇒ the ConvNeXt arm is D5/D17.** |
| **D14** | "add a 5th view to the Raptor arm" — **two independent attempts, both 0.941 flat.** |
| **D1 / D3 / D13** | fusion-weight / routing / calibration tuning, incl. the community's "one weight" claim: **five readings all 0.941**; mattiaangeli's own metadata says *"sub-resolution blend tweaks should be treated as noise"*. |
| **D11 / D12** | turning the routing or calibration switches OFF is **negative** (monotone 0.941 / 0.940 / 0.939 / 0.938 four-corner). |

**And the instrument cannot resolve the target.** Public-LB **SE ≈ 0.008** ⇒ **differences ≤0.003 are
unjudgeable here**, while our target is +0.001. Confirmed too: `exclude-fracture` — the one
externally-claimed "+0.001" — was **already submitted (`56166772`) and read 0.941, same as canonical.**

⚠️ **RESOLVED 2026-10-03 — what bought the +0.002, and it is not a lever we still hold.**
`state/FACTS.md` **C79** records that the public frontier wrote down its own 0.942→0.943 changes, and
**we forked that finished artifact**. The three changes are:

1. a **third CoAt reader, `Global96`** — same 96-slice stack as D4 but a **different head** (full-width
   gated head vs D4's Rank8 finding-to-slot attention). Their words: *"共享几何、不同读者，这才是
   集成真正用得上的多样性"*;
2. the **reduction**: from *rank-each-then-average-the-ranks* to **average-the-probabilities-then-rank**
   (the former throws away confidence);
3. the outer CoAtNet default **0.60 → 0.65**.

Plus the Raptor window count **62 → 94** (with the speed engineering that made it affordable), and
**C81**'s removal of `'Fracture'` from `_RAD_EXCLUDE` — a decision that had been made on **gold-58**
and was **wrong**: re-measured on 4185 studies it is Fracture AUC 0.9187 → **0.9305**, reproduced
independently by five label sources, with **gold-58 the only dissenter at −0.0097**.

⇒ **All of it is already inside our kernels** (`kernel-0943f` is the fracture fork; A118 records
`3×0.943 + 2×0.942` across the five arms). **We are not below the public recipe — we are on it.**

⇒ **And the frontier itself annotated this whole change as "排行榜只动了 0.001"** — a three-part
structural edit plus 32 extra windows moved the public LB by 0.001. Combined with **LB SE ≈ 0.008**,
that is the clearest possible statement that **our +0.001 target is below this instrument's
resolution**, and that a null LB result cannot distinguish "no gain" from "gain we cannot see".

⚠️ **One recorded barrier is now stale and should not be quoted:** **D17** says CoAtNet cannot be
trained on `torch_xla` (it died in `timm/layers/pos_embed_rel.py` → `torch_xla/csrc/helpers.cpp`
broadcast check). **Our own msft2 v4 falsified that** — it trained
`coatnet_rmlp_2_rw_384.in12k` on TPU for 4/4 epochs to `partial: false` once the XLA activation-
checkpoint shim was in place. So "convnext is only a substitute" is no longer the reason to avoid
CoAtNet on TPU. (This does **not** revive the msft2 member: that is still closed by **D7** on
ensemble-value grounds, and by **D5** on the sub-0.90 rule — a different and better reason.)

## Where we stand

| | |
|---|---|
| Competition | **RSNA Knee Abnormality Detection** — code competition, 12-label macro ROC-AUC |
| Reward / teams | **$77,000** (incl. an **$18,000** efficiency track) / **4,692** scored |
| **Deadline** | **2026-10-22 23:59Z** — 20 days |
| Daily submissions | **5** (verified from the API) |
| ⭐ Our board | **rank 578 / 4692 · score 0.943 · 38 submissions** (`p pxl16`) |
| Board max | **0.961** — gap **+0.018**, i.e. 2.3× the ±0.008 noise band |
| ⭐ **The target** | **+0.001 → 0.944.** Our 0.943 is a **599-team block (ranks 263–861)** and the bronze cutoff (rank ≤ 462) falls **inside** it, so at 0.943 whether we medal is decided by an **opaque tie-break**. 0.944 is a 47-team block at ranks 216–262 — **+0.001 escapes it entirely.** |

## ⭐⭐ The headline: the fine-tune WORKED and clears the member gate

`ppxl16/rsna-knee-msft2` v4 — **COMPLETE, 4/4 epochs, `partial: false`**:

```
[zeroshot] GOLD macro-AUC 0.8961 | VERDICT: PASS (parent bar 0.8963)   <- warm start verified
ep0 0.8903 | ep1 0.8946 | ep2 0.9007 | ep3 0.9048
SWA over epochs [3,2,1]  gold 0.9069   <- better than the best single epoch
```

| reference | gold58 |
|---|---|
| parent checkpoint reloaded through **our** pipeline | **0.8963** (the bar) |
| this run, zero-shot (warm start, no training) | 0.8961 — PASS |
| this run, best epoch | 0.9048 |
| ⭐ **this run, SWA** | ⭐ **0.9069** |

**+0.0106 over our own warm-start number, and 0.9069 CLEARS the A167 member gate (`solo ≥ 0.90`).**
A167 established that **nothing this project can train from scratch passes it** (our ceiling ~0.877), so
fine-tuning a published strong checkpoint was the only route — **this is the first artifact of ours that
passes.** Calibration: exp085's author reports `gold58 0.9062`; our SWA reaches **0.9069**.

✅ **DONE — the checkpoints are now a dataset we own**: `ppxl16/rsna-knee-msft2-ckpt` (private, 8 files),
published 2026-10-02. The exp085 failure mode (a public weights dataset withdrawn mid-session, then
silently dropped by `kernels_push` while it reported success) no longer applies to this asset.

⚠️⚠️ **BUT the premise that this is "paid-for headroom" is contradicted by our own recorded evidence.**
`docs/HANDOFF.md` §13.2 (2026-09-11, from others' LB-measured results) says:

> **加自训模型没用。** 0.910 的模型按 0.10/0.15 权重加进 0.936 的公开底 → 变化 0.000；
> **0.874 的模型加进去反而 0.936 → 0.935（更差）**。基于 Gold58 预测的 +0.004 实际交出了 −0.004。

> **58 例 gold 集不能当裁判。** 它的 bootstrap 95% CI 比要追的效果还宽…… **不要用 Gold58 调权重。**

**msft2 is 0.9069 — inside the band that was already measured at 0.000.** It is also a CoAtNet, i.e. the
same *family* as the CoAt readers already in the stack, and §13.2 says the ensemble lives on
**divergence, not quality** (the weakest member, solo 0.843, contributed +0.009; another of the
*strongest* DINO folds contributed −0.0007). So **the honest expectation for msft2 is ≈0.000, possibly
slightly negative** — and the "0.943 → 0.944 is untested" framing below must be read with that in mind.
The arm is therefore built to **measure divergence first** and defaults to weight 0.0.

## ⚠️ Where the headroom is, and where it is NOT

### Closed: better public artifacts
Full sweep of newest + top-voted notebooks, datasets and the forum
(`docs/FIND-2026-10-01-rsna-sweep.md`). **The public frontier IS 0.943 and we are on it:**

| name claims | the artifact's own evidence |
|---|---|
| `…-d4-public**0946**` family | **0.943** — *"the 0946 … overstates what the public pipeline alone achieves"* |
| `0-biohub-…-**0947**-lb` | private assets |
| `parent**0957**-routing` | **0.941** — *"Public score 0.941"* |
| `the-**2965**-master-hybrid-engine` (Kaggriculture, 09-29) | **byte-identical to `frontier-top2`** |

⇒ **A notebook name on this competition is systematically inflated, always upward.** Only the
artifact's own text and a local gate are admissible.

### Closed: exp085 as a member
Reconstructed from its checkpoint alone (weights + config, code private). Module tree **proven**
(`strict=True`, 245/245), **head order A** recovered (+0.13), **`eval_group_agg: mean`** recovered
(+0.02, max/median ruled out), `slice_select` and `crop_mm` **eliminated**. Best **0.8191 against the
author's 0.9062** — **below the 0.90 gate**, and the dataset was then made private. Recorded and closed.

### ⭐ OPEN — the one thing that matters, and it is not built yet
**The msft2 member has never been added to the ensemble and never submitted.** +0.0106 on gold is a
*solo* number; **A163 says a member must buy DECORRELATION, not just strength.** Until an arm carries it
and the board is re-read, `0.943 → 0.944` is untested. **This is the whole of the remaining headroom
that is already paid for.**

## Are there OTHER arms worth fine-tuning? — yes, but only ones we can still attach

The msft2 recipe is: take a **published strong checkpoint**, carry the head, **verify the warm start at
zero training**, then fine-tune with a low backbone LR and a higher head LR for ~4 epochs, keeping the
best epoch **and** an SWA. Applying it to a **different family** is how a second, *decorrelated* member
would be produced — which is what A163 demands.

| candidate | accessible? | why it is interesting |
|---|---|---|
| `kawaharataishi/rsna-knee-coatnet-distilled-checkpoints` (543 MB, 09-29) | ❌ **403** | the D4/CoAtNet distilled family — the most on-family candidate |
| `tbongc/rsna-knee-raptor-v8geo/v10geo` (271 MB each) | ❌ 403 | Raptor variants, same family as msft2 → **low decorrelation value anyway** |
| `apoorvadixit/rsna-knee-weights-v4cx` (5 × 199 MB) | ✅ | **ConvNeXt-small** — a *different architecture*, so the best decorrelation candidate on the list. ⭐ **CHOSEN — `v4cx0` is running now as `ppxl16/rsna-knee-msft2-cx` (QUEUED, TPU).** |
| `soyaoki/rsna-knee-my-models` (20 files, 89 MB each) | ✅ | a 5-fold set with `_epoch5`/`_epoch10` variants |
| `lixin73/rsna-knee-train-checkpoints` (82 MB) | ✅ | unchecked |

⚠️ **Budget check before starting one**: TPU remaining ≈ **17 h** and a 4-epoch run takes **2.6 h** — so
several are affordable, but the **GPU quota is 0.00 h until 2026-10-03**, and **no arm can be pushed or
read before then.** A second fine-tune can be *run* today; it cannot be *used* today.

## The instrument, and its honest limits

* the metric is **macro ROC-AUC over 12 binary findings** — only **ordering per column** counts, so any
  monotone transform is score-neutral *alone* and matters only inside a blend;
* **ties are expensive**: a study nobody can score gets a constant, and a block of identical constants
  earns half credit against everything it ties with — **the same mechanism as the 599-team block that
  decides our medal**;
* **58 gold studies is a small set.** The +0.0106 is real but has poor precision, and it is a *solo*
  screen, not a leaderboard prediction;
* local OOF cannot settle member value: **A84** established the public artifacts are checkpoints rather
  than features, so `coat_private_alpha` and the individual CoAtNet members **cannot be priced on OOF**.

## Blockers, both with dates

1. ⚠️ **GPU quota 0.00 h account-wide until 2026-10-03** — `enable_gpu` kernel versions cannot be
   created, so **no new arm can be pushed or scored for ~18 h.** (ARC is blocked by the same quota.)
2. **The efficiency track needs a browser action**, and the board has no page (it is a staff notebook
   output). ⚠️ The user has said plainly they cannot see it — **treated as out of scope**; the main
   track is the target.

## Next actions, in order

> **2026-10-04 delta — current sheet is `docs/FINAL-SELECTION-2026-10-04.md`.**
> * ⛔ **EffB0-MIL LINE CLOSED 2026-10-04 (user decision).** v3 COMPLETE: the per-term pin fixed
>   (**7/7 PASS**), dose **0.150** confirmed, and the safety net fired cleanly (`_EffOutOfTime` →
>   `_pipeline_stage.csv` = exactly `0943cfm`'s bytes ⇒ **duplicate, not submitted**). **The real
>   blocker was never the budget: the rebuild produced `0 slot-series` for all four TRAIN studies, so
>   the gate would REJECT even with unlimited budget.** Closed on four reasons — falsified rationale,
>   effect below the board's resolution, cost, and nothing new left to learn. Full text:
>   `docs/RESULT-2026-10-04-effb0-v3.md` (closure section).
> * 🔄 **`0943cfm2eff` v3 was the last EffB0 run** — nothing further will be pushed for this line.
> * ✅ **`0943cfm` = `56791368` (+MCL) LANDED: COMPLETE, public 0.942.** The pre-registered rule fired
>   (§"Steps" step 5 of `docs/FINAL-SELECTION-2026-10-04.md`: ≥0.942 ⇒ legitimate primary candidate).
>   ⭐ **The amendment it triggered is in that doc** — the second slot changes from `0943p` to
>   **`0943cflat` = `56613303`**, an arm the current sheet had dropped from its candidate table even
>   though **A161** had already recorded that it *weakly dominates* `0943p` (same flattening, plus the
>   +0.0028 member and +0.00099 fracture fix). The "concentrated bet" objection that stopped A161 from
>   acting **does not survive being written out**: over the better-of-two aggregator, pairing
>   (`0943cfm`, `0943cflat`) is **pointwise ≥** (`0943cf`, `0943p`) whenever the member/fracture effect
>   and the MCL effect are ≥ 0. **This is now the recommendation; the click is still browser-only.**
>   ⚠️ New: **`submission-limits` is not a verb in kaggle 2.1.2 *or* 2.2.2** (`2.2.2` has
>   `team-submissions` instead), so A135/A162's quota reading is **no longer obtainable from the CLI**.
> * ✅ **Reusable from the closed line**: `scripts/verify_effb0_arm.py` (acceptance test, battle-tested),
>   `scripts/harden_effb0_budget.py` (built, verified, never pushed), and the honest-dose method
>   (`work/eff_min.py`, `eff_dose.py`, `eff_dose_parents.py`, `eff_convention.py`).
> * ⭐ **EffB0 findings that stand regardless**: honest OOF +0.00403 at w=0.10, **4/4 folds**, CI
>   [+0.00285,+0.00491], and **the OOF estimate EXCEEDS the in-sample one**; **w=0.15 is the largest
>   dose unanimous under BOTH soft and binary conventions**. ⛔ But "helps most where we are weakest"
>   is **FALSIFIED** (weakest-3 +0.00414 vs strongest-3 +0.00544; Fracture +0.0013; Baker's negative).
>   `docs/FIND-2026-10-04-effb0-honest-test.md`.
> * ⭐ **MEDAL BAND, measured from a third-party leaderboard scrape** (`docs/FIND-2026-10-04-medal-band-density.md`):
>   the bronze line is **0.943 with 794 teams in that single 0.001-wide bin** (16.4 % of the board), and
>   **the line has been FLAT at 0.943 since 2026-09-25** while the field grows (`bronze_rank` 455→482).
>   ⇒ the medal is decided **inside a dense band**, so **this corrects my "below resolution so pointless"
>   framing**: it was right about *measurement* and wrong about *ranking*. A sign-stable +0.001 has real
>   expected rank value even though the board cannot see it — which **strengthens the MCL bet**.
>   ⚠️ The panel's `our_*` fields are the *collector's* account (0.948), not ours; the band is public
>   while the medal is private, so it is a proxy for density, not for ordering.
> * ✅ **Provenance AUDIT (`docs/AUDIT-2026-10-04-provenance.md`)**: `kernel-0943` is a **byte-identical**
>   fork of the frontier source (**0 of 51 cells differ**); `0943cf` = +cn0 view +fracture fix;
>   `0943cfm` = +MCL. No accidental deviations anywhere.
> * ✅ **`AMP_PREF='bf16'` CLOSED on provenance** — byte-identical in the frontier source, so switching
>   would deviate from a validated config on a foreign anecdote (§5a of the artifact scan).
> * ⚠️ Also recorded: `EXPERIMENTS.md` D1 says "all pre-registered axes" but **`rad_alpha` was never
>   submitted** — closed by *category*, not by a completeness proof.
> ⚠️ **Acceptance test for `0943cfm2eff` is a script: `scripts/verify_effb0_arm.py`** — it decodes the
> **UTF-16** log (which `kaggle kernels logs … *> file` writes under PS 5.1 and which made a successful
> run look empty), parses the gate verdict, and compares the arm against its parent.
> ⛔ **Do NOT judge a duplicate by `_pipeline_stage.csv` alone.** That file holds **ranks**, and the
> public run covers only **3 visible studies** (6 possible orderings per label), so a genuinely
> applied change can leave it byte-identical. **Verified against a known-applied change**: `0943cfm`
> (+MCL) and `0943cf` have the *same* 3-study stage file and `submission.csv`, while their
> `v558_rad_receipt.json` and `_coat_arm_receipt.json` **differ** — so MCL did apply, and
> **`56791368` is NOT a duplicate**. The **receipts are the primary signal**; the log arbitrates; the
> stage hash is corroboration only. An earlier version of this rule would have rejected a working arm.

0. ⭐⭐ **THE ONLY ACTUAL DECISION LEFT: the two final submission slots — and it is a BROWSER job.**
   `docs/FINAL-SELECTION.md` is the pre-registered decision (written before the last four readouts
   landed, then read against its own table). Its recommendation is **`0943cf` + `0943p`**:
   * **primary `0943cf` = ref `56599932`** (public 0.943) — the only arm carrying **two** locally
     measured changes (`cn0` as a 5th Raptor view *and* the C81 fracture fix), and the only member
     this repo has ever measured **above** the frontier's own DINOv2 seed members (0.8769 vs
     0.8676–0.8697, A130);
   * **second `0943p` = ref `56599159`** (public 0.941) — outer routing **fully flattened**, i.e. the
     frontier's *own* written remedy for the per-label map it describes as tuned on "a public split
     that is a fraction of the test set" and as "the likeliest place to give back points privately".
     Pre-registered alternative at the same hedge direction, one notch weaker: **`0943h` = `56568524`**
     (0.942).
   ⚠️ **This supersedes A118's `0943f` + `0943h`.** ⚠️ **The auto-ticked default is NOT this pair**
   (Kaggle auto-selects `0943` + `0943f`, A123) — so it must be changed by hand.
   ⚠️ **No CLI/API route exists**: `kaggle` has no `select` command, and `ApiSubmission` exposes no
   `selected` field (re-verified 2026-10-03, kaggle 2.1.2). Only the browser can do this.
   ⚠️ The doc explicitly leaves one thing to human judgment: the recommended pair deliberately carries
   the **single lowest-scoring** arm, because the public board *cannot* rank these arms — nine arms
   spanning four axes came back across a total spread of **0.941→0.943 = 0.002** against **±0.008
   noise**. If carrying the minimum is unwelcome, take `0943cf` + `0943h`.

0b. ✅ **`ppxl16/rsna-knee-msft2-cx` DELETED from the TPU queue** (2026-10-03). It was still `QUEUED`,
   so **nothing was destroyed**, and the single TPU slot (D18: concurrency = 1, a *queued* session
   already occupies it) is now free. Two independent reasons: the ConvNeXt member is closed on
   ensemble-value grounds (**D5** sub-0.90 rule, **D7** measured-flat), and TPU had only **1.88 h of
   20 h** left while the run needs ~2.6 h — so it could not have finished anyway.

0c. ⚠️ **`state/lb_ground_truth.csv` was STALE and has been regenerated** (2026-10-03, now 38 rows).
   It was missing `56669951` = **`0943s-2coat`**, public **0.942**. That arm is a clean single-axis
   A/B: `0943s` (sparse, **with** Global96) 0.943 → **minus Global96** 0.942, i.e. **Global96 measures
   at ≈ −0.001** — the same sign as C79's account, but well inside ±0.008, so it does not resolve.
   Its own description shows it is an **efficiency** play (≈50 min less hidden-rerun runtime against a
   break-even AUC cost of 0.034), not a score play.

1. ✅ **DONE — `ppxl16/rsna-knee-msft2-ckpt` published** (private, 8 files, `EXIT=0`). The exp085
   failure mode — depending on a dataset we do not control — is closed for this asset.
2. ⛔ **msft2 / ConvNeXt as ensemble members: CLOSED.** See the D-table at the top.
3. ⭐ **2026-10-03, GPU quota refreshed** — but there is **no member left to push.** GPU budget noted
   in `FINAL-SELECTION.md` was 21.85 h / 30 h at time of writing; selecting spends none.
4. **The only unmeasured axis left is corpus density**, and by the project's own pre-registered banding
   its best measurement (+0.0068 soft OOF, C84) lands in *"only worth upgrading the instrument, not
   building a new member"*. The layer-selection rule needed to build a denser corpus is **fitted and
   validated 34/35** (C82), so it is an engineering job on CPU, not a compute bet.

<details>
<summary>Superseded 2026-10-02 plan (kept only so the reversal is auditable)</summary>

1. ✅ `ppxl16/rsna-knee-msft2-ckpt` published (8 files, `EXIT=0`) — still true and still valuable.
2. ~~🔄 `ppxl16/rsna-knee-msft2-cx` IN FLIGHT on TPU~~ — **deleted, see 0b.** It was pursuing **D7**
   (our-member ensemble value measured flat) and **D5** (sub-0.90 members are net harmful). Its v1 had
   already died at the head guard; v2 was relaunched before the D-table was read.
3. ~~Build the msft2 arm into `kernel-0943s-2coat`~~ — **cancelled.** The arm was D7-closed before it
   was built; the subagent was stopped and left nothing behind (the 0.943 notebook is untouched,
   mtime 09-29).
4. ~~Push the arm and submit on 10-03~~ — **cancelled**, per the same D7/D5 closure.

⚠️ **Why the reversal happened:** this file's premise was "the msft2 member is the whole of the
remaining headroom that is already paid for". That premise came from reading **A167/A163** (a member
gate and a decorrelation rule) without reading **`state/EXPERIMENTS.md` §D**, which had already
*measured* the answer. The lesson recorded for next time: **before designing an experiment, read
§D of the ledger — it is the list of experiments already run.**
</details>
