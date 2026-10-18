"""Append C195 (idempotent): the flat twin measured 0.957, and an external team measured S6 as -0.001."""
import io, os

p = 'state/FACTS.md'
ADD = """| C195 | 🎯🎯 **【三条外部/内部读数一起改了两件事】未拟合孪生体 = 0.957（⇒ 那组 per-finding 权重在公榜上真值一千分之一）；而外部独立团队测量【S6 是 −0.001】⇒ tick① 退回"不加 S6"的那一件。**<br>**① 未拟合孪生体的读数**：`57333305`（= 0.958 配方 + S6@0.40，但 `W_RAPTOR_C` 放回全平 0.5）= **0.957**。**对照 0.958 的 tuned 版** ⇒ **heliosli 那组 per-finding 权重在公榜上确实值 0.001**（C186 说它"无法确认"是指**无法确认它是真信号还是拟合**；现在至少知道它**在公榜上确实产生 0.001**，只是仍不知道私榜是否也这样）。<br>**② ⭐ 外部独立复现（决定性的那条）**：`goodpjw2008/rsna-knee-does-s6-help-next-to-r6-lb-0-957` 的 README 逐字：<br>*"Version 4 showed that S6 next to R6 **costs 0.001 (0.957 against 0.958 for the same recipe without it)** … this notebook is now two measured negatives: **do not add S6 to R6**, and do not tune per-finding weights on the 58 gold studies."*<br>⇒ **他们独立测到 S6 是负的 0.001**，而我们三枪（S6@0.40 / @0.40-flat / @0.60）都不低于 0.957、且 tuned 版是 0.958。<br>**③ ⇒ 两个测量在 0.001 上不一致，而 0.001 正好落在半集噪声（SE≈0.003）里。判据只能靠【不对称性】**：**S6 从来没有测出过正效应**（最好是不动，最差是 −0.001）。而 **tick 的职责是"不要丢",不是"多赚"** ⇒ **tick① 退回 `57250245`（同一条 0.958 配方，但不含 S6）**（`work/tick_2026_10_18.py`），`verify_final_pair.py` ⇒ `pair present, COMPLETE, and legal`。**S6 的两件没有被丢弃**：`57291788` / `57313021` 仍然已评分、状态保留为 **S6-SPARE**；如果私榜奖励多样性，截止日仍可以勾它。<br>**④ 权重仍然保持 heliosli 的原值**：理由是三条独立读数同向 —— 我们的全平孪生 0.957、以及他们用 gold-58 调权重的 v6/v7 分别 0.957 / 0.955。⇒ **"在 gold-58 上重新调权重"被我们和他们都测成负的。**<br>**⑤ 今天最后用一个额度做判别实验**：`rsna-knee-0958-s6-s6w80`（S6 占 compact half 的 **0.80**）。0.40/0.60 都平坦 ⇒ 若 **0.80 掉到 0.957**，说明外部那条 −0.001 只是**在高权重处才显现**（我们能调和两个测量）；若 **0.80 仍是 0.958**，则 S6 的权重到 0.80 都无害，外部的结果更可能是他们那套实现的差异。<br>**⑥ 另外补做了一次【按更新时间】的数据集普查**（以前只按票数）：`kritikabenjwal/rsna-knee-bleg-s0-f01`（只有缓存代码）· `neronrx/mycelis-rsna-knee-4fold-hive`（"4-Fold ConvNeXt **384px**"四个 209.6 MB 权重,**无描述/无代码/无报告分数**）· `abhiiiish/rsna-knee-grandmaster-ensemble-3m`（resnet34 + convnext_tiny 折权重,同样无描述）⇒ **都没有报告分数、也没有可跑的代码**,从权重反推架构性价比太低 ⇒ **跳过**。**⇒ 按更新时间也确认：公开面今天没有新的强资产。** | 2026-10-18 |
"""

if '| C195 |' in io.open(p, encoding='utf-8').read():
    raise SystemExit('REFUSING: C195 is already present')
with io.open(p, 'a', encoding='utf-8') as fh:
    fh.write(ADD)
print('appended C195; file now', os.path.getsize(p), 'bytes')
