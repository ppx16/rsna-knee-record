"""Append C199 (idempotent): the source of C197/C198 says its own recipe is unvalidated."""
import io, os

p = 'state/FACTS.md'
ADD = """| C199 | ⚠️ **【给 C197/C198 划清证据等级】那本笔记本【自己说】V33 没有测过公榜** —— 我们跟的是一个**未被验证的假设**，不是一条已证实的增益。**<br>**① 原文逐字**（`laimerbotrolenk/rsna-four-model-ensemble`, md 0）：<br>*"**The reported baseline is 0.957; V33 has no measured leaderboard score.**"*<br>*"TTA is the arithmetic mean of the plain and mirrored probabilities, then ranked per finding. Mixing these ranks is not equivalent to changing the probability average to 75/25. **The weights are an unvalidated hypothesis, not fitted on labels.**"*<br>*"The attached log covers three example test studies. It establishes execution on those studies, **not hidden-test AUC or evidence of an extra +0.001**."*<br>**② ⇒ 必须改口的说法**：C197（224 plain/TTA 拆分）与 C198（384 拆分）**不是"公开配方有、我们缺"**，而是**"公开面有人提出、但自己没测过"**。区别很重要：<br>· **C196（224 镜像）** 有**实测**背书（`one-free-flip`：显示分不变、rank 357→289）;<br>· **C197/C198** 只有**机制论证**（plain 与 TTA 是两个不同信号,且零额外前向),**没有分数证据**;而且 **C198 连机制论证都没有** —— 他们的表里 **384 臂是**单一 TTA 槽 32.5%,我们把它也拆开是**超出公开配方的一步**。<br>**③ 我们的先验（如实写）**：C197 **弱正**（机制清楚 + 他们显式给了这个配比）;C198 **不确定**（384 的 TTA 本身更稳,拆开可能反而降低那条臂的质量）。**⇒ 两枪都值得测,但不能当成"已知的增益"。**<br>**④ 顺带确认我们与他们的结构一致**：他们 `R7 30% + S6 20%`（compact = 50%,**S6 占 compact 的 40%** —— 与我们 C192 的 0.40 完全一致）;`384 TTA 32.5% + 224 plain 8.75% + 224 TTA 8.75%`（Raptor = 50%,**224 合计 17.5% = 0.35** —— 与我们一致）。⇒ **结构没错,分歧只在"拆 384"这一步。** | 2026-10-19 |
"""

if '| C199 |' in io.open(p, encoding='utf-8').read():
    raise SystemExit('REFUSING: C199 is already present')
with io.open(p, 'a', encoding='utf-8') as fh:
    fh.write(ADD)
print('appended C199; file now', os.path.getsize(p), 'bytes')
