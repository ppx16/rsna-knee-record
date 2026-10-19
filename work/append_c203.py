"""Append C203 (idempotent): an independent author's version history pins what actually bought the last thousandth."""
import io, os

p = 'state/FACTS.md'
ADD = """| C203 | 🎯🎯🎯 **【独立作者的版本史把我们最后那一千分之一钉死了：是 per-finding 权重,不是 reader】** `goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958` 的 README（10-16 更新）逐字：<br>· *"**Versions 25 and 26 scored 0.958** … heliosli's per-finding Raptor weights on top of the R6 flip-TTA recipe, **with the 224 mirror and this reader at 1/10**"*;<br>· *"**Versions 21, 22 and 24 scored 0.957**"*（同一套：R6 flip TTA + nartaa 384 + **224 mirror** + **reader at 1/10**,**但没有** per-finding 权重）;<br>· *"Version 23 (flip TTA for R7 too) **0.957**"*。<br>**① ⇒ 三条硬结论**：**(a)** **0.957 → 0.958 那一步 = 加 heliosli 的逐 finding 权重** —— 这与我们自己的**平权重孪生体 = 0.957**（C195）**独立吻合**（两个团队、两套实现、同一个数）;**(b)** **reader 在 0.957 的版本里就已经存在** ⇒ **reader 单独买不到那一千分之一**;**(c)** **flip TTA 用在 R7 上是负的**（v23 = 0.957）。<br>**② ⇒ 我们手上的 `57250245`（0.958 = 权重 + R6 flip TTA + 384/224,无 reader）** 与"公开面最好那一版"的差别**只剩下 reader@0.1**,而按 (b),它换不来可测的增益。**⇒ 我们已经站在公开面能达到的位置上。**<br>**③ ⇒ 顺带对 `09586` 给出【预登记预测】**：heliosli 那本是 **平权重 + reader**（C202④）,而按 (a) 平权重 = 0.957、(b) reader 补不回来 ⇒ **预测 `ppxl16/rsna-knee-09586` 显示 0.957**,不会超过我们手上的 0.958。**这样最后 1 次额度就不必押在它身上。**（若它真出 0.958 或更高,那就是"reader + 平权重"胜过了"权重"这一反例,同样有价值。）<br>**④ 又一个 R6 v1 的公开副本**：`rostigor/rsna-knee-r6-fe055406`（与 `heliosli/…-fe055406-w1` 逐字节相同,goodpjw 逐字说明）⇒ **R6 v1 现在有三条公开来源**（C186 的 w1、`mizeroluckygall/…`、`rostigor/…`）。<br>**⑤ 方法论**：**版本史比任何单次读数都硬** —— 同一本笔记本的 21/22/24 → 25/26 只差一组权重,而分数从 0.957 跳到 0.958。**这是本项目里最接近"受控实验"的外部证据**（比"某本 0.957 说它加了 X"强得多）。 | 2026-10-19 |
"""

if '| C203 |' in io.open(p, encoding='utf-8').read():
    raise SystemExit('REFUSING: C203 is already present')
with io.open(p, 'a', encoding='utf-8') as fh:
    fh.write(ADD)
print('appended C203; file now', os.path.getsize(p), 'bytes')
