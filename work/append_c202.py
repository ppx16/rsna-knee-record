"""Append C202 (idempotent): 0.9586 is an unrounded claim; the board still shows 0.958."""
import io, os

p = 'state/FACTS.md'
ADD = """| C202 | ⚠️ **【纠正 C200 的一句话】`0-9586` 那本的 `bestPublicScore` 在榜上仍然是【0.958】—— 标题里的 "=0.9586" 是作者自报的【未取整】值,不是更高的一档。**<br>**① 事实**：`work/audit_components.py` 的精确 slug 清单里,`heliosli/this-is-best-public-dataset-new-version-0-9586` 的 `bestPublicScore` 打印出来是 **0.958**（不是 0.959）。⇒ **公开面【显示意义上】的天花板仍然是 0.958**,我们和它同档。<br>**② ⇒ C200③ 那句"显示即 0.959"是错的**,应读作:**它是一次【档内前移】的尝试**（作者的 0.9586 若为真,只会在同一档里排更前）。**⇒ 我们现在做的每一件事都属于这一类**:224 镜像、reader 腿、plain/TTA 拆分 —— 全都在同一个显示档内争位置。<br>**③ 这也解释了为什么"档内位置"是当前唯一可行动的指标**：我们 rank ~383（档内 207/1155）,银线在 70 名外;而**任何**显示分的跳档都需要一个**明显更强的基座**,公开面今天没有。<br>**④ 顺带一个可核对的结构差异**：`0-9586` 的 224 臂是**单视图**（它的 md 逐字："a 0.8 centre crop of the 384-pixel field resized to 224, **one view**"）+ **goodpjw 的 reader 按 0.003–0.035 的逐 finding 权重**融合,而**它的 `W_RAPTOR_C` 是平的**。⇒ **公开面最好那一版 = 平权重 + 224 单视图 + 极小权重的 reader**;我们测过的 224 镜像是 0.958（不损失）,平权重的孪生体是 0.957（掉档）。**⇒ reader 那一块我们还没测过,`09586` 就是在测它。**<br>**⑤ ⚠️ 教训（同 A198/C164 族）**：**别把标题里的数字当成读数**。`0.9586` 出现在**笔记本标题**里,而**读数是 `bestPublicScore`** —— 两者一个是作者的声明,一个是平台的记录。**这已经是本项目第 N 次"把声明当读数"**（C122 的 0.954 标题同族）。 | 2026-10-19 |
"""

if '| C202 |' in io.open(p, encoding='utf-8').read():
    raise SystemExit('REFUSING: C202 is already present')
with io.open(p, 'a', encoding='utf-8') as fh:
    fh.write(ADD)
print('appended C202; file now', os.path.getsize(p), 'bytes')
