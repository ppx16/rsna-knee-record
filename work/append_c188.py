"""Append C188 (idempotent): the 0.958 breakthrough and what caused it."""
import io, os

p = 'state/FACTS.md'
ADD = """| C188 | 🎉🎉🎉 **【突破：0.958】rank 565 → **231**、铜余量 +40 → **+377**、银余量 −263 → **+73**（重新进银牌区）。而**功劳不在我们加的腿上,在于换掉了整个基座。**<br>**① 今天的五枪出分（服务端逐字）**：<br>| ref | 内容 | 显示分 |<br>|---|---|---|<br>| `57249966` | 0.957 基座 + band32@0.04 | 0.956 |<br>| `57249972` | 0.957 基座 + band32@0.15 | **0.955** |<br>| **`57250245`** | **0.958 那本逐字节 fork**（0.958 公开笔记本,R6 `-w1` 包） | **0.958** |<br>| `57250300` | 0.958 基座 + band32@0.08 | **0.957** |<br>| `57250670` | 0.958 基座 + band32@0.15 | 0.956 |<br>**② ⇒ 决定性的对照**：**同一个 band32 腿,在旧基座上 @0.08 是【正】(C183 被选中),在新基座上 @0.08 却把 0.958 拉到 0.957、@0.15 拉到 0.956** ⇒ **+0.002 全部来自换基座（R6 包 + flip TTA + 那次 per-finding 重加权）,腿是负的。** ⇒ **结论收窄：band32 那条腿的优势不是普适的,它只在较弱的基座上补了多样性;基座变强之后它变成稀释。**（与你昨天那条"别过拟合公榜"的判断一致：**能买到整档的是配方换代,不是腿**。）<br>**③ 位置（`work/daily_rsna.py`,板 2026-10-16T12:42Z,N=6085）**：**rank 231 / 0.958**;铜线 608 ⇒ **+377 在内**;银线 304 ⇒ **+73 在内**;我们所在的 0.958 档 = ranks **113..438（n=326）**,档内 **119/326**。**⇒ 这是开赛以来我们第一次【同时】**舒服地**在两条奖牌线之内。**<br>**④ ⭐ 普查原话**：`none above ours -> we hold the reproducible public ceiling`（1100 本公开笔记本、693 本有真实分,当日**没有**任何一本高于 0.958）。<br>**⑤ ⇒ tick 已更新（`work/tick_2026_10_16.py` → `state/final_pair.json`）**：**tick① = `57250245`（0.958,OAI 血统）**、tick② 保持 **`56878092`（0.944,非 OAI）**;`verify_final_pair.py` ⇒ **`VERDICT: pair present, COMPLETE, and legal`**。⚠️ 更新原因不只是"分更高"：校验器报了 **UNDECLARED CLASH** —— 新旧两枪的描述都以 `byte-for-byte fork of the public n` 开头（两本不同的公开笔记本）,**这正是那个唯一性检查存在的意义**,所以新条目里显式写了 `clashes_with: ["57127191"]`。<br>**⑥ ⚠️ 诚实标注（已写进 tick 的 `honest_uncertainty`）**：0.958 那本里的 per-finding 重加权是**前一天才改的、拟合在公榜半集上**（C186）,而公榜半集的标准误约 0.003 > 前 400 名的总分差 0.0022（C184）⇒ **0.958 里有一部分可能只是公榜拟合**。**扎实的部分是：我们现在持有与公开面天花板【同一件】产物**,而奖牌由私榜决定。 | 2026-10-16 |
"""

if '| C188 |' in io.open(p, encoding='utf-8').read():
    raise SystemExit('REFUSING: C188 is already present')
with io.open(p, 'a', encoding='utf-8') as fh:
    fh.write(ADD)
print('appended C188; file now', os.path.getsize(p), 'bytes')
