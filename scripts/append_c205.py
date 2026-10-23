"""Append C205 (idempotent): evidence grade for C204's mechanism -- 3-row correlation is suggestive only."""
import io, os

p = 'state/FACTS.md'
ADD = """| C205 | ⚠️ **【给 C204③ 的机制解释标证据等级：它只在【3 行可见样本】上成立,属于"与观测一致",不是证据。】**<br>**① 我测出来的**：在 `out_p384` 的产物上算逐 finding 的 `spearman(plain, TTA)` —— **384 臂 = 1.000（12/12 findings）**、**224 臂 = 0.917**。⇒ 与"拆 384 等于没拆、拆 224 才真的分掉信息"完全一致。<br>**② 但它是在【3 个 study】上算的**：kernel 里可见的 `test.csv` 是 **3 行**(这也是为什么每个 `blend_receipt.json` 都写 `rows: 3`,而平台仍给我们 0.958 —— **这是 code competition,评分时平台用隐藏全集重跑内核**,3 行只是可见样本)。3 个点的 Spearman 只有 3! = 6 种排序,**"1.000" 完全可能是巧合**。<br>**③ ⇒ 正确表述**：C204 的**观测**(0.958 vs 0.957)是**证据**;C204③ 的**机制**(两臂的 plain 与 TTA 相关性不同)目前只是**与观测一致的假设**,`1.000 vs 0.917` 这个数**不足以支撑它**。要验它需要在**全量 640 个 study** 上算 —— 那需要一次跑全量的内核(或本地有全量 test.csv)。<br>**④ 为什么仍然要留着这条假设**：它给出一个**可操作的分支** —— 若为真,则"拆不拆"取决于 `spearman(plain, TTA)`:高相关就拆(无害)、低相关就别拆(有害)。这比"某条臂能拆某条不能"更可迁移。<br>**⑤ 这是本项目的第 N 次"别把弱证据当结论"**（A198/C164/C199/C201/C204）—— 但这次是**我自己刚提出的解释**,所以更要当场标等级。 | 2026-10-20 |
"""

if '| C205 |' in io.open(p, encoding='utf-8').read():
    raise SystemExit('REFUSING: C205 is already present')
with io.open(p, 'a', encoding='utf-8') as fh:
    fh.write(ADD)
print('appended C205; file now', os.path.getsize(p), 'bytes')
