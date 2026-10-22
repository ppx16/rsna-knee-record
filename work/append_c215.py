"""Append C215 (idempotent): the selected pair is optimal under both lineage outcomes."""
import io, os

p = r'D:\kaggle\rsna-knee\state\FACTS.md'
ADD = """| C215 | ✅🎯 **【两票的组合被证明在【两种血统世界】下都是最优 —— 我们并没有白丢一千分之一】** `work/pair_optimality.py` 把手上有分的 10 件产物按分数与 OAI 标记列表,再枚举可选配对。<br>**① 事实清单**：**0.958 全部是 OAI**（`57250245` / `57291788` / `57376589` / `57394596` 四件）;**最好的非 OAI 是 `57478661` = 0.950**（S6 单跑）,次之 `56878092` = 0.944。⇒ **不存在"非 OAI 的 0.958"** ⇒ **两票都是 0.958 的唯一办法是两票都 OAI。**<br>**② 三种配对的两种世界**：<br>| 配对 | OAI 可用 | OAI 被裁定不可用 |<br>|---|---|---|<br>| 两个最好的 OAI（0.958 + 0.958） | 0.958 | **（什么都没有）** |<br>| **我们的配对（0.958 + 0.950）** | **0.958** | **0.950** |<br>**③ ⇒ 结论**：两票都 OAI **在大概率世界里不多得一分,在极小概率世界里归零**。⇒ **混合配对严格占优**（除非 OAI 被禁的概率严格为 0,而那不是一个可以假设的数）。**唯一严格更差的选择是"① 不是最好的 OAI 产物"或"② 不是最好的非 OAI 产物"** —— 脚本对这两条都做了断言,当前 `VERDICT: the pair is optimal under both lineages`。<br>**④ 这条也回答了那个看起来像漏洞的问题**："我们有五件 0.958,为什么只勾一件?" —— 因为**第二票的用途不是追分数,是买断"归零"那个分支**。而 `numScoredSubmissions = 2`（C213）说明**平台只取这 2 条里更好的那条**,所以第二票的 0.950 在大概率世界里根本不参与计分,不构成机会成本。 | 2026-10-22 |
"""
txt = io.open(p, encoding='utf-8').read()
if '| C215 |' in txt:
    raise SystemExit('REFUSING: C215 already present')
with io.open(p, 'a', encoding='utf-8') as fh:
    fh.write(ADD)
print('appended C215; file now', os.path.getsize(p), 'bytes')
