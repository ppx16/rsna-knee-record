"""Append A201 + A202, refusing to run twice (the reason A202 exists)."""
import io, os, sys

p = 'state/FACTS.md'
ADD = """| A201 | ⚠️⚠️ **`kaggle kernels status <slug>` 对一个【刚刚创建】的内核返回 404 —— 我的"等空闲就推"循环因此把已经推上去的内核反复重推,每次重推都是一个新版本 + 一次新的 GPU 跑。**<br>**① 经过**：`band32-w15` 推送时客户端报 `Expecting value: line 1 column 1 (char 0)`（回执丢失，A199 见过）。随后循环用 `kernels status` 复查 ⇒ **404** ⇒ 判"没推上去" ⇒ **重试**。四次重试全是这个循环。**而 `work/kstate.py`（新，走 `ListKernels(user=...)`）显示：`rsna-knee-r7-c224-band32-w15  run=2026-10-14T14:33:16`** —— **它早就在那儿了**。`w30` 同理（14:33:31）。<br>**② ⇒ 两条写死**：**(a)** **存在性判断一律走 `ListKernels`**（`work/kstate.py` 的 `state()` / `mine()`），**不要用 `kernels status`** —— 它对新建内核会说谎；**(b)** **`kernels list --user` 也不可靠**（截断 + 把私有笔记本显示成空名字，今天 `--search band32` 只回了一行 `,[Private Notebook],,2010-04-01`）。<br>**③ ⚠️ 代价是真实的**：重复推送会**占满 2 个 GPU 并发槽**，把后面真正要跑的排到队尾。**这与 A199 是同一天内第二次被"工具报的失败"骗到**，但这次它**不是无害的**：无害的版本是"多问一次服务端"，有害的版本是"服务端说没有，其实有，于是我又推了一遍"。<br>**④ 修法已落地**：`work/push_when_free.py` 的 `status()` 现在先问 `ListKernels`（并要求**每次推送后 `forget()` 缓存**），只有 `ABSENT` 才推。 | 2026-10-14 |
| A202 | ⚠️ **【我把自己写的追加脚本跑了两次,把 C176/A200 写进了 FACTS.md 两遍】** —— 原因是那个脚本**没有幂等性检查**（纯粹的 `open(p,'a')`）。<br>**① 怎么发现的**：跑完第二次后文件从 541,460 B 变成 545,523 B，我核对了 C/A 编号才发现 `| C176 |` 与 `| A200 |` 各出现 **2 次**。<br>**② 怎么修的**：`rfind('| C176 |')` 定位第二份的起点，**断言前半部分各出现 1 次**后截断写回。⇒ 现在 C176/A200 各 1 次。<br>**③ ⇒ 写死的规矩**：**任何"追加台账"的脚本,第一件事必须是"拒绝重复"** —— 例如 `if '| A201 |' in text: raise SystemExit('already appended')`。台账的**编号唯一性**是它可被检索的前提（C1–C175 / A1–A200 一路没有重号，不能从我这里破）。<br>**④** 与 A186/A190 同族,但这次是**我自己的工具**在污染**我自己的判据文件**：**"我写的东西"同样要当成"别人的东西"来验证。** | 2026-10-14 |
"""

if '| A201 |' in io.open(p, encoding='utf-8').read():
    raise SystemExit('REFUSING: A201 is already in %s' % p)
with io.open(p, 'a', encoding='utf-8') as fh:
    fh.write(ADD)
print('appended A201, A202; file now', os.path.getsize(p), 'bytes')
