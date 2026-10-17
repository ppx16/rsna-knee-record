"""Append A203 (idempotent)."""
import io, os

p = 'state/FACTS.md'
ADD = """| A203 | ⚠️ **【我在错误的目录里重写了一个已经存在的工具】ARC 的工具在 `D:\\kaggle\\arc\\work\\`,而 ARC 的**台账/状态文档**在 `D:\\kaggle\\arc-agi-3-ledger\\` —— 我在后者里新建了一个 `submit_code.py`,它其实早已存在于前者。**<br>**① 经过**：为了"用服务端真实错误去问 ARC 的每日额度",我在 `arc-agi-3-ledger\\work\\` 里**从零写了一个** `submit_code.py`（并且真的用它发了那次探测）。之后做 `glob **/submit_code.py` 才发现 **`D:\\kaggle\\arc\\work\\submit_code.py` 早就在那儿**,而且 `COMP`、`file_name='submission.parquet'` 都是对的。<br>**② ⇒ 写死的规矩**：**写任何新工具之前,先在【整个工作区】里按文件名搜一遍**（`glob "**/<name>.py"`）—— 这个项目有**两棵树**（`arc\\` = 工具+state,`arc-agi-3-ledger\\` = findings+STATUS),而我只搜了正在看的那一棵。**这与 A198/C164 同族**（"局部正确 ≠ 全局正确"）：我的新文件**本身没错**,错在**它不该存在**。<br>**③ 顺带发现一个会让人读到【过期数字】的坑**：`arc\\work\\recon.py` 读的是 **`arc\\state\\lb\\*.zip`**,而我今天把新榜下到了 `arc-agi-3-ledger\\leaderboard\\` ⇒ recon 打印的是**旧榜**（rank 1260 / 24.53 / subs 8）,而真值是 **rank 1161 / 26.41 / subs 10**。**⇒ 下榜必须下到 `arc\\state\\lb\\`**（或改 recon 的路径）。<br>**④ 处置**：已删除我新建的那个重复文件（它没有产生任何提交：探测被服务端 400 拒了）,并把重跑过的 recon 读数记进 HANDOVER。 | 2026-10-14 |
"""

if '| A203 |' in io.open(p, encoding='utf-8').read():
    raise SystemExit('REFUSING: A203 is already present')
with io.open(p, 'a', encoding='utf-8') as fh:
    fh.write(ADD)
print('appended A203; file now', os.path.getsize(p), 'bytes')
