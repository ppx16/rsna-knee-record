"""Append A207 (idempotent): Get-Date's literal Z is NOT UTC."""
import io, os

p = 'state/FACTS.md'
ADD = """| A207 | ⚠️ **【PowerShell 的 `Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ"` 打的是【本地时间】,那个 `Z` 是我自己写在格式串里的字面量】** —— 我因此把 00:55Z 读成了 08:55Z,误以为"新一天已经过了 9 小时"。<br>**① 怎么发现的**：服务端提交时间戳（UTC）是 `2026-10-15 01:17–01:43`,而我这边以为当时是 09:4xZ;同一轮里 `daily_rsna.py` 自己打印的榜单时间戳是 `board read 2026-10-15T00:55:35`,与 `UtcNow` 一致。⇒ **三个读数里两个说 00:5x、一个说 08:5x,那个孤立的正是我本地拼格式串拼出来的。**<br>**② 正确写法**：`[datetime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")`（`UtcNow` + 显式格式），或 `(Get-Date).ToUniversalTime().ToString("o")`。**永远不要用本地时间拼一个 `Z` 上去。**<br>**③ 为什么这次没造成损失、但下次会**：今天所有动作都是**按服务端的额度/时间戳**判定的（提交计数、榜单时间戳、`status`），所以"我以为现在几点"没有改变任何决定。**但截止日的算术靠它**（RSNA 2026-10-22 23:59Z、ARC 2026-11-02 23:59Z）—— 差 8 小时足以让人误判"还剩一天"。<br>**④ 与 C140⑥/C171③ 同族**：**时间也是"要从源读"的量**,源是 `UtcNow` 或服务端时间戳,不是我自己拼出来的字符串。 | 2026-10-15 |
"""

if '| A207 |' in io.open(p, encoding='utf-8').read():
    raise SystemExit('REFUSING: A207 is already present')
with io.open(p, 'a', encoding='utf-8') as fh:
    fh.write(ADD)
print('appended A207; file now', os.path.getsize(p), 'bytes')
