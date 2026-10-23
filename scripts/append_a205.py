"""Append A205 (idempotent): "the kernel exists" is NOT "the code I just built is on the server"."""
import io, os

p = 'state/FACTS.md'
ADD = """| A205 | 🚨 **【我自己的"等空闲就推"工具在"我改了代码要出新版本"这个场景下会静默拒绝工作】—— 它看到内核已存在且 COMPLETE 就 `return 0`,于是**修好的 v2 永远推不上去,服务器上跑的还是坏掉的 v1**。**<br>**① 经过**：band32 v1 跑完(C177 的成员数守卫失败) ⇒ 我改好构建器、重建三个内核 ⇒ 挂后台 `push_when_free.py` ⇒ 它逐字回 *"EXISTS with status COMPLETE -- nothing to push"* 并**正常退出(rc=0)**。⚠️ **这是我在 A201 里为了"别再重推已存在的内核"而加的判断,它救了一个场景,却堵死了另一个场景。**<br>**② ⇒ 写死的规矩**：**"存在性"与"版本正确性"是两件事。** 推新版本要用**强制推送**（`python work/push_kernel.py kernels\\<dir>`,它不看存在性,直接 `SaveKernel` 建新版本）,**`push_when_free.py` 只适用于"这个内核还没建过"**。⇒ 修法：给 `push_when_free.py` 加一个 `--force`/`--expect-new-version` 参数（看 `lastRunTime` 是否比我构建的时刻新），**但今晚来不及验证,所以先写死在文档里**。<br>**③ 今晚的实际状态（必须让下一次会话知道）**：`band32-w15`/`w30` 在服务器上**只有坏的 v1**（`lastRunTime = 14:33`),修好的 v2 **只在本地磁盘上**；四次强制推送**全部撞上 ProxyError/SSL EOF 传输失败** ⇒ **v2 尚未落地**。**⇒ 下次开机的第一条 RSNA 命令就是**:<br>`cd D:\\kaggle\\rsna-knee; python work\\push_kernel.py kernels\\kernel-r7-c224-band32-w15`（对 w30/w08 同样）。<br>**④ 又一次同族**（A198/C164/A203/A204）：**"我加了保护"和"保护覆盖了所有场景"是两件事** —— 这次是我在**修一个坑的时候挖了另一个坑**,而它要等到**下一次"改代码重推"**才暴露。 | 2026-10-14 |
"""

if '| A205 |' in io.open(p, encoding='utf-8').read():
    raise SystemExit('REFUSING: A205 is already present')
with io.open(p, 'a', encoding='utf-8') as fh:
    fh.write(ADD)
print('appended A205; file now', os.path.getsize(p), 'bytes')
