"""Append A209 (idempotent): the public repo went public, then its history was rewritten."""
import io, os

p = 'state/FACTS.md'
ADD = """| A209 | 🔐 **【把 rsna-knee-record 改成公开,然后按用户要求重写了历史】** 过程与两条必须记住的后果。<br>**① 时间线**：仓库 `ppx16/rsna-knee-record` 原为 **Private**(匿名 API 404) ⇒ 用户要求公开 ⇒ `gh repo edit --visibility public --accept-visibility-change-consequences`,并用**匿名** API 复核(200 / `private=False`)⇒ 随后用户要求把本机 SSH **私钥路径**换成占位符 ⇒ 先做**当前树**的替换(commit `f36648c`),再按要求**重写全部历史**(`git-filter-repo --replace-text`)。<br>**② 替换规则(四处,顺序即优先级)**：`C:\\Users\\<user>\\.ssh\\autodl_ed25519` → `<local-ssh-key>` · `C:/Users/<user>/.gitconfig` → `<user-home>/.gitconfig` · `C:\\Users\\<user>\\anaconda3\\Scripts\\kaggle.exe` → `$env:USERPROFILE\\anaconda3\\Scripts\\kaggle.exe`(**保持脚本可运行,占位符会让它坏掉**) · 剩余裸用户名 → `<local-user>`。<br>**③ 结果(逐项验证)**：`git filter-repo` 把 **285 个 commit 全部重写**,新 HEAD `2dfbf21`(旧 `f36648c`);`git log --all -S'yanglang'` → **NONE**;远端 `main` = `2dfbf21`,新 HEAD 的 `state/FACTS.md` blob 里 **username=False / placeholder=True**。备份留在 `D:\\kaggle\\rsna-knee-git-backup`(重写前的完整 `.git`)。<br>**④ ⚠️ 两条诚实的后果**：**(a)** `raw.githubusercontent.com` 有 **CDN 缓存** —— 刚推完的那几分钟它还在发旧内容,别拿 raw 当"仓库没更新"的证据,要看 **git blob API**;**(b)** **旧 commit `f36648c` 仍然可以按 SHA 取到**(GitHub 对不可达对象保留到 GC),⇒ **"重写历史"不等于"立刻从 GitHub 抹掉"**;真要抹掉得走 GitHub Support 请求清除不可达对象。**已如实告诉用户。**<br>**⑤ 工具**：`work/redact_local_paths.py`(幂等,当前树的替换)与 `scratch/redact-expressions.txt`(`filter-repo` 的规则表)。**⚠️ `filter-repo` 会删掉 `origin` remote**,推之前要重新 `git remote add`;第一次强推撞 **HTTP 408**(重写后所有对象要重传,~108 MB),调 `http.postBuffer=524288000` 并**放到后台跑**才成功(前台工具 10 分钟会截断,但**截断不等于失败** —— 这是 A189/A199 那条"回执丢了要问服务端"的又一次应用)。 | 2026-10-17 |
"""

if '| A209 |' in io.open(p, encoding='utf-8').read():
    raise SystemExit('REFUSING: A209 is already present')
with io.open(p, 'a', encoding='utf-8') as fh:
    fh.write(ADD)
print('appended A209; file now', os.path.getsize(p), 'bytes')
