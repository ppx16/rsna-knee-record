"""Append C177 + A204 (idempotent)."""
import io, os

p = 'state/FACTS.md'
ADD = """| C177 | ⭐⭐ **【band32 腿第一次跑失败,但失败得有信息量】根因是腿自己的【成员数守卫】:`[X] expected 3 models, found 10` ⇒ 它默认要 3 个成员,而数据集里有 10 个 ⇒ 退出码 1 ⇒ 我们的失败安全把基座文件原样还原(没有任何损失)。**<br>**① 决定性读数**（`_band32_child.log`,我们自己那次 Save 跑的产物）逐字:<br>```<br>[model] + rsna-knee-band32-ours   small crop 130 mm, val 0.8728   ... (共 10 个成员, val 0.8728–0.8880)<br>[model] n_slices 16 -> 32, taken from the checkpoint.<br>[model] ensembling 10 model(s) over crops [130.0] mm<br>[X] expected 3 models, found 10. Attach Notebooks -> all members listed in this kernel's inputs.<br>    Refusing to write a submission that would look valid and score wrong.<br>```<br>**② 为什么会有这个守卫**（作者在他自己的笔记本里写明了）:*"a loader that silently assembles 2 of 3 members writes a valid-looking submission that scores wrong — so the number is DECLARED, never inferred from what was found."* ⇒ **它要的是环境变量 `RSNA_N_EXPECTED`**,而作者的笔记本是从**他附件里的 `members.json`** 读出这个数再传进去的。<br>**③ ⇒ 修法（已落地,`work/build_c224_band32.py` v2）**：在起子进程之前,**从 `/kaggle/input` 下按有界深度找到 `members.json`、读出 `len(members)`、写进 `RSNA_N_EXPECTED`** —— **声明而不是猜**（与作者的规矩一致）。⇒ 三个内核已重建（w15/w30/w08,sha 25501a8d / 826bc255 / 9d615cea）。<br>**④ ⚠️ 服务器上 band32-w15/w30 的 v1 是坏的**（跑完但腿回落到基座）⇒ **要用 v2 重推,不要在 v1 上花提交额度**。`w08` 在修好之前还没推上去 ⇒ 它推上去就是修好的版本。<br>**⑤ 顺带拿到的独立读数（以后有用）**：这 10 个成员是 **DINOv2-small + slot attention,130 mm 裁剪,32 切片**,自报 val macro AUC **0.8728–0.8880**（他自己的验证集,不是公榜）⇒ **这是一条真在 0.87–0.89 的腿**,与我们基座(0.956)差得远,但**架构族完全不同**。 | 2026-10-14 |
| A204 | ⚠️ **【全零下行也要付出"额度以外"的代价】今天 band32 腿的第一版失败,`w15/w30` 两个 GPU 槽白跑了约 8 分钟,而且 `w08` 还排在队里。**<br>**① 为什么能提前拦住而我没拦住**：腿在**加载完 10 个检查点之后**才因为成员数守卫退出 —— 也就是说**失败发生在"读权重"之后**,不只是读参数那一步。**⇒ 如果我在把任何 kernel 交给 GPU 之前,先在本地把它的"环境变量契约"逐条核对一遍**(这里:腿要 `RSNA_OUT`/`RSNA_SPLIT`/**`RSNA_N_EXPECTED`** 三个),这一次就能省下。**<br>**② 写死的规矩**：**移植别人的腿时,把它的源码里所有 `os.environ` / `getenv` 的键【列一张表】,逐个决定取值来源** —— 这不是"读一遍代码",而是"列一个契约清单"。今天我就是**只看到 `RSNA_OUT`/`RSNA_SPLIT` 就以为齐了**（与 A198/C164"局部正确 ≠ 全局可执行"同族,第四次）。<br>**③ 好消息是失败安全按设计工作了**：`band32_leg_report.json` 写着 `fallback to the base file`,而 `_base_submission.csv` 与最终 `submission.csv` 一致 ⇒ **没有损失,也没有把坏产物交上去**。 | 2026-10-14 |
"""

if '| C177 |' in io.open(p, encoding='utf-8').read():
    raise SystemExit('REFUSING: C177 is already present')
with io.open(p, 'a', encoding='utf-8') as fh:
    fh.write(ADD)
print('appended C177, A204; file now', os.path.getsize(p), 'bytes')
