"""Append C192 (idempotent): the S6 all-data arm is IN, and how it was fixed."""
import io, os

p = 'state/FACTS.md'
ADD = """| C192 | 🎉🎉 **【S6 全量腿接通了 —— 公开配方里我们唯一缺的那条腿，而且是【非 OAI】血统】** `S6_STAGE_COMPLETE {"status": "PASS", "arm": "E_prior", "rows": 3, "scored": 3, "fallback_studies": 0, "seconds": 70.1}` + `[s6] folded into the compact half at 0.4 over 3 studies`。<br>**① 两个 bug 逐一定位并修掉（C191 之后的第二次、第三次尝试）**：<br>· **v1 `ALL_FALLBACK`**：extras 的代码只通过 `--code` 传给了控制器，**没有进子进程的 `PYTHONPATH`** ⇒ worker 按模块名导入时找不到 ⇒ 每个研究都失败。**修法**：`PYTHONPATH = _s6_code : CODE : ...`，**并加了 worker 日志捕获**（这一次性的诊断设施直接决定了第二次能不能修）。<br>· **v2 仍然 `ALL_FALLBACK`，但 worker 日志给出了逐字根因**：`ValueError: views v3k need --anchor-weights (SKM-TEA checkpoint) to rebuild the knee anchors at inference`。同时日志里 `MODEL_READY {"models": 1, "spatial_models": 1, "crop": 288}` ⇒ **权重与几何都已经对了,只差一个参数**。**修法**：在 `/kaggle/input` 下按**钉住的 sha256**（`5904dd16…`）找 `anatomy.pt` 并传 `--anchor-weights`。<br>**② ⇒ 第三次就通了。** 现在的成品：基座（0.958）+ S6 全量腿，`compact = 0.60*rank(R6) + 0.40*rank(S6)`，再做**未改动**的 `final = 0.5/0.5`。**这正是公开解释帖给出的权重**。<br>**③ 为什么这条腿值得单独记**：**(a)** 它是公开 0.957 配方里我们唯一缺的组件;**(b)** README 明确 **"No OAI data was used for this model"** ⇒ 在我们全 OAI 血统的 0.958 里掺进一条**非 OAI** 的腿,**同时是对主办方裁定未决那条风险的对冲**（C165）;**(c)** 它单独 0.950,**比 band32（0.87–0.89）和 r9all（0.942）都强**,是三条腿里唯一"既新又不太弱"的。<br>**④ 失败安全按设计工作了两次**：v1/v2 都打印 `S6_STAGE_FALLBACK` 并把成品留在原来的 0.958 —— **两次失败没有产生任何损失,也没有浪费任何提交额度**（我只在 v3 真的 PASS 之后才提交 `57291788`）。<br>**⑤ 方法论（写死）**：**"移植一条腿"的调试顺序应当是【先让 worker 日志可达,再猜原因】**。v1 我猜"PYTHONPATH"是对的,但如果 worker 日志一直拿不到,v2 那个 `--anchor-weights` 的根因就只能靠再猜 —— **今天真正的转折点是"把 worker 日志复制进输出"这一步**（与 A186/A190 同族:**判据要从能读到那个量的源里去读**）。 | 2026-10-17 |
"""

if '| C192 |' in io.open(p, encoding='utf-8').read():
    raise SystemExit('REFUSING: C192 is already present')
with io.open(p, 'a', encoding='utf-8') as fh:
    fh.write(ADD)
print('appended C192; file now', os.path.getsize(p), 'bytes')
