"""Append C197 (idempotent): the 224 plain view was being computed and thrown away."""
import io, os

p = 'state/FACTS.md'
ADD = """| C197 | 🎯 **【又一块有出处的组件：224 的 plain 视图本来就算出来了却被丢掉】** 来源 `laimerbotrolenk/rsna-four-model-ensemble`(0.957) 的 V33，逐字：<br>*"TTA is the arithmetic mean of the plain and mirrored **probabilities**, then ranked per finding. **Mixing these ranks is not equivalent to changing the probability average to 75/25.**"* … *"The plain 224 forward **already runs** in the supplied code. Saving it separately adds **no extra forward passes or checkpoints**."*<br>它的配比：`Raptor-384 TTA 32.5% + Raptor-224 **plain** 8.75% + Raptor-224 **TTA** 8.75%`（= 384 占 0.65，224 的 0.35 **对半分**成 plain/TTA 两半）。<br>**① ⇒ 我们 C196 刚补上镜像之后，同一个地方还剩一个浪费**：`gpu_windows(..., mirror=True)` 那条分支里 `infer_probs(model, xw, d)` 的**结果被内联丢掉了**，只留下 `0.5*(plain+mirror)`。⇒ **plain 是一个免费的第二信号**，而且它与 TTA 的**秩混合 ≠ 概率平均**。<br>**② 构建器 `work/build_0958_m224p.py`（四处外科补丁,每条都断言只匹配 1 次）**：**(A)** 224 脚本多一个 `arm_plain` 数组;**(B)** TTA 分支把 `_p = infer_probs(model, xw, d)` **存下来**（不增加任何前向）;**(C)** 多写一个 `raptor224_plain.csv`;**(D)** blend 格里**先**把 224 的 TTA 秩与 plain 秩按 0.5/0.5 平均，**再**套用作者原来的 `(1-A224)*r384 + A224*r224` —— 这等价于"把 224 的份额对半分"，而**作者的公式一个字没改**。<br>**③ 失败安全**：plain 文件缺失 / 列行不符 / 读取异常 ⇒ 打印 `[s9b] ... not used` 并把 blend 保持得与 `m224` **逐字节相同**（只在 plain 可用时才做那 4 行替换）。审计断言：**只有第 4、8 格变**，且两格**都能单独 `compile()`**。<br>**④ 顺带的方法论**：这条和 C196 都来自**读别人笔记本的 markdown 表格**，不是猜。⇒ 已写 `work/audit_components.py`：批量拉 0.956–0.958 的公开笔记本，只打印描述"它加了什么"的散文行（按 mirror/flip/TTA/crop/views/slice/checkpoint/weight 过滤）,**不执行、不提交** —— 下次找组件先用它,而不是凭印象。 | 2026-10-19 |
"""

if '| C197 |' in io.open(p, encoding='utf-8').read():
    raise SystemExit('REFUSING: C197 is already present')
with io.open(p, 'a', encoding='utf-8') as fh:
    fh.write(ADD)
print('appended C197; file now', os.path.getsize(p), 'bytes')
