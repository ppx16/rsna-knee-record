"""Append C198 (idempotent): the 384 arm had the same waste, at four times the weight."""
import io, os

p = 'state/FACTS.md'
ADD = """| C198 | 🎯🎯 **【同一处浪费在 384 臂上,而它的权重是 224 臂的四倍】** C197 给 224 补上 plain/TTA 拆分之后,回头读 384 臂的 `infer_pair_probs`:<br>```python<br>def infer_pair_probs(model, plain, mirrored, device):<br>    p = infer_probs(model, plain, device)<br>    q = infer_probs(model, mirrored, device)<br>    return 0.5 * (p + q)        # ← p 被丢掉<br>```<br>⇒ **同一个"算出来又扔掉"的浪费,而 384 臂占 Raptor half 的 0.65（224 是 0.35）、占最终 0.325（224 是 0.175）** ⇒ **同样的改动,收益面是四倍。**<br>**① 外部依据（同一条）**：`laimerbotrolenk/rsna-four-model-ensemble` 的表里 384 那一行也是 **TTA 32.5%**,而它给 224 的两半各 8.75% ⇒ **他们的 384 臂没有拆** —— 所以我们这一步是**在公开配方之上再走一步**,先验未知（可能是正的,也可能因为 384 的 TTA 更稳而被拆坏）。**这正是需要一次提交来测的原因。**<br>**② 构建器 `work/build_0958_plain384.py`（六处补丁,每条断言只匹配 1 次）**：**(A)** `infer_pair_probs` 改成 `return 0.5*(p+q), p`;**(B)** 唯一调用点 `_avg, _pl = infer_pair_probs(...)`;**(C)** 384 脚本加 `arm_plain` 数组并 dump `raptor_plain.csv`;**(D)** blend 里先载入它（缺失/不符则不用），**再**在作者公式之前把 384 的 TTA 秩与 plain 秩 0.5/0.5 平均。⇒ **作者表达式一个字没改,等价于"把 384 的份额对半"**。<br>**③ 基线选择**：新内核 `ppxl16/rsna-knee-0958-p384` **建立在 `m224p` 之上**,所以它同时拥有 **384 plain/TTA 拆分 + 224 plain/TTA 拆分 + 两臂镜像**,而 compact 侧保持 R6（无 S6,与 tick 一致）。审计：**只有第 2、8 格变**,两格都能单独 `compile()`。<br>**④ 失败安全**：plain 文件缺失/行列不符/读异常 ⇒ 打印 `[s9c] ... not used`,blend 与 `m224p` 逐字节相同。 | 2026-10-19 |
"""

if '| C198 |' in io.open(p, encoding='utf-8').read():
    raise SystemExit('REFUSING: C198 is already present')
with io.open(p, 'a', encoding='utf-8') as fh:
    fh.write(ADD)
print('appended C198; file now', os.path.getsize(p), 'bytes')
