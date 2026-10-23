"""Append A208 (idempotent): two attached leg datasets make 'the first members.json' ambiguous."""
import io, os

p = 'state/FACTS.md'
ADD = """| A208 | ⚠️⚠️ **【挂两条腿之后,"找到第一个 `members.json`"就变成了【猜】—— 三叠加那一枪的 band32 阶段因此拿着 r9all 的清单(1 个成员)去跑,腿自己拒绝,整阶段回落到基座。】**<br>**① 症状**（`band32_leg_report.json`）：`"status": "fallback to the base file: AssertionError: the leg exited with code 1"` 且 **`"n_expected": 1`** —— 而 band32 有 **10** 个成员。**"n_expected=1" 就是决定性线索**：那个 1 只可能来自**另一个**数据集 `bliverrigpdoge/rsna-knee-leg-r9all` 的 `members.json`（它确实只列 1 个成员）。<br>**② 根因**：我写腿的 runner 时,输入发现是"按有界深度扫描 `/kaggle/input/**,members.json`,取**第一个**"。**单个腿数据集时它是对的（A204 那次修好了 n_expected=3→10）,两个腿数据集时它变成了掷骰子。** ⇒ **单变量验证过的代码,在多输入场景下会退化**（与 A198/C164 同族,但这次是"我的修法本身引入了新的脆弱性"）。<br>**③ 修法（已落地,`build_c224_band32.py` 的 RUNNER）**：**用内容把它钉住** —— 遍历所有候选 `members.json`,对每一个数**它自己那棵子树下的 `model_v3.pt` 个数**,取"能对上 checkpoint 的那个"（并把 `members_json` / `model_v3_files` 写进报告,便于事后核对）。⇒ 这条规矩可以一般化：**当输入里有多个同名文件时,判据必须用"它与目标内容的关联",不能用"它在扫描顺序里靠前"。**<br>**④ 失败安全按设计工作了**：那一枪（`legs2 v1`）交上去也只会等于"r7ds + r9all@0.10",**没有坏产物**;修好的 v2（sha `8778d926…`）已强制推上去重跑。 | 2026-10-15 |
"""

if '| A208 |' in io.open(p, encoding='utf-8').read():
    raise SystemExit('REFUSING: A208 is already present')
with io.open(p, 'a', encoding='utf-8') as fh:
    fh.write(ADD)
print('appended A208; file now', os.path.getsize(p), 'bytes')
