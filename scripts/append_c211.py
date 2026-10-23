"""Append C211 (idempotent): the non-OAI floor was 0.006 below what it could be, so build an S6-only artifact."""
import io, os

p = r'D:\kaggle\rsna-knee\state\FACTS.md'
ADD = """| C211 | 🛟 **【补的是"地板"，不是天花板：tick② 的兜底一直是 0.944，而 nartaa 的 S6 模型是【非 OAI】且单跑报 0.950 —— 我们从来没单独交过它】**<br>**① 怎么发现的**：用户问"RSNA 今天不做吗"。我当时的答案是"没什么可做的"，但**把 50 次提交按内容过了一遍**之后发现：**没有任何一件是"只用 S6"的**。而 `57291788` 的描述里自己写着 S6 是 *"public score 0.950 alone"*，FACTS C185/C190 也记着那是 **"No OAI data was used"** 的模型。<br>**② ⇒ 这是一个真实的、被我自己漏掉的缺口**：tick②（`56878092` = 0.944）存在的唯一理由是**万一 OAI 血统被裁定不可用时的兜底**。而兜底高度一直是 **0.944**，可用的最好非 OAI 产物其实是 **0.950**（S6 单跑）。**⇒ 天花板没变（还是 0.958），但地板能抬 0.006。** 在"OAI 被禁"那个（低概率但非零的）分支里，0.944 与 0.950 是两个不同的档。<br>**③ 构建（可审计，四处改动全在同一个 cell）**：`work/build_s6_solo.py` 从 `kernel-0958-s6-s6w40` 出发——<br>· `ours = (0.6*ours.rank + 0.4*s6.rank)` → `ours = s6.rank`（**compact half 全给 S6**）;<br>· `W_RAPTOR, W_OURS = 0.5, 0.5` → `0.0, 1.0`;<br>· `W_RAPTOR_C = {...}` → `{}`（**逐 finding 的 Raptor 权重全归零**）;<br>· ⚠️ 还修了一句**会说谎的日志**：`[s6] folded ... at 0.4` 里的 0.4 是**硬编码字符串**,改了算法却没改它 ⇒ 日志会报错误权重,而日志正是我的验证依据 ⇒ 已改为 `at 1.0 (S6 only)`。**这条本身值得记：日志文本与算法分离时,前者会变成假证据。**<br>**④ 审计与失败安全**：断言"只有第 10 格变"、两个锚点各匹配 1 次、两格仍能单独 `compile()`、折入日志必须报 1.0;S6 阶段原本就是 fail-safe（失败则保留 0.958 的 submission）。<br>**⑤ 预登记**：若 S6 单跑显示 **≈0.950**，则 tick② 换成它（0.944 → 0.950）;若它显示 ≤0.944，则说明 nartaa 的"standalone 0.950"不可复现，tick② 保持不动。**两种结果都无害。** | 2026-10-21 |
"""
txt = io.open(p, encoding='utf-8').read()
if '| C211 |' in txt:
    raise SystemExit('REFUSING: C211 already present')
with io.open(p, 'a', encoding='utf-8') as fh:
    fh.write(ADD)
print('appended C211; file now', os.path.getsize(p), 'bytes')
