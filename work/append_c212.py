"""Append C212 (idempotent): S6 alone measured 0.950, so tick 2 moved from 0.944 to 0.950."""
import io, os

p = r'D:\kaggle\rsna-knee\state\FACTS.md'
ADD = """| C212 | 🛟✅ **【S6 单跑实测 0.950 —— 与 nartaa 的文档一致;tick② 的兜底从 0.944 抬到 0.950】**<br>**① 读数**：`57478661`（S6 单跑，Raptor 权重归零、compact half 全给 S6）= **0.950**。⇒ **nartaa 写的 "standalone 0.950" 精确复现**。也因此，tick② 的兜底**一直是低报的**：我们用了三天的 0.944 不是我们最好的非 OAI 产物。<br>**② 已换票**：`work/tick2_swap.py 57478661` ⇒ `tick = ['57250245','57478661']`;`verify_final_pair.py` ⇒ **`pair present, COMPLETE, and legal`**（1 票 OAI、1 票非 OAI）。旧票 `56878092` **降级为候选人,未删除**。<br>**③ ⚠️ 换票脚本本身有一个"证据为空却通过"的缺陷（已修）**：它把旧票分数读自 `candidates[ref]['public']`，而那个字段没存 ⇒ `float(None or 0)` = **0.000** ⇒ 比较变成"0.950 > 0.000",**结论对但依据是假的**。已改为**从服务端读旧票分数**，读不到就**拒绝执行**（而不是拿 0.0 顶上）。**这是本项目第 N 次"检查通过但没真正看见"**，和 README 开头那条规则同源。<br>**④ 审计工具当场验证了自己有用**：换票后立刻重跑 `work/audit_docs_tick.py`，它**立刻抓出三处**活文档仍写着旧票 `56878092`（`DEADLINE-CHECKLIST.md` 的票表、`FINAL-TICK.md` 的"勾选按上表"与"②必须保持非 OAI，56878092 继续当锚"）。全部改完后 **exit 0**。⇒ **"改真源 → 审计文档"这个顺序现在是有工具保障的,不是靠自觉。**<br>**⑤ ⇒ 最终两票**：**① `57250245`（0.958, OAI）** + **② `57478661`（0.950, 非 OAI）**。<br>**下行**：若 OAI 被禁,按 0.950 计（此前是 0.944）;**上行**：0.958。 | 2026-10-22 |
"""
txt = io.open(p, encoding='utf-8').read()
if '| C212 |' in txt:
    raise SystemExit('REFUSING: C212 already present')
with io.open(p, 'a', encoding='utf-8') as fh:
    fh.write(ADD)
print('appended C212; file now', os.path.getsize(p), 'bytes')
