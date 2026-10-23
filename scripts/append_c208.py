"""Append C208 (idempotent): compliance verified, and the public bronze margin is on track to go negative."""
import io, os

p = r'D:\kaggle\rsna-knee\state\FACTS.md'
ADD = """| C208 | 🧾⚠️ **【截止前 46 小时的两件事：合规检查通过;但公开榜的铜线余量按趋势会在截止时【转负】】**<br>**① 合规（已核对）**：我们提交的两个内核（`kernel-0958-blend` = tick① 的来源、`kernel-0958-s6-s6w40`）的 markdown 里 **OAI/NDA 致谢声明齐全**：`Osteoarthritis Initiative` ×1、`NDA` ×3、`NIMH` ×1、`OAI` ×2、`acknowledg` ×2 ⇒ **不是裸用 OAI 数据**,与公开配方的声明方式一致。**⇒ 这一项没有剩余风险。**<br>**② ⚠️ 铜线余量的趋势（必须如实说）**：<br>| 日期 | rank | 铜线余量 | 0.958 档 n |<br>|---|---|---|---|<br>| 10-18 | 360 | **+257** | 1072 |<br>| 10-19 | 383 | **+236** | 1155 |<br>| 10-20 | 458 | **+123** | 1313 |<br>| 10-20 晚 | 504 | **+95** | 1382 |<br>| **10-21** | **534** | **+95** | **1444** |<br>48 小时里余量掉了约 141 ⇒ **−2.9/小时**;按这个速度,**46 小时后（截止时刻）余量 ≈ −38 ⇒ 公开榜上会掉出铜牌区**。<br>**③ 但这不等于丢奖牌,理由要写清楚**：奖牌按 **private split** 定,不是公开榜的截止名次。公开榜的掉线只反映**档内与他人的未取整分比较**（C206：重跑噪声 ±150）,而私榜是**另一份数据上的重新排序**。⇒ **正确表述：公开榜是代理,代理正在变差;真正的判定在私榜。**<br>**④ ⇒ 有没有可做的事？** 按 C206/C207,**没有**:五件 0.958 产物在公榜上无法区分,换 tick 不会改变档内位置（我们甚至不知道哪件的未取整分更高）。**唯一能改变档内位置的是"更高的未取整分",而那需要更强的基座,公开面 46 小时内没有出现**。<br>**⑤ ⇒ 这段要写进交接和截止清单**:截止日的动作不变（勾 `57250245` + `56878092`）,但要**明确告诉用户"公开榜名次可能变差,那是代理而非结果"**,免得那天看到掉出铜线而误以为要换策略。 | 2026-10-21 |
"""
txt = io.open(p, encoding='utf-8').read()
if '| C208 |' in txt:
    raise SystemExit('REFUSING: C208 already present')
with io.open(p, 'a', encoding='utf-8') as fh:
    fh.write(ADD)
print('appended C208; file now', os.path.getsize(p), 'bytes')
