"""Append C213 (idempotent): the competition settings settle four open questions at once."""
import io, os

p = r'D:\kaggle\rsna-knee\state\FACTS.md'
ADD = """| C213 | ⚙️✅ **【比赛设置有四个字段,一次关掉四个未知量 —— 其中两个是最后一天才敢确认的】** 来源：`competitions.CompetitionService/GetCompetition`（服务端，非文档）。<br>**① `numScoredSubmissions = 2`** ⇒ **只有 2 条提交会被计分**，正是我们一直在准备的两票结构。**⇒ 勾选不是"锦上添花",而是决定这两票是谁。**<br>**② ⭐ `usesSynchronousReruns = True`** ⇒ **重跑发生在【提交时】,不在截止时**。⇒ **截止不会再次重跑内核** ⇒ **"数据集被别人更新导致我们重跑时取到不同权重"和"重跑超时"这两个风险都不存在**;已经出分的 0.958 / 0.950 **就是最终值**,勾选只是指定哪两条生效。**这是我在最后一天最想确认的一条。**<br>**③ `scoreTruncationNumDecimals = 3`** ⇒ 显示分**截断**到 3 位小数（不是四舍五入）。⇒ 与我们反复观察到的"0.958 是一个宽档"完全一致,也解释了为什么档内名次不稳定。<br>**④ ⭐ `leaderboardPercentage = 30`** ⇒ **公开榜只用 30% 的隐藏测试集,私榜用 70%**。隐藏集 1322 ⇒ 公开侧约 **397 个 study**、私榜约 **925** 个。**⇒ 两件事**：**(a)** 我们此前引用的"公开半集 SE≈0.003"其实**偏乐观**（样本比设想的小）⇒ **公开榜的可分辨力比我们以为的更差**,"一千分之一不可测"这个结论只会更成立;**(b)** **私榜样本更大 ⇒ 私榜是更可靠的测量**,而公开档位在私榜上会明显重排。⇒ **公开名次变差不代表结果变差**,这一点在 C206 已写过,现在有了机制层面的解释（30% 的小样本噪声 + 70% 的重新排序）。<br>**⑤ 另外**：`maxDailySubmissions = 5`、`onlyAllowKernelSubmissions = True`、`requiredSubmissionFilename = submission.csv`、`maxGpuRuntimeMinutes = 540`（与我们的内核运行时长吻合）。 | 2026-10-22 |
"""
txt = io.open(p, encoding='utf-8').read()
if '| C213 |' in txt:
    raise SystemExit('REFUSING: C213 already present')
with io.open(p, 'a', encoding='utf-8') as fh:
    fh.write(ADD)
print('appended C213; file now', os.path.getsize(p), 'bytes')
