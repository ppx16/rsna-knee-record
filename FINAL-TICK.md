# FINAL TICK — moved

> ⚠️ **本文件已于 2026-10-21 收敛，内容不再维护。** 它与 `docs/FINAL-TICK.md` 曾经**各有一份、并且已经漂移了 49 行** ——
> 两份"最终指令"是这类项目里最不该存在的结构：照哪一份做，取决于你先打开了哪一个。
>
> # ⇒ 唯一权威：**[`docs/FINAL-TICK.md`](docs/FINAL-TICK.md)**
>
> 而 `docs/FINAL-TICK.md` 本身也**不存任何真相**：它只描述动作。真正的唯一真源是
> **[`state/final_pair.json`](state/final_pair.json)**，校验靠：
>
> ```powershell
> python work\deadline_report.py       # exit 0 才允许勾选
> ```
>
> 那条命令会用**实时服务端**复核配对、检查两个仓库是否已推送、读榜，并打印两条 ref 的
> **描述开头与结尾**（页面只能靠文字认，因为我们有二十多条显示 0.942/0.943 的提交，
> 而平台的默认勾选不是我们这一对）。
>
> （这条收敛是 `work/audit_docs_tick.py` 的产物。它先后抓到：`FINAL-TICK.md` 里"退回 0.954"的陈旧兜底、
> 一句与现行策略相反的"① 永远是最高分"，以及这一处双份漂移。）
