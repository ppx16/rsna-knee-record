"""Append C190 (idempotent): 10-17 state, band inflation, and the manifest-driven S6 route."""
import io, os

p = 'state/FACTS.md'
ADD = """| C190 | 📉📌 **【0.958 档 13 小时从 326 涨到 621 —— 我们 rank 231→419、银余量 +73→−114】而公开面仍是 `none above ours`。** 另外,S6 腿的接入路径被查清为**manifest 驱动**,不需要改推理代码。<br>**① 读数**（`work/daily_rsna.py`,板 2026-10-17T01:30Z,N=6113）：**rank 419 / 0.958**;铜线 611 ⇒ **+192 在内**;银线 305 ⇒ **−114 在外**;我们所在的 **0.958 档 = ranks 131..751（n=621）**,档内 **289/621**。<br>**② 与昨天同一时刻对照**：昨天 rank 231 / 档 n=326 / 银 +73。⇒ **24 小时里档涨了 295 支队伍,我们掉了 188 名。** 这是 C187 那条机制的继续:**分数没变,是榜面在涨**。⇒ **0.958 也已经不再是安全位置**;要稳必须 **0.959+**,而公开面天花板今天是 0.958（`none above ours`）。<br>**③ ⇒ 结论（写死）**：**公开面已经无料可抄**（我们持有天花板）。剩下 5.9 天里唯一还能自己造的上行,是**配方级改动**（S6 全量腿 / 镜像视图）,而不是继续扫权重（C184/C189 已证那在噪声量级）。<br>**④ ⭐ S6 全量腿的接入路径（今天的实质进展）**：读基座 cell 6 发现 **compact worker 的全部参数都来自 bundle 的 `manifest.json`** —— `WORKER_ARGS` 是 `--backbone/--size/--crop/--eval-stride/--views/--slices/--phys-mm`（并按 manifest 选配 `--state-spec` / `--side-model` / `--models2`）<br>⇒ **加 S6 臂不需要改推理代码**,只要:给它**写一份自己的 `manifest.json`**（`backbone convnext_small, size 320, crop 288, eval_stride 1, views v4, slices 3, phys_mm 3.5`,models=[`E_prior_s42_fall.pt`],arm `E_prior`）,然后**用同一套 stage-2 runner 再跑一次**（`--bundle <s6 目录>`）,再把两枝并进 compact half:`compact = 0.60*rank(现役) + 0.40*rank(S6)`。<br>**⑤ 但有一个真障碍**：`s6_extras.bin` 今天**下不下来**（`storage.googleapis.com` 连续 5 次 `SSL_EOF`,A189 同族）。⇒ **下次要么重试,要么直接让 Kaggle 内核在运行时解包**（数据集是挂载的,不需要本地下载）—— 后者本来就是计划,所以**这条障碍不阻塞**。<br>**⑥ 今天交的三枪（额度 3/5）**：`rsna-knee-0958-blend-a50`（A224=0.50）、`-a65`（A224=0.65）、`rsna-knee-0958-r9all-w10`（0.958 基座 + r9all 腿 @0.10）。⚠️ **按 C189,这三枪的先验都是"测不出"**,它们的作用是把 0.958 基座上【还没被扫过的两个旋钮】各问一次,顺带留下降级余量。 | 2026-10-17 |
"""

if '| C190 |' in io.open(p, encoding='utf-8').read():
    raise SystemExit('REFUSING: C190 is already present')
with io.open(p, 'a', encoding='utf-8') as fh:
    fh.write(ADD)
print('appended C190; file now', os.path.getsize(p), 'bytes')
