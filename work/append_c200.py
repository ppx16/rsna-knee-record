"""Append C200 (idempotent): the public ceiling moved to 0.9586 and it is a reader leg, not a weight."""
import io, os

p = 'state/FACTS.md'
ADD = """| C200 | 🎉🎉 **【公开天花板又抬了：`heliosli/…-0-9586` = 0.9586 —— 而且它加的是 goodpjw 的 2.5D ConvNeXt MIL reader】** 这是我们**错过的一整档**（0.9586 显示即 **0.959**）。**起因是一次"按精确 slug 重查"** —— 我之前那份 0.957+ 清单里是**猜的 slug**（两个拉取失败），改成从 `ListKernels` 精确取 slug 之后，0.958 一族从 3 本变成 **5 本**，其中就有它。<br>**① 它的构成（逐字读它的 md 与 cell 6/9）**：<br>· 基座 = 我们熟悉的 0.958 配方：Raptor-384（解剖镜像）+ Raptor-224（**单视图**）+ R6 包（`fe055406`,**flip TTA**）;<br>· ⚠️ **但 `W_RAPTOR_C` 是【平】的**（`{'Medial OA':0.5,'Lateral OA':0.5,'PF OA':0.5}`）—— 也就是说 **heliosli 自己把 10-16 那次 per-finding 重加权撤掉了**,改用 reader 腿来顶。**这与我们 C195 的"重加权值 0.001"并不矛盾**（在他那一版里,重加权 + reader 一起给出的总数是 0.9586;哪一半贡献多少没有拆开）;<br>· **新增 cell 9 = "Part B (V2): goodpjw2008's 2.5D ConvNeXt MIL reader"**：把他的三个源文件**逐字**写到 `/kaggle/working/own_src/`（sha256 校验），然后**原样跑他的 runner**,按**他自带的逐 finding 权重（0.003–0.035）**秩融合进 stage-3 的 submission;**失败或时间不够就保持原状**。<br>**② ⇒ 这正是 C159 当年关掉的那条"reader 腿",但换了新一代**：C159 关的是 `goodpjw2008/rsna-knee-2-5d-convnext-reader` 那个**旧三联**在 0.943 基座上无增益;而 goodpjw 现在发表了 `rsna-knee-coatnet-r6-2-5d-mil-lb-0-958`（**0.958**）,heliosli 把它接进 0.958 基座得到 **0.9586**。⇒ **"reader 腿没用"这个结论要按版本限定**,旧版不成立不代表新版不成立。<br>**③ 复现的两个门槛（都已处理）**：**(a)** 它 `C_SLUG = 'rsna-knee-r6-frozen-fe055406'` 且 `BUNDLE_ID = 'fe055406fad6a12f'` ⇒ **必须 v1**,而 fork 只拿得到最新版 v2(R7)（C182）⇒ 已用 `work/patch_09586_slug.py` 把 **C_SLUG 换成 `…-fe055406-w1`**（单版本公开副本,我们的 0.958 基座就用它）;**只改这一行**,并逐格审计;**(b)** 它的 `dataset_sources` 里还挂了 `goodpjw2008/rsna-knee-2-5d-convnext-reader`（公开 ✓）与 `kernel_sources: [goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958]` —— **kernel_sources 原样继承**,因为 cell 9 的源码是内嵌字符串,但 runner 可能从那份挂载里取。<br>**④ 已 fork 并推**：`ppxl16/rsna-knee-09586`（notebook 646,285 B）。**预登记：≥0.959 ⇒ 接管 tick①**;<0.959 ⇒ 平台保最好,仍不损失。 | 2026-10-19 |
"""

if '| C200 |' in io.open(p, encoding='utf-8').read():
    raise SystemExit('REFUSING: C200 is already present')
with io.open(p, 'a', encoding='utf-8') as fh:
    fh.write(ADD)
print('appended C200; file now', os.path.getsize(p), 'bytes')
