"""Append C196 (idempotent): the 224 arm had no mirror -- a real missing component, now ported."""
import io, os

p = 'state/FACTS.md'
ADD = """| C196 | 🎯🎯 **【找到一块【真的缺】的配方组件：224 臂没有解剖镜像】C194 我只验了 384 臂就下结论"镜像我们早就有"——**那是不完整的**：逐个 cell 数过之后,384 臂有 `mirror=True`/`_mirror_tri`/`infer_pair_probs`,而 **224 臂三者计数全为 0**。**<br>**① 外部证据（逐字）**：`varshithakolupuri/rsna-0-956-one-free-flip-68-ranks-up`:<br>*"nartaa's 384-pixel CoAtNet **already averages two views** … The 224-pixel checkpoint … **was published with one view only**. This notebook gives it the mirror too, with nartaa's own mirror code."*<br>效果逐字：**显示分不变 0.956,但 rank 357 → 289**,"from bronze back into silver"。<br>**② ⇒ 为什么这块现在格外值钱**：我们的 0.958 档有 **~1,155 支队**,我们在榜上约第 **207** 位,**银线在 ~70 名之前**。⇒ **一个"档内前移"的改动就是真金白银**,而这正是公榜能给出的唯一可行动信号（C184 的噪声结论针对的是"显示分的千分之一",不针对"同一档内的名次"）。<br>**③ ⚠️ 纠正 C194 的说法**：C194 写"解释帖列出的组件一个不缺" —— **这句只对 384 臂成立**。正确表述：**384 臂已有镜像;224 臂没有;C197 起两臂都有**。<br>**④ 移植方式（写成可审计的）**：先 diff 我们基座的 224 脚本（cell 4, 95,469 字）与 flip 那本的 224 脚本（cell 9, 96,577 字）——差异**恰好是两类**：**(a)** 镜像块（`SLOT_SIZES`/`SAGITTAL_SLOTS`/`_slot_index`/`_mirror_tri`、`gpu_windows(..., mirror=)` 参数、以及推理处 `0.5*(infer_probs(xw) + infer_probs(xm))`）;**(b)** 我们自己的 `raptor224_raw.csv` 三行 dump（blend 需要,flip 那本没有）。<br>⇒ **构建器 `work/build_0958_m224.py`：取【他们的】224 脚本,把我们那三行 raw dump 按 `[gpu] done` 这一行（两版逐字相同）插回去**;审计断言：**9 格里只有第 4 格变**、去掉那三行能逐字还原他们的脚本、且基座的 224 臂原本确实没有镜像（否则拒绝构建）。⇒ 新内核 `ppxl16/rsna-knee-0958-m224`,**写出的 224 脚本单独 `compile()` 通过**（`writes sub: True | writes raw: True | mirror: True`）。<br>**⑤ 失败安全**：224 阶段本来就是"时钟不够就跳过、失败就不动 blend"的（`R224_SKIPPED`/超时保护）,所以这一枪若出问题,成品与今天的 0.958 完全一致。 | 2026-10-19 |
"""

if '| C196 |' in io.open(p, encoding='utf-8').read():
    raise SystemExit('REFUSING: C196 is already present')
with io.open(p, 'a', encoding='utf-8') as fh:
    fh.write(ADD)
print('appended C196; file now', os.path.getsize(p), 'bytes')
