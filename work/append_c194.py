"""Append C194 (idempotent): the published recipe is now fully covered; every knob costs a thousandth."""
import io, os

p = 'state/FACTS.md'
ADD = """| C194 | 🎯 **【公开配方到此刻【已全部覆盖】—— 而且我们多一条 S6;同时把每个旋钮的价格量清了:在 0.958 基座上动任何一处都是 −0.001。】**<br>**① 昨天最后一个悬念有答案**：`57313021`（S6@**0.60**）= **0.958** ⇒ 与 S6@0.40 **同为 0.958** ⇒ **S6 的权重在 0.40–0.60 之间是平坦的**,0.40 够用,不必再扫。<br>**② ⭐ "镜像视图"这块【我们早就有】**：读我们基座的 Raptor 脚本（cell 2）发现它**已经在调用** `gpu_windows(vol, mask, K_EVAL, res, d, norm=NORM, **mirror=True**)` 与 `infer_pair_probs(model, plain, mirrored, device)`,并且 `_mirror_tri()` 的注释逐字写着 *"final-g1-bundle MIRROR_TTA: true L/R mirror … sagittal → reverse the 3 slice-channels; coronal/axial → flip the width. Done on uint8 BEFORE the per-channel LUT"*。⇒ **公开 0.957 配方里"两条 Raptor 臂的解剖镜像"我们【已经】在跑**（与 nartaa 的 `python work` 版逐行相同,只差输出文件名）。**⇒ 解释帖列出的组件,我们现在一个不缺,还多一条 S6。**<br>**③ ⇒ 每个旋钮的价格（全在 0.958 基座上实测,今天汇总）**：A224=0.50 → 0.957;A224=0.65 → 0.957;r9all@0.10 → 0.957;band32@0.08 → 0.957;band32@0.15 → 0.956;S6@0.40 → **0.958**;S6@0.60 → **0.958**。**⇒ 除了 S6（不动分）,其余每一个都吃掉一千分之一。0.958 是我们能从公开零件构造出来的局部最优。**<br>**④ 公开面也没新料**：按票数普查模型数据集,仍是那几件（`raptor-knee-widedense` 87 票 = 0.924 的原版 Raptor、`goodpjw` 的 2.5D reader 36 票、nartaa 的 SWA 28 票…）,**没有更新更强的独立模型可加**;而且按 C178,同族更弱的腿是负的。<br>**⑤ ⇒ 今天额度用在一个【有真实意义】的方向**：0.958 那本的 `W_RAPTOR_C` 是 **10-16 在公榜半集上拟合**出来的（C186）,而公榜半集 SE≈0.003,那个 +0.0004 无法确认、也未必迁移到私榜（**私榜才决定奖牌**）。⇒ 已建 **`rsna-knee-0958-s6-flat`** = **同一件产物但把 per-finding 权重放回全平 0.5**（= 0.9576 那版的原值）+ S6@0.40 ⇒ **"最不拟合"的 0.958 级产物**。目的不是提高公榜分,而是**在截止日多一个可选的孪生体**：一件公榜调过、一件没调过,显示分相同,由我们选哪件去赌私榜。<br>**⑥ 已装每小时自检**：`work/hourly_watch.py`（后台跑,日志 `state/kernel_watch.txt`,每 30 分钟一轮,任何内核从 RUNNING 变成 COMPLETE/ERROR 就把结论行记下来）。**它从不提交任何东西**（A193）。 | 2026-10-18 |
"""

if '| C194 |' in io.open(p, encoding='utf-8').read():
    raise SystemExit('REFUSING: C194 is already present')
with io.open(p, 'a', encoding='utf-8') as fh:
    fh.write(ADD)
print('appended C194; file now', os.path.getsize(p), 'bytes')
