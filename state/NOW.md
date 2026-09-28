# NOW -- 当前状态速读（**本文件由 `python scripts\now.py` 生成，不要手改**）

> ## ⚠️ 本页已过时 —— **当前状态请看仓库根目录的 `STATUS.md`**
>
> `now.py` 仍在读 `scripts/cv_queue.py --status` 与 `state/director.log`，但**那套机制已在 §164 删除**，
> 所以下面出现的队列是 **09-14** 的、事件日志来自一个**已不再运行**的组件。
> **生成时间只证明这个文件是刚写的，不证明它的内容是当前的。**
> 真正的最新状态：`STATUS.md`（一屏）→ `state/FACTS.md`（已确证/已否决）→ `docs/REVIEW-2026-09-27.md`（计划）。

生成时间：**2026-09-28 04:13 UTC**（本地 UTC+8，请换算后再读 `state/cv_push*.log` 那种本地时间戳）

> 读法：**这一页 → `state/EXPERIMENTS.md`（做过的实验台账）→ `state/FACTS.md`（已确证/已否决）→ 只在需要细节时翻 HANDOFF.md。**
> ⭐ **开新实验前先去 EXPERIMENTS.md 查一行** —— 它存在的全部目的就是「不要重复做一个实验、浪费 GPU 额度」。
> HANDOFF.md 有 9000+ 行，**每回合通读它就是上下文被烧掉的原因**。

## GPU 内核队列（来自 `scripts/cv_queue.py --status`）
```
done       ppxl16/rsna-knee-ens-v2src-radcal00-parent 2x2 CORNER 4 (HANDOFF 100.1): flat 0.60 routing AND rad_cal_w 0.00. ~20 min of a slot, then submitted in the same 00:00 UTC window as radcal00.
done       ppxl16/rsna-knee-ens-v2src-radcal00        FRONTIER ALIGNMENT (HANDOFF 87): rad_cal_w 0.40 -> 0.00, the value the current public notebook uses. ~20 min of a slot, then submit it at 00:00 UTC
done       ppxl16/rsna-knee-cv-a-fold0                A arm (12 ep) fold 0 -- pushed 09-14 08:40 UTC, ETA ~14:45 UTC
done       ppxl16/rsna-knee-cv-a-fold1                A arm (12 ep) fold 1 -- pushed 09-14 08:40 UTC, ETA ~14:45 UTC
done       ppxl16/rsna-knee-cv-gap-fold0              CV EXP #1: 2.5D window spacing +-1 -> +-2 (a SLICE-SAMPLING change; author's own measurement puts sampling at +0.006..+0.011 vs architecture +0.001). Cost MEASURED at 2.8 GPU-h, not 6.2.
done       ppxl16/rsna-knee-cv-r288-fold0             CV EXP #2: --res 288. The corpus stores 336px slices and the baseline trains at 224, discarding 30% of the linear detail; round-trip PSNR keeps climbing to 320. ~4.6 GPU-h.
done       ppxl16/rsna-knee-cv-r288-fold1             CONFIRM r288: the fold-0 screen read +0.0049 OOF / +0.0261 gold-58 (HANDOFF 116). Needs the matching a-fold baseline. 4.6 GPU-h.
done       ppxl16/rsna-knee-cv-r288-fold2             CONFIRM r288: the fold-0 screen read +0.0049 OOF / +0.0261 gold-58 (HANDOFF 116). Needs the matching a-fold baseline. 4.6 GPU-h.
done       ppxl16/rsna-knee-cv-a-fold2                A arm baseline fold 2 -- enables confirming an experiment on a SECOND fold and grows the pooled A-arm OOF. 2.8 GPU-h.
done       ppxl16/rsna-knee-cv-r336-fold1             CONFIRM r336 (fold 1): fold 0 read +0.0121 OOF 95% CI [+0.0054, +0.0194] REAL, 100% of bootstraps -- the l
```

## 最近事件（`state/director.log` 尾部，UTC 时间戳）
```
[2026-09-24T10:16:05Z] API calls this pass: polled 0 of 25 watches (25 finished -> skipped)
[2026-09-24T13:16:05Z] API calls this pass: polled 0 of 25 watches (25 finished -> skipped)
[2026-09-24T16:16:06Z] API calls this pass: polled 0 of 25 watches (25 finished -> skipped)
[2026-09-25T03:19:35Z] API calls this pass: polled 0 of 25 watches (25 finished -> skipped)
[2026-09-25T04:16:05Z] API calls this pass: polled 0 of 25 watches (25 finished -> skipped)
[2026-09-25T07:16:04Z] API calls this pass: polled 0 of 25 watches (25 finished -> skipped)
[2026-09-25T11:25:36Z] API calls this pass: polled 0 of 25 watches (25 finished -> skipped)
[2026-09-25T13:16:05Z] API calls this pass: polled 0 of 25 watches (25 finished -> skipped)
```

## 关键文件活动时间
```
state/FACTS.md                 09-28 04:06  (307943 bytes)
state/cv_targets.json          09-24 10:29  (12149 bytes)
state/cv_queue.log             09-25 07:29  (74550 bytes)
state/LB_ground_truth.csv      09-28 04:09  (3897 bytes)
```

## 开机固定动作（幂等）
**两个计划任务已经在跑，正常情况下不需要手工做任何事**：
```
RSNAKneeDirector  每 180 分钟 : download / submit / score（scripts\director.cmd）
RSNAKneeQueue     每 180 分钟 : 把空闲 GPU 槽补满（scripts\queue.cmd，日志 state\queue_task.log）
```
⚠️ 两条政策（用户 2026-09-24）：**GPU 只用来提交算成绩，实验一律走 TPU**；TPU 排队期间【坑位】也要一直填着（A28/A33）。
手工补做（开机后想立刻推进时）：
```
python scripts\director.py --once      # 交掉到期的提交、收分、记名次
python scripts\status.py               # 榜面 + 该选哪两个最终提交
python scripts\cv_queue.py --once      # 把空闲 GPU 槽补满（幂等，不会因 GPU 上限而死）
python scripts\lb_ledger.py            # 用比赛 API 重生成 LB 真值账本（否则它会悄悄过期）
python scripts\now.py                  # 刷新本文件
```

## ⚠️ Kaggle 凭据（2026-09-14 轮换过，**出 401 先看这里**）
```
机制 : KGAT_... 形式的新 token **不能**放进 ~/.kaggle/kaggle.json 的 key 字段
       （实测：2.1.2 回 401、2.2.4 直接要求重新认证）——而且**读路径与写路径要放两处**：
         读（list/status/submissions） : 环境变量 KAGGLE_API_TOKEN
         写（kernels push）            : **文件** ~/.kaggle/access_token
       两处都已配好；环境变量设在 **User 级**（重启后的新 shell / 计划任务自动继承）。
本会话注意 : 本进程继承的是旧环境，所以在本会话里跑脚本前要显式带上：
       $env:KAGGLE_API_TOKEN = [Environment]::GetEnvironmentVariable('KAGGLE_API_TOKEN','User')

排查 : `401 Unauthorized` ⇒ 环境变量；
       `Save the token to ~/.kaggle/access_token` ⇒ 那个文件（写路径）；
       `400 … CreateCodeSubmission body=None` ⇒ **先查是不是少了 `-f submission.csv`**。
       ⚠️ 这条旧记录原本写的是「⇒ 当天提交配额用尽」，**那是错的**：2026-09-27 实测在
       `Remaining today: 1` 时仍然 400，补上 `-f submission.csv` 立刻成功（`56613303`）。
       ⇒ 配额用尽的判据是 `submission-limits` 报 `Remaining today: 0`，**不是**这个 400。
```

## ⚠️ 读导演输出时的两个坑（2026-09-14 都踩过）
```
1) `director.py --once --dry-run` 可能只打一行 `another director pass is active` 就退出
   （计划任务的导演正持有锁）。**那一行 ≠ 队列是空的** —— 等一轮或看状态文件。
2) 状态的 handled 里出现 `submitfail:v<N>` **不代表提交被放弃**：那只是收件箱去重标记；
   真正的去重键是 `submit:v<N>`。**配额挡住的提交每一轮都会重试**（代码在 do_submit）。
```

## CV 折落地后（**唯一入口，不要照记忆手工做**）
```
python scripts\cv_report.py --group cva --download
```
它会：下载 → 分组判分 → gate 1 → SWA/epens 预注册检查 → arm-join → 打 **DECISION 块**。

## ⭐ 在飞 / 待判读（**下一轮先看这里**）
```
决策表（都是【写在读数之前】的预注册，不要临场改）：
  采样实验的阈值与加折条件   → HANDOFF §92.6 / §106.2
  终局两个勾选位怎么选        → HANDOFF §100.3（+ §109.2 的第四角）
  前沿对齐 radcal00 怎么读    → HANDOFF §87.4
  SWA / epoch-ensemble 判决   → HANDOFF §85.3（已判负：§105.4）

按优先级的下一步：
  1) 有折落地       → python scripts\cv_report.py --group cva --download
  2) fold-0 两跑都完 → python scripts\cv_score.py --dir downloads\cv --pair cva0 gap0 --boot 2000
  3) 槽位空出       → python scripts\cv_queue.py --once
  4) 有新读数       → python scripts\lb_ledger.py   （然后用 status.py 看名次）
  5) 推内核之前     → python scripts\preflight_queued.py   （problems 必须为 0）
```

## 常被重新推导、其实早已量过的东西 → 见 `state/FACTS.md`
语料形状 / gold-58 的来源 / 单折成本与瓶颈 / 各项误差棒 / 集成到底挂了谁 /
以及**已经否决、不要再想的方向**。

## 语料迁移准备（**GPU 空窗期做这件事**；租算力时只搬 ~15 GB 的 zip，不是 570 GB）
```
D:\kaggle\corpus\zips                      10 files     16.45 GB
D:\kaggle\corpus\data                      10 files     21.89 GB
D:\kaggle\corpus\packs                      1 files      0.02 GB
ledger state/CORPUS_MANIFEST.json      09-24 06:57  (5490 bytes)

本机 : python scripts\corpus_manifest.py build|verify|summary   /  python scripts\make_pack.py
租机 : python3 scripts/corpus_fetch.py --dest /data/corpus --layout /kaggle
       python3 scripts/corpus_manifest.py verify --root /data/corpus   # 必须 VERIFY OK
```
⭐ 关键一招（A37）：把租机目录摆成 `/kaggle/input/datasets/<owner>/<slug>/…` + `/kaggle/working/`，
**训练脚本零改动**。**不要**为本地跑分叉出“租机版脚本” —— CV 是折内配对的仪器。全文见 `MIGRATION.md`。
