# DEPLOY — AutoDL 上跑 `msft2-long`（CoAtNet 12 epoch）：开机前清单

> **为什么要这一步**（三个都不是推测）：
> ① **Kaggle GPU 周额度 30/30 已用尽**，10-10 才刷新；② 我们要的臂是 **CoAtNet**，而 **D17 实测它在 torch_xla 上训不了** ⇒ **TPU 不是备选**；③ 于是 **AutoDL 是 10-10 之前唯一能训 CoAtNet 的地方**。
>
> ⚠️ **本文是清单，不是执行记录。** 盒子**已关机**，以下每一步都要在开机后**逐条核实**，不要假定。

---

## 0. 成本与时间（先说清，避免开一个空转的盒子）

| 项 | 值 |
|---|---|
| 单价 | ￥2.88/h（**空闲也计费**） |
| 余额 | ~**￥27.94 ≈ 9.7 h**（这是上限，不是预算） |
| 预计训练 | msft2 的 4 epoch 在 T4 上是小时级 ⇒ **12 epoch 在 4090 上大概率 1–2 h** |
| 预计总计（含部署与验证） | **2–3 h ≈ ￥6–9** |
| ⚠️ 关机 | **没有 API，只能你手动去控制台点**（开放 API 只有 `/instance/pro/*`） |

---

## 1. 开机后先【只核实、不训练】——五分钟省掉一次白跑

```bash
ssh -i ~/.ssh/autodl_ed25519 -p 34769 root@connect.cqa1.seetacloud.com
```

| # | 核实什么 | 命令 | 期望 |
|---|---|---|---|
| 1 | **语料是否还在** | `ls -la /kaggle/input/datasets/dreaddevelopment/knee-raptor-corpus/all_vols.npy /kaggle/input/datasets/dreaddevelopment/knee-raptor-corpus-ext/extra_vols.npy` | **15,895,756,928 B** 与 **5,995,680,896 B**（若不在，见 §2 备选） |
| 2 | **Raptor 训练器是否已在** | `ls -la /root/*.py /root/rsna* 2>/dev/null; find / -maxdepth 3 -name 'knee-mri-training*' 2>/dev/null` | 我上次只看到 `arm3d_*.py` ⇒ **大概率不在**，要传 |
| 3 | **fold 文件 / 标签是否已在** | `find / -maxdepth 4 -name 'fold_file.json' -o -maxdepth 4 -name '*label*.parquet' -o -maxdepth 4 -name 'report_labels_v4hybrid.csv' 2>/dev/null` | 不确定 |
| 4 | **v5 检查点是否已在** | `find / -maxdepth 4 -name 'raptor_ft_coatnet_v5_full_swa.pt' 2>/dev/null` | 不确定 |
| 5 | **磁盘余量** | `df -h /root/autodl-tmp` | 语料 ~22 GB 已在，别的地方要留几 GB |

⚠️ **第 4 项若不在，必须传** —— 它是 `msft2` 配方的 **warm start**（本地路径：`work/above/tony/x/raptor-knee-maxspan/raptor_ft_coatnet_v5_full_swa.pt`，**292,831,426 B**）。

---

## 2. 要传的东西（都很小，除了检查点）

| 资产 | 本地路径 | 大小 | 必须？ |
|---|---|---|---|
| **训练器** | `kernels/kernel-msft2-long/knee-mri-training-the-twelve-finding-model.py` | ~48 KB | ✅ |
| **v5 warm start** | `work/above/tony/x/raptor-knee-maxspan/raptor_ft_coatnet_v5_full_swa.pt` | **293 MB** | ✅（若 §1.4 不在） |
| **fold 文件** | `ppxl16/rsna-knee-cv-folds` 里的 `fold_file.json` | 小 | ✅（若 §1.3 不在） |
| **标签** | `flight0234/rsna-knee-hybrid-report-labels` 的 `report_labels_v4hybrid.csv` | 小 | ✅（若 §1.3 不在） |
| 语料 | `/kaggle/input/.../all_vols.npy` + `extra_vols.npy` | **22 GB** | ❌ **已在盒子上，不要传** |

---

## 3. 路径对接（训练器认的是 `/kaggle/input/...`）

训练器的 `_find()` / `_open_ids()` 会在 `/kaggle/input` 下递归找 `all_vols.npy` / `extra_ids.npy` / `fold_file.json` / 标签 CSV。
盒子上语料在 `/kaggle/input/datasets/...` ⇒ **大概率直接可用**；若不在，用**符号链接**而不是复制（省一次 22 GB 的搬运）：

```bash
mkdir -p /kaggle/input/datasets
ln -sfn /root/autodl-tmp/kaggle/input/datasets/dreaddevelopment /kaggle/input/datasets/dreaddevelopment
```

⚠️ **`-ext` 必须一起可读** —— **A56 的坑**：只挂 `knee-raptor-corpus` 不挂 `-ext`，held-out 会变成 **642** 而不是 **870/881**，而**日志看起来完全正常**。**跑起来第一件事就是核对 held-out 的 n。**

---

## 4. 训练命令（照抄 `kernel-msft2-long` 的 argv，只改路径）

```bash
cd /root && nohup /root/miniconda3/bin/python /root/knee-mri-training-the-twelve-finding-model.py \
  --labels <labels.parquet> \
  --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k \
  --res 384 --epochs 12 --bs 2 --k 12 --k_eval 24 \
  --grad_ckpt --ckpt <v5.pt> \
  --seed 42 --tag msft0L --folds 5 --fold 0 \
  --fold_file <fold_file.json> --workers 0 \
  > /root/msft2_long.log 2>&1 &
```

⚠️ **`--ckpt <v5.pt>` 是本臂的灵魂**：训练器的 `load_raptor` 会**解包 `model` 容器、0 张量就 `raise`、并把那 8 张 head 一起带上**（A170/C87）。**跑起来必须看到 `head tensors carried: 8`**，否则就是从 timm 从头训。

---

## 5. 跑起来后立刻核对三行（不核对就会白跑）

```bash
sleep 120; tail -40 /root/msft2_long.log
```

| 看什么 | 期望 | 不对就是 |
|---|---|---|
| `loaded ... tensors` | **478**，且 `head tensors carried: 8` | 载入失败（A170 的静默零） |
| held-out 的 **n** | **870 或 881** | **A56**：`-ext` 没读到 |
| 逐 epoch 的 `gold_auc` | 曲线（**本臂就是要看它是否仍在上升**） | — |

---

## 6. 训练的判据（运行前已预登记，见 C104）

1. **逐 epoch 曲线必须仍在上升** —— 若走平/回落，"更长"不是问题所在，**此线停止**
2. **决定性数是成员增量**（**在本机就能算，零 GPU**）：把产出的 `raptor_oof_msft0L_fold0.npz` 拿回本地，
   跑 `python scripts/member_add_test.py --arm msft0L@f0 --strong`，**打 msft2 的 +0.0084**
3. **solo 分不作判据**（C90）
4. ⚠️ **过拟合防线**：gold 先升后回落 ⇒ **取峰值 epoch**，不取最后一个

⚠️ **已有的不利先验**：`msft0L` 在被取消前**第 1 个 epoch 就到 gold-58 = 0.9027**，而 msft2 跑满 4 epoch 的 best 是 0.9048 ⇒ **只差 +0.0021**。⇒ **这一枪有可能是平的**，我先把话说明。

---

## 7. 收尾（省钱的唯一办法）

```bash
# 拿回：raptor_ft_msft0L.json / raptor_gold_msft0L.npz / raptor_oof_msft0L_fold0.npz
scp -i ~/.ssh/autodl_ed25519 -P 34769 root@connect.cqa1.seetacloud.com:/root/raptor_* .
```
⇒ **然后你必须去控制台关机**（没有 API；空闲也计费）。
