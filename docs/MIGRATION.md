# MIGRATION —— 把训练侧搬到租来的算力上（2026-09-24 建）

> **一句话**：训练路径只吃 **≈22 GB、CC0 许可、已经解码好的语料**，不吃那 570 GB 比赛 DICOM。
> 所以"租算力"这个动作 = **搬 22 GB**，不是搬 570 GB。这份文件就是那份搬迁清单 + 校验门槛。

背景（2026-09-24 用户定的规矩）：
**以后 GPU 只用来提交算成绩，实验用 TPU；算力不够就买机器并迁移数据集。**
于是本仓现在有两种资源路线，搬家准备工作全部围绕它们：

| 路线 | 在哪 | 能干什么 | 不能干什么 |
|---|---|---|---|
| Kaggle（现用） | TPU v3-8 跑折（77 min/折，队免费）+ GPU 只提交 | 全部 | — |
| 租来的机器（备用） | 自己的 GPU | **训练折 / 跑 CV / 造新语料** | **不能提交**（提交必须走 Kaggle，见 §7） |

---

## 1. 要搬的（清单，截至 2026-09-24）

训练侧的全部输入 = 4 个数据集。前两个是大头，后两个是几 MB。

| 数据集 ref | 关键文件 | 大小（实测） | 许可 | 为什么必须 |
|---|---|---|---|---|
| `dreaddevelopment/knee-raptor-corpus` | `all_vols.npy` (3200,44,336,336) uint8 + masks/ids/window_index | **15,895,756,928 B = 14.80 GiB**（zip **11.10 GiB**） | **CC0** | 主语料 |
| `dreaddevelopment/knee-raptor-corpus-ext` | `extra_vols.npy` (1207,44,336,336) uint8 + … | **5,995,680,896 B = 5.58 GiB**（zip **4.21 GiB**） | **CC0** | 另外 1207 例（**按 4407 索引 `all_vols` 会越界**，A2） |
| `flight0234/rsna-knee-hybrid-report-labels` | `report_labels_v4hybrid.csv` | 978,591 B | 公开 | 训练标签（gold-58 已在训练时整列排除，A4） |
| `ppxl16/rsna-knee-cv-folds` | `fold_file.json` | 312,908 B | 我们自己 | **没有它，重跑出来的折和我们的 OOF 不可比** |
| ⚠️ 比赛 `train.csv` | `rsna-knee-abnormality-detection` 的 `train.csv` | 5,690,007 B | **比赛数据** | 训练脚本用它取"合法 study 列表 + gold-58 真值"。**只取这一个 CSV，570 GB 的 DICOM 一个字节都不搬**；它也不会被发布，只放在我们自己的机器上 |

外加**代码包**（`scripts/` + `state/` + `kernel-*/` + 根 md + 判分证据）：**15.9 MB 压缩后**
（507 个文件），见 `scripts/make_pack.py`。

**总计：zips 15.4 GiB（要传输的）+ 解包后 20.4 GiB；账本 15 个文件 / 38.34 GB（zip 与解包都登记）。**
（这就是"迁移数据集"的真实体量 —— **不是 570 GB**。）

## 2. 不要搬的（以及为什么）

| 东西 | 大小 | 为什么不搬 |
|---|---|---|
| 比赛 DICOM 原始数据 | **≈570 GB** | **比赛规则不允许再分发**；而且**训练路径根本不读它** —— 语料已经是"每例 44×336×336 uint8 的定长堆叠"，DICOM 解码在作者侧就做完了（A1/C38） |
| `outputs/` | 12.7 GB | 内核跑出来的副产品，重跑可再生成 |
| `downloads/` | 9.4 GB | 本地缓存。**例外**：里面 `raptor_oof_*.npz` / `raptor_gold_*.npz` / `raptor_ft_*.json` 是**判分证据**，代码包的 `--evidence`（默认开）会把它们带上 |
| 权重 `.pt` | 每人 0.3 GB | 按需单独搬，见 §8 |

⚠️ **推理（不是训练）确实是 DICOM 绑定的**：集成内核要读 1322 个 study 的原始切面。所以
**租来的机器只能做训练/CV，不能承担出分那一步**。这一点决定了租算力的用途边界。

### 2.1 ⚠️ 语料里有一个**主成分级的洞**，搬过去的人必须知道（2026-09-24 全量实测）

`all_masks.npy` / `extra_masks.npy` 是"这个槽是不是真数据"的标记。全量统计（3200 study × 5 组）：

| 组（在 44 槽里的位置） | 整组 mask=0 的 study 数 | 占比 |
|---|---|---|
| **Coronal fluid=0（行 30–35）** | **824** | **25.8%** |
| Sagittal fluid=0（行 12–21） | 86 | 2.7% |
| Sagittal fluid=1 / Coronal fluid=1 / Axial | 0 | 0% |
| 合计 | 910 / 16000 组 | 5.7% |

整体 mask 密度 **0.959**。**mask=0 的槽在 `all_vols.npy` 里就是纯 0 像素**（随机抽 300 个 study、
626 个无效槽，非零像素 = 0）。

⇒ **三条必须记住的**：
1. **44 个槽里有 6 个，对四分之一的 study 是黑的** —— 这不是边角，是输入里的一个主成分级缺口；
2. 训练脚本靠 **mask** 把这些槽排除（`StudyWindows._centers` 只用 valid 槽）⇒ **迁移/重写 loader 必须保留 mask 这条通路**。
   忽略 mask 不会读到垃圾（读到黑图），但**会把四分之一的 study 的 6 张黑图当数据用**；
3. 未来重建更密语料时，**必须原样复现"取不到该 (plane,fluid) 就整段留空 + mask=0"这个行为**，否则训练分布不可比
   （细节与证据见 HANDOFF §143.11 / FACTS C73–C76）。

## 3. 三条搬迁通道

| 通道 | 命令 | 适用 | 代价 |
|---|---|---|---|
| **A. Kaggle 直连**（推荐先试） | 在租机上 `python3 scripts/corpus_fetch.py --dest /data/corpus` | 租机有公网 + 能放 `~/.kaggle/kaggle.json` | 0，且**服务端有对照表**（见 §5） |
| **B. 本机中转** | 本机已下好（`D:\kaggle\corpus\`），用 `scp`/`rsync` 推上去 | 租机在国内、Kaggle 直连慢 | 吃你的上行带宽 |
| **C. 网盘** | 本机 → 夸克/百度网盘 → 租机 | 通道 B 太慢时的兜底 | 大文件要会员；**多一次校验** |

**建议顺序：A 先试（10 分钟见分晓），不行再 B，最后 C。** 通道 A 唯一的前置是租机能访问 kaggle.com；
如果租机是国内云，通常 A 也能跑，只是慢。

**时间量级**（22 GB）：12 MB/s ≈ 30 min；50 MB/s ≈ 7 min。本机实测下载 10–30 MB/s。

### 3.1 ⚠️ 别用 `kaggle datasets download` 搬大文件 —— 用 `scripts/kaggle_resume.py`

实测（2026-09-24）：`kaggle datasets download` 把已经下到 **7.30 GB** 的语料**截断回 6.80 GB 重新开始**
（服务器对 Range 请求回 200 时，CLI 会以 `"wb"` 重开文件），一次丢掉约 40 分钟。
此外**跳转 `storage.googleapis.com` 的 SSL 握手会间歇性 `SSLEOFError`**（连 60 KB 的 `train.csv` 都会中招）。

⇒ 本仓改用 **`scripts/kaggle_resume.py`**：**只追加、永不截断**，Range 续传，失败指数退避无限重试，
最后**用 Kaggle API 报的文件大小核对**。两边都用它：

```powershell
# 本机（Windows）—— 一条命令把 4 个数据集 + train.csv 全部就位，可反复重跑
powershell -File scripts\fetch_corpus_local.ps1
```
```bash
# 租机（Linux）—— corpus_fetch.py 会自动优先调用同目录下的 kaggle_resume.py
python3 scripts/corpus_fetch.py --dest /data/corpus --layout /kaggle
```

（`kaggle_resume.py` 也支持裸 URL：`--url '<signed>' --out f.csv --expected 12345`，
比赛文件的签名 URL 就是那个会抖的跳转。）

## 4. 本机侧已经做好的（2026-09-24，**全部实测通过**）

```
D:\kaggle\corpus\
  zips\        knee-raptor-corpus.zip            11.10 GiB
               knee-raptor-corpus-ext.zip         4.21 GiB
               report_labels_v4hybrid.csv / fold_file.json / train.csv
  data\        knee-raptor-corpus\      all_vols.npy 14.80 GiB (+ masks/ids/window_index)
               knee-raptor-corpus-ext\  extra_vols.npy 5.58 GiB (+ …)
               rsna-knee-hybrid-report-labels\ report_labels_v4hybrid.csv
               rsna-knee-cv-folds\       fold_file.json
  packs\       rsna-knee-code-<date>.zip  （507 文件 / 15.9 MB，含账本 + 判分证据）
```
- `state/CORPUS_MANIFEST.json` —— **校验账本**：**15 个文件 / 38.34 GB**，逐文件 sha256 + 字节数 +
  `.npy` 头（shape/dtype：已核对 `3200×44×336×336` 与 `1207×44×336×336`），
  外加 Kaggle 服务端文件清单（`kaggle datasets files`）作**独立锚点**（4 个 ref 全部取到）。
- 生成/校验：`python scripts/corpus_manifest.py build|verify|summary`。
  ✅ **本机已跑出 `VERIFY OK`**（2026-09-24 06:57Z）。
- ✅ **`--layout` 目录镜像也在本机实测过**：10 个硬链接摆出 `/kaggle/input/datasets/...` 的形状。

⚠️ **账本必须跟着代码包一起走**（它在 `state/` 里，`make_pack.py` 默认带上）。
没有账本，租机上就没有"这份拷贝 == 我们训练用的那份"的证据，跑出来的折就不能和 OOF 配对。

⚠️ 账本**故意不登记** `packs/`（每次改代码都会变，且它正是账本的载体）与 `layout/`（只是 `data/` 的硬链接镜像，
在租机上它位于 `--root` 之外）—— 登记它们会让 `verify` 因为与语料无关的原因报错。

## 5. 租机侧落地步骤（照抄）

```bash
# 0) 环境自检（python / torch+cuda / timm / 磁盘 / kaggle CLI）
python3 scripts/corpus_fetch.py --dest /data/corpus --env

# 1) 下载 + 解包 + 把目录摆成 Kaggle 的样子（这一步是关键，见 §5.1）
python3 scripts/corpus_fetch.py --dest /data/corpus --layout /kaggle

# 2) 硬门槛：校验（必须打印 VERIFY OK）
python3 scripts/corpus_manifest.py verify --root /data/corpus

# 3) 看一眼形状与前缀（可选）
python3 scripts/corpus_manifest.py summary
```

**磁盘**：zips + 解包 ≈ 44 GB，留 60 GB 舒服；`--layout` 默认用**硬链接**（同一份字节两个路径），
**不额外占空间**。校验通过后可以删 `zips/` 省一半（但那样只剩一层校验，建议**先校验再删**）。

`kaggle` CLI 的凭据放在 `~/.kaggle/kaggle.json`（本机那份 `{"username":"ppxl16","key":...}` 即可；
**不要**把凭据塞进代码包）。若租机装不了 CLI，用 Kaggle 网页手动下载这 4 个 dataset zip + `train.csv`
放进 `zips/`，然后 `python3 scripts/corpus_fetch.py --dest /data/corpus --unpack-only --layout /kaggle`
—— **通道 A 不是必须的**。

### 5.1 ⭐ 为什么要把目录摆成 `/kaggle/input/...`（这是"少写代码"的那一招）

`kernel-raptor-train` 是这样找输入的：`_find()` **递归遍历 `/kaggle/input`** 找文件名，
输出写到 `HERE = /kaggle/working`。所以只要租机上的目录长这样：

```
/kaggle/input/datasets/dreaddevelopment/knee-raptor-corpus/      all_vols.npy  all_masks.npy  all_ids.npy ...
/kaggle/input/datasets/dreaddevelopment/knee-raptor-corpus-ext/  extra_vols.npy ...
/kaggle/input/datasets/flight0234/rsna-knee-hybrid-report-labels/report_labels_v4hybrid.csv
/kaggle/input/datasets/ppxl16/rsna-knee-cv-folds/                fold_file.json
/kaggle/input/competitions/rsna-knee-abnormality-detection/      train.csv
/kaggle/working/                                                 （输出：权重 / OOF npz）
```

**同一个 `.py` 文件就能不加改动地跑**。⚠️ 这一点比"方便"重要：我们的 CV 是**折内配对**的仪器
（OOF 配对 CI 只有 ±0.004），一旦为了本地跑分叉出一份"租机版脚本"，两边就不再是同一个仪器。
**宁可让目录去迁就代码，也不要让代码去迁就目录。**

落地后可以直接这样跑一折（示例：`convnext_base` / 336 / fold 0）：

```bash
python3 kernel-raptor-train/knee-mri-training-the-twelve-finding-model.py \
  --labels /kaggle/input/datasets/flight0234/rsna-knee-hybrid-report-labels/report_labels_v4hybrid.csv \
  --folds 5 --fold 0 --fold_file /kaggle/input/datasets/ppxl16/rsna-knee-cv-folds/fold_file.json \
  --arch convnext_base --res 336 --epochs 12 --tag rent0
```
（⚠️ `--grad_ckpt` **不要**在 TPU/XLA 上加（X5），CUDA 上可以用它省显存。）

## 6. 为什么校验是硬门槛（不是仪式）

- OOF（870 study）配对 CI ≈ **±0.004**；gold-58 配对 CI ≈ **±0.021**。
  也就是说**我们整条判决链条的分辨率就是千分之几**。
- 如果租机上的语料**少了/多了/顺序不同**，那么"租机上折 0 比 Kaggle 上折 0 好 0.003"
  这种结论**根本读不出来**，而且不会报错 —— 只会静默给出一个错的答案。
- `verify` 比对的是 **sha256**，不是大小；`summary`/`build` 还会比对 **shape 与 dtype**
  （语料换一版就是另一份语料，见 §7）。

⇒ **没通过 `VERIFY OK` 就不要在该机器上启动任何折。**

## 7. 迁移之后能做什么 / 不能做什么

| 能做 | 怎么做 |
|---|---|
| 跑折（GPU 上比 TPU 快得多，且**没有 9 h 会话上限**） | 复用 `kernel-raptor-train` 的训练脚本本体；把数据根指向 `/data/corpus/data` |
| 跑本地 CV 判分 | 把产出的 `raptor_oof_*.npz` / `raptor_gold_*.npz` 带回本机，用 `scripts/cv_score.py` 判分（**判分器必须在同一个口径下比较**） |
| **训练 CoAtNet（GPU 专属）** | Kaggle GPU 上 12 epoch 的 coatnet 折 ≈ 10–18 h > 9 h 会话上限（C62）⇒ **这是租算力最正当的理由** |
| 批量重训 | 语料换成更密的版本之后，一次性重训多折 |

| 不能做 | 原因 |
|---|---|
| **提交** | code competition 的提交只能在 Kaggle 上完成；租机最多产出权重，权重还得回传 |
| 端到端出分 | 推理要读 **570 GB 比赛 DICOM**，搬不动也不许搬（§2） |
| 用 TPU 的 XLA 补丁 | 那些是 torch_xla 专用（X1–X7），GPU 上用 CUDA 原路径 |

## 8. 权重与证据怎么来回搬

**去的时候**（可选，按实验需要）：
- 上游 CC0 成员权重（`dreaddevelopment/raptor-knee-*`、`mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2`）
  —— 都能用 `kaggle datasets download` 直接拉，**不必从家里传**。
- 我们自己的权重：`downloads/raptor_v9/raptor_ft_coatnet_v9_full.pt` (0.28 GB)、
  `downloads/nx224/`、`downloads/nx336/`（各 0.33 GB + OOF/gold npz，几 MB）。

**回来的时候**（必须带）：
1. 训练权重 `raptor_ft_<tag>.pt` / `_swa.pt`；
2. `raptor_oof_<tag>_fold<k>.npz`（**判分唯一入口**）；
3. `raptor_gold_<tag>.npz`（gold-58 口径，**唯一能与前沿公开数字绝对值对比的量**）；
4. 训练日志（逐 epoch 曲线，`cv_score.py` 要用）。

然后在本机：`python scripts/cv_score.py ...` / `python scripts/experiments_ledger.py` 上账
（新折必须进 `state/EXPERIMENTS.md`，否则下次会重复造轮子）。

## 9. 以后换更密的语料，怎么接进来（这条是真正的上限所在）

已确证的杠杆排序：**容量已封顶（D15，89M→198M 是 −0.0154）**，
**分辨率是真的（OOF +0.0078，D16）**，**数据/更密语料是唯一还没试的大杠杆**
（作者自报 +0.006–0.011，且作者原话是"数据的杠杆大约是模型的三倍"）。

所以将来很可能要**重造语料**（`--k` 窗口数、span、取层规则；配方反推见 C58 与
`kernel-recipe-probe`）。这类语料的正确造法是：

1. **在 Kaggle 上造**（CPU/TPU 内核不占 GPU 额度，且 DICOM 就在那儿）；
2. 产出的 `*_vols.npy` 发布成**私有数据集**（注意 A16：喂给 code competition 的数据集必须是**已版本化**的）；
3. 回到本机 `python scripts/corpus_manifest.py build --root <新语料目录>` —— **重新生成账本**，
   老的账本**不要覆盖**（改 `--out`），因为两版语料的 OOF **不可比**；
4. 之后它就是 §1 表里的一行，迁移照走。

## 10. 命令速查

```powershell
# 本机（Windows）
python scripts\corpus_manifest.py build                 # 建账本（会跑 sha256，22 GB 约几分钟）
python scripts\corpus_manifest.py verify                # 复核本机拷贝
python scripts\corpus_manifest.py summary               # 看形状/大小表
python scripts\make_pack.py                             # 出代码包（含判分证据）
```
```bash
# 租机（Linux）
python3 scripts/corpus_fetch.py --dest /data/corpus --layout /kaggle   # 拉 + 解包 + 摆成 Kaggle 目录
python3 scripts/corpus_manifest.py verify --root /data/corpus
```

⚠️ 本机路径写成 `D:\kaggle\corpus` 是默认值；换机器用 `--root` / `--dest` 覆盖，**不要改脚本默认值**
（改了就等于本机和租机不再是同一套账）。
