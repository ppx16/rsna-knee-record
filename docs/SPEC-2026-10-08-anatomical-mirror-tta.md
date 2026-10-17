# SPEC — 2026-10-08 — 「anatomical mirror」TTA：完整机制（从公开实现逐字读出）

> 来源：`nartaa/rsna-knee-0949-anatomical-mirror`（46 票，自称 **0.949 public AUC**）的推理代码。
> **这是本项目近期找到的、唯一一个【推理侧、零重训、且公开面已证有效】的 ≥0.005 杠杆。**
> ⚠️ 本文只记录**从代码读出的机制**；**尚未在我们的实现里落地，也尚未在本地仪器上量过**。

---

## 1. 机制：按解剖轴选择的几何镜像（**不是**标签交换）

公开实现逐字：

```python
def _mirror_tri(tri, cs, D):
    """final-g1-bundle MIRROR_TTA: true L/R mirror of (k,3,H,W) uint8 eval triplets
    (research/rnd4/mirror/mirror_f0.py). The slot of each window's CENTRE slice decides
    the axis: sagittal (slots 0,1) -> reverse the 3 slice-channels; coronal/axial (2,3,4)
    -> flip the width.  Done on uint8 BEFORE the per-channel LUT, i.e. 'mirror the knee,
    then normalise'."""
    s_, _ = _slot_index(D)
    ...
    return torch.where(sag, tri.flip(1), tri.flip(-1))
```

**四条要点：**

| # | 要点 |
|---|---|
| 1 | **镜像做在【输入】上**，不是交换输出列 |
| 2 | **镜像轴由【窗口中心切片所在的槽】决定**：<br>· **矢状位（槽 0,1）** → L/R 就是那 **3 个切片通道** ⇒ **`flip(1)`（反转通道）**<br>· **冠状/轴位（槽 2,3,4）** → L/R 就是**图像宽度** ⇒ **`flip(-1)`（翻转宽度）** |
| 3 | **在 uint8 上、per-channel LUT 【之前】做** —— 原话 *"mirror the knee, then normalise"* |
| 4 | **两个视图等权平均**：`infer_pair_probs(model, plain, mirrored)`，两者形状都是 `(94, 3, 320, 320)` |

⇒ **为什么叫 "anatomical" 而不是 flip**：因为**镜像轴是按解剖平面选的**，所以镜像之后**膝盖的左右仍然是它自己的左右** —— 普通的 `flip` 会把 medial/lateral 的语义打乱（这也是我们训练器里 `NO flips (laterality is signal)` 的由来）。**§2 的收益表证明这个区分是真的。**

## 2. ⭐ 为什么这条对准我们（公开实现自报的收益分布）

代码里逐字：

```
# Biggest gains land on the findings that were capping us: Lateral Meniscus +0.071,
# Fracture +0.057, Lateral OA +0.048, Medial Meniscus +0.035, ACL +0.028.
```

对照我们自己的弱项（FACTS **C97** 的逐 finding 矩阵，`msft0` 的 gold-58 分项）：
**Lateral OA、Lateral Meniscus、Contusion、PF OA、Synovitis 是我们最低的几项** ——
而**榜单收益最大的三项（Lateral Meniscus / Lateral OA / Medial Meniscus）恰好落在我们最弱的那一片**。

⇒ **不是巧合**：镜像 TTA 补的正是"单侧视角看不到对侧同名结构"的那部分信息。

## 3. 同本 notebook 的另外两个设置（对照用）

| 项 | 它的值 | 我们 |
|---|---|---|
| 训练分辨率 | **384 full-field** | msft2/rdr 用 384 |
| **推理分辨率** | **native 320px 中心裁剪** | 我们读预计算语料（336/384） |
| **窗口数 K** | **94** | 我们 `k_eval` 24（或 62/78） |
| 归一化 | per-channel LUT（放在镜像**之后**） | 见 C98/C99 |

⚠️ **不要把"320 + K94"与"镜像"当成一个包一起搬** —— 那是三个独立变量。**要一次只动一个**（本仓库反复的教训：混在一起就分不清是谁的功劳，见 C86 ⑤ 的密度/归一化混淆）。

## 4. 落地步骤（尚未执行）

> ⭐ **可复用核心已实现并自测通过**：`work/mirror_tta.py`
> · `anatomical_mirror(tri, centre_slot)` —— 单个 `(3,H,W)` uint8 窗口
> · `mirror_batch(tri, centre_slots)` —— **逐窗口**选轴（这才是关键，批量统一翻转是错的）
> · `test_axis_choice()` —— 断言**每槽的轴**、**镜像两次 = 恒等**、以及**"批量宽度翻转"那个陷阱**
>
> **44 槽表（我们训练器读的那份）已从 `kernels/kernel-diag-slots/diag_slots.py` 逐字抄入**：
> `[("Sagittal",1,12),("Sagittal",0,10),("Coronal",1,8),("Coronal",0,6),("Axial",-1,8)]`
> ⇒ **sagittal 0..21 → 反转通道 · coronal 22..35 → 翻转宽度 · axial 36..43 → 翻转宽度**
>
> ⚠️ **自测当场抓到了我自己的一个错**：我按 18/32/64 写期望边界，而 **sagittal 有两组**（12+10=22）——
> 边界是**断言出来的**，不是假设的。**这正是自测的价值。**

1. **先只加镜像**，其余一律不动（推理分辨率、K、归一化都保持现状）—— 这样**镜像的效应是孤立的**。

### 4.1 集成点（已定位，来自 `kernels/kernel-raptor-train` 的实际源码）

```python
def __getitem__(self, i):
    ...
    m = self.masks[row]
    valid = np.where(m > 0)[0]              # 语料 D 轴上的有效切片索引
    ...
    cs = self._centers(valid, n_centers)    # 中心切片索引
    vol = self.vols[row]                    # (D,H,W) u8
    for c in cs:                            # 用 c-1, c, c+1 组成 (3,res,res)
```

⭐ **关键（已核实）**：`_centers` 返回的是 **`range(lo+1, hi)` 里的【原始索引】** ⇒ **`c` 就是语料的原始槽索引**
⇒ **平面可直接查 `slot_planes(SLOT_GROUPS_44)[c]`**，不需要任何间接映射。这是最容易出错的一步，而它恰好是干净的。

**接法（最小改动、且隔离）**：
1. 加一个 `--mirror` 开关（默认关）→ `self.mirror`
2. 在 `__getitem__` 里、**窗口三元组刚拼好、`_resize` 之前**应用镜像
   —— 因为公开实现的原话是 *"mirror the knee, then normalise"*，即**在 uint8 上、LUT 之前**
3. **跑两次评测**（无 `--mirror` / 有 `--mirror`）→ 得两份 `raptor_gold_*.npz` → **在本地平均**
   ⇒ **内核里不需要任何平均逻辑，镜像的效应被完全隔离**

⚠️ **唯一未核实的细节**：`__getitem__` 里拼好的三元组**变量名**（源码在 line 280 之后）。**这是唯一还需要读的一处**，不需要任何新概念。

### 4.2 ✅ 已接线（同日晚些，`work/patch_raptor_mirror.py`）

核实源码后确认：三元组变量名是 **`tri`**，且它**在同一行**就做了 `/255.0` 归一化：

```python
tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0  # (3,H,W)
```

⭐ **公开实现要求 "mirror the knee, then normalise"，而这里可以放在 `/255` 之后** ——
因为**反转通道**与**翻转宽度**都是**线性索引操作，与标量除法可交换** ⇒ **算术上完全等价**。
**这是性质，不是偏好**（否则就要重构数据路径）。

**补丁（两处锚点，都从源码核实过，不猜）**：
1. `class StudyWindows(Dataset):` 之前 —— 插入 `_SLOT_GROUPS_44` / `_SLOT_PLANES`（**含 `assert len == 44`**）/ `MIRROR_ENABLED`
2. `tri = np.stack(...)` 之后 —— 插入按 `_SLOT_PLANES[c]` 选轴的镜像

**开关用环境变量 `RAPTOR_MIRROR`**，**不需要改构造函数或 argparse**（第一次尝试猜了构造函数锚点，构建器**在写盘前正确 FATAL**，没有损坏文件）。
两份训练器（`-train` / `-train-rule`）都已打好并 `compile()` 通过。

⚠️ **一条必须写明的限制**：`_SLOT_PLANES` 是 **44 槽表**。**只有跑 44 槽语料时它才正确。**
若哪天用 **64 槽**语料跑，**必须换成 `[("Sagittal",1,18),("Sagittal",0,14),("Coronal",1,12),("Coronal",0,8),("Axial",-1,12)]`** —— 否则镜像轴会用错平面，**那正是会毁掉 laterality 的错误**。`work/mirror_tta.py` 里两张表都有，且自测覆盖两者。

⚠️ **尚未实跑验证**：接线完成、编译通过，但**没在有语料的环境里跑过**。下一步是跑两次评测（无/有 `RAPTOR_MIRROR`）并在本地平均。

---

## 6. 实测结果（`kernel-eval-mirror` v1，COMPLETE）与一个**必须先排除**的疑点

```
ev_v5_plain   0.8965243697042471     <- 与 kernel-eval-v5v9 那次的 v5 值【逐位相同】
ev_v5_mirror  0.8836737306936034
Δ             -0.0129                <- 按预登记（>= +0.005 才提交）=> 关闭
```

**① 对照精确复现到小数点后 10 位** ⇒ **镜像补丁没有引入任何回归**，这个负数是**可信的读数**，不是管线故障的假象。

### ⚠️ ② 但有一个疑点必须先排除，否则这个负数可能【读错】

`_SLOT_PLANES` 抄自 `kernels/kernel-diag-slots/diag_slots.py` 的 `_SLOTS44`
—— 那是 **community-stack notebook 对槽的【词汇表】**，**不一定等于我们预计算语料 `all_vols.npy` 的 D 轴实际排列**。

⇒ 若语料的 44 个槽**不是按平面分组排列**（例如按序列分组），那么 `_SLOT_PLANES[c]` 会**取到错的镜像轴**，
于是这次的 **−0.0129 就变成"错的镜像"的读数，而不是"镜像没用"的读数**。

**⇒ 在下结论"anatomical mirror 对本管线无效"之前，应当先验一件事**：
**语料的 D 轴第 c 个槽，其平面是否与 `_SLOT_PLANES[c]` 一致。**
（廉价的验法：取一个已知平面的研究，看它在各槽上的非零 mask 分布是否与表的分组吻合。）

⚠️ **这条疑点不改变"不提交"的处置**（−0.0129 无论哪种解释都不该提交），
**但它改变【结论的措辞】**：现在只能说"**按我们的实现，它没帮上忙**"，**不能说"这个机制无效"**。
2. **在我们的 870 仪器上量**（±0.004，能判 ≥0.005），**不占公榜名额**。
3. ⚠️ **实现要点**：镜像轴必须**按每个窗口中心切片的平面**决定，**不能**统一对图像宽度 `flip` —— 后者就是那个会毁掉 laterality 的朴素翻转。
4. **判据**：本地 ≥+0.005 才提交；否则记录并关闭。

## 5. 诚实边界

- **我们没有这个实现的许可证说明**；机制已从公开代码读出，**但"照做"仍需自己复现并验证**。
- **它的 0.949 是它自己的完整管线**（含 320px / K94 / 它自己的权重），**不等于"我们加镜像就 +0.005"**。
- 我们的弱项与它的收益表**重合**是**支持性证据，不是因果证明**。
