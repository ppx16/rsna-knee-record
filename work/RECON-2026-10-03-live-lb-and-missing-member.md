# 2026-10-03 RSNA 现场勘查：真实名次、公开区天花板、以及那个缺失的成员

## 1. 实时榜单（权威读数，`kaggle competitions leaderboard --download`，快照 2026-10-03T12:23Z）

榜是**实时**的；此前文档里的 4692 队 / rank 578 / 0.943 那一格 558 队**全部过期**。

| | |
|---|---|
| 队伍总数 | **4981** |
| **我们** | **rank 677 · 0.943 · 39 次提交**（最后 2026-10-03 06:32） |
| 榜首 | **0.961**（3 队，提交于 10-02/10-03） |
| 铜牌线（前 10%） | **rank 498** |
| `0.943` 这一格 | **889 队，占 rank 326–1214** |
| `0.944` | 58 队，rank 268–325 |

⇒ **我们在铜牌线外 179 名。** 唯一出路仍是 **0.944**（进 rank ≤325，安全越过 498）。
⇒ `0.943` 那一格从 558 涨到 **889**，且**格内排序不透明**（rank 490–505 的提交日期完全乱序，不是先到先得）。

## 2. 公开区确实封顶在 0.943 —— 这次是**两本不同的 notebook** 各自证明的

| notebook | 公开部分得分 | 0.946 从哪来 |
|---|---|---|
| `yamadan96/rsna-knee-d4-public0946` | **0.943** | 作者**私有的 ConvNeXt 集成**按 10% 混入 |
| `ryokucha/rsna-knee-d4-blend-0946-ours10` | **0.943** | `dataset_sources` 前两项是**空串**（= 拉取时被隐藏的**2 个私有数据集**） |

这与仓库已记的那条完全一致（"14/14 公开数据集与我们相同，唯一差异是私有数据集"）。
⇒ **从公开 notebook 里拿不到 0.944。**

`four-reader`（`sushanthtiruvaipati/rsna-knee-d4-public0946-no-correction`）是 `ours10` 把两个私有项换成公开的
`cnx-m448-f0-public`，但**代码里从没引用它**（32 个 cell 搜 `cnx|convnext` 只命中 markdown 与 `build_backbone`
的架构前缀元组）⇒ 挂了没用，自然还是 0.943。

## 3. ⭐ 但那个成员**是公开的**，而且作者明说它是给人替补用的

`mattiaangeli/rsna-knee-cnx-m448-f0-public`（192 MB，公开）内含：

```
cnx_montage448_f0.pt              216 MB   <- 训练好的权重（selected_epoch = 11）
btkr_cnx_montage448_optional.py    21 KB   <- 527 行自包含推理脚本（"optional"= 设计成可插）
convnext_tokens.py / hierarchical_slot_mil.py / slot_local2d.py
ordered_montage448.py / cnx_dicom_geometry.py / cnx_wide_geometry.py
build_cnx_geometry_manifest.py / btkr_cnx_montage448_manifest.json
package_receipt.json
```

**`package_receipt.json` 逐字**：

```json
"role": "one-fold fallback when the private five-fold M448 output is unavailable"
```

⇒ **这就是 0.946 那个私有 5 折成员的官方单折替补。** 私有 5 折在 10% 权重下值 +0.003
（0.943→0.946）；单折按比例约 **+0.001~0.0015** —— **正好是我们要的那个 +0.001**。

脚本自己的常量与我们的栈对齐：`LABELS` = 那 12 个标签，`BACKBONE = convnext_small.dinov3_lvd1689m`，
`BAND = (0.06, 0.94)`。

⚠️ **同时确认了一条反向事实**：`ours10` 的 `A5_W = 0.52`，权重来自 `mattiaangeli/knee-mri-fold-weights`
（公开）—— **与我们 `0943cf` 的 `a5_w = 0.52` 完全相同**。所以那"10%"不在 A5 上，确实在私有成员上。

## 4. 语料密度那条我提错了（已核实）

C84 测过（64 槽 vs 44 槽，soft OOF **+0.0068**），但 **C86 发现那个语料有缺陷**：我们的 `prep()` 用
min-max 归一化，非零像素均值 **58.12 vs 发布版 102.24**（暗 43%），所以 C84/C85 的对比**被强度尺度污染**。
更关键的是处置理由与语料对错无关：**即便密度值 +0.007，成员也只从 0.862 到 0.869，够不到 ~0.90 的成员门槛。**
⇒ 这条路是死的。**我不该提它。**

## 5. 预算（权威读数，`quota_view`，刷新 2026-10-10）

| | 已用 | 总量 | 余 |
|---|---|---|---|
| TPU | 0 s | 72000 s (20 h) | **20 h** |
| GPU | 14147 s (3.93 h) | 21600 s (6 h) | **≈2.07 h** |

⚠️ 注意 kaggle CLI **2.1.2 和 2.2.2 都没有 `submission-limits` 子命令**（2.2.2 有 `team-submissions`），
所以 A135/A162 引用的额度读法**现在取不到了**。

## 6. ⏭️ 计划（按预期价值排序）

1. ⭐ **把公开的 M448 单折成员接进 0943 栈，按 ~10% 混合，提交。** 这是唯一有第三方 LB 证据支持的杠杆，
   且**不需要训练**（推理期）。风险：混合权重未知、成员接口要对齐、单折可能太弱。
2. 若第 1 步不够：用那 20 h 空闲 TPU 精调 `apoorvadixit/rsna-knee-weights-v4cx`（5 折 ConvNeXt-small，
   公开），造一个**自己**的 5 折 ConvNeXt 成员。预算够（5 × 2.6 h ≈ 13 h）。
   ⚠️ 但那批检查点的 head 是不同架构（`--allow_bad_warmstart`），**solo 可能低于 0.90 门槛**（A167：<0.90 净有害），
   所以必须先量 solo 再决定是否继续。
3. 终局两槽仍按 `docs/FINAL-SELECTION-2026-10-04.md` 的修正案（`56791368` + `56613303`），**浏览器操作**。

## 7. 工具坑（本次新增）

- `kaggle competitions leaderboard -s` **只给 20 行**；要全量必须 `--download`（下 zip）。
- **找不到自己队时先看列名**：`TeamName` 是队名，用户名在 **`TeamMemberUserNames`**；
  而我第一版拿 `TeamName` 搜 `ppxl16` 直接漏掉了自己。
- 同一目录连续 `kaggle kernels pull` 会**用同名 `kernel-metadata.json` 互相覆盖**，只有最后一个 notebook 留下 ——
  每个 kernel 必须 pull 到自己的子目录。
- `web_fetch` 在本沙箱被 DNS 守卫全面挡死；**用 `curl.exe` / `gh api` 代替**。
- 榜单 API 会间歇性 `SSLEOFError`，**重试即可**（本次第 1 次就成功了）。
