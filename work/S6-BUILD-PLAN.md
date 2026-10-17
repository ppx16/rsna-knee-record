# S6 全量腿 —— 接入计划（2026-10-16 侦察完毕，可直接动手）

> 目标：把 **nartaa 的 S6 ConvNeXt-small 全量权重（单独 0.950）** 作为**第 5 个 stage** 接进我们的
> 0.958 基座，并按公开配方把它并进 **compact half**：`compact = 0.60*rank(R6/R7) + 0.40*rank(S6)`。
> **两个理由**：**(a)** 它是公开面 0.957 配方里我们唯一缺的一条腿（C184/C185）；
> **(b)** README 逐字 **"No OAI data was used for this model."** ⇒ **我们全 OAI 血统的对冲**（C165 记过主办方裁定未决）。

---

## 1. 资产（全部公开）

| 项 | 值 |
|---|---|
| 权重数据集 | `nartaa/rsna-knee-s6fall-weights`（186 MB，2026-10-14 19:08） |
| 权重文件 | `E_prior_s42_fall.pt`（200,185,003 B，sha256 `0c7297d58f061ab4e5d7c55046f51fa3e8934928046f0647a0c7854e98371c12`） |
| 随附 | `E_prior_s42_fall.run_config.json` · `.DONE.json` · `checkpoint-manifest.json` · `code_manifest.json`（钉住每个代码文件 sha256） |
| **代码补丁** | 同数据集里的 **`s6_extras.bin`**（25,601 B）—— 补 heliosli 数据集缺的 `compact_*` 脚本 + 16 个 MONAI 0.6.0 文件 |
| 代码主体 | `heliosli/rsna-orthofoundation-fold0-epoch7-20261004`（**我们的基座已经挂了它**） |
| 参考实现 | `nartaa/rsna-knee-s6fall-solo`（**script** 内核，`rsna-knee-s6fall-solo.py`，194,371 B / 1212 行） |
| 单独分 | **0.950 public**（README 逐字） |

---

## 2. 逐字接口（从 solo 脚本里抠出来的，行号是原文件行号）

**几何与预算常量**（脚本 83–105 行）：
```python
B_GEOMETRY = {"backbone": "convnext_small", "size": 320, "crop": 288, "views": "v4", "slices": 3,
              "phys_mm": 3.5, "eval_stride": ..., "chunk": ..., "item_timeout": ...}
B_AMP = "fp16";  B_WORKERS = 2
B_MODEL = {"file": "E_prior_s42_fall.pt", "sha256": "0c7297d5...", "arm": ..., ...}
# 预算（脚本自述：隐藏测试集 = 1,284 个研究）: B_TIMEOUT_S = 14400, B_DEADLINE_MARGIN_S = 20*60
```

**worker 命令行**（`b_cmd`，802–810 行，逐字）：
```
python -u <launcher> --data <comp> --split test --code <code_dir> --models <s6_weights_dir>
        --arm <B_MODEL["arm"]> --backbone convnext_small --size 320 --crop 288 --views v4
        --slices 3 --phys-mm 3.5 --eval-stride <g> --anchor-weights <anatomy.pt> --amp fp16
        --chunk <g> --workers 2 --item-timeout <g> --assignment <ids.json>
        --output <out_dir> --tmp <tmp_dir> --dump-logits <out_dir>/logits.json --deadline <epoch>
```
**环境**（`b_env`，819–825 行）：
```python
dict(os.environ, CUDA_VISIBLE_DEVICES=str(gi), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
     OPENBLAS_NUM_THREADS="1", MALLOC_ARENA_MAX="2", PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1")
```
**完成标志**（`b_done`）：`<out_dir>/receipt.json` 存在 **且** `logits.json` 存在 **且** `log.txt` 里有 `WORKER_DONE`。

**launcher 已抠出并落盘**：`work/s6_launcher_reference.py`（4,807 字符 / 112 行，含三处守卫：
distutils shim、`_setup` 失败时重置 `self.native` 并重抛、无 anchor 的已评分项转成失败项）。
它把 `sys.path` 设成 `[SRC, CODE]`，并**原样调用 Heliosli 的 `compact_infer.main()`**。

---

## 3. ⭐ 接入方式已查清：**manifest 驱动，不需要改推理代码**

读基座 cell 6 得到的关键事实：**compact worker 的全部参数都来自 bundle 的 `manifest.json`**
```python
WORKER = 'compact_infer.py'
WORKER_ARGS = ['--backbone', manifest['backbone'], '--size', str(manifest['size']),
               '--crop', str(manifest['crop']), '--eval-stride', str(manifest.get('eval_stride', 3)),
               '--views', manifest.get('views', 'v1')]
if manifest.get('state_spec'):  WORKER_ARGS += ['--state-spec', str(BUNDLE / manifest['state_spec'])]
if manifest.get('side_model'):  WORKER_ARGS += ['--side-model', ...]
WORKER_ARGS += ['--slices', str(manifest.get('slices', 3)), '--phys-mm', str(manifest.get('phys_mm', 0.0))]
if manifest.get('models2'):     WORKER_ARGS += ['--models2', str(BUNDLE), '--arm2', ...]
```
⇒ **加 S6 臂只要三步**：
1. **给 S6 写一份自己的 `manifest.json`**（基座的 stage 2 会 `assert manifest['bundle_id'] == BUNDLE_ID`
   并逐文件核 sha256，所以它必须自洽）：
   ```json
   {"bundle_id": "<我们自己的字符串>", "arm": "E_prior", "backbone": "convnext_small",
    "size": 320, "crop": 288, "eval_stride": 1, "views": "v4", "slices": 3, "phys_mm": 3.5,
    "models": ["E_prior_s42_fall.pt"],
    "sha256": {"E_prior_s42_fall.pt": "0c7297d58f061ab4e5d7c55046f51fa3e8934928046f0647a0c7854e98371c12"}}
   ```
2. **用同一套 stage-2 runner 再跑一次**（`--bundle <s6 目录>`）⇒ `pred_s6/` ⇒ `s6_sub.csv`；
3. stage 3 里 compact half 改成两支：`compact = 0.60*rank(现役) + 0.40*rank(S6)`，
   再做 `final = 0.5*Raptor_half + 0.5*compact_half`。

**`s6_extras.bin` 的解包**（magic `b"K1PAYLOAD1\n"` = 11 字节 + 未压缩 tar）**在 Kaggle 内核里做**
（数据集已挂载；2026-10-17 本地连续 5 次 `SSL_EOF` 下不下来，所以不要依赖本地下载）：
```python
b = open(payload, 'rb').read()
assert b[:11] == b'K1PAYLOAD1\n'
tarfile.open(fileobj=io.BytesIO(b[11:])).extractall(code_dir)
# 再逐个成员对 code_manifest.json 的 sha256（脚本自己也会核；不一致就是硬错误）
```

**必须的失败安全**（与 band32/r9all 两条腿同样的纪律）：
S6 缺失 / 非零退出 / 列漂移 / 覆盖率 < 50 % ⇒ **原样保留现状**（即手上的 0.958 那一枪），并在
`_s6_report.json` 里写清原因。**任何一条腿都不许把已经握在手里的东西弄丢。**

---

## 4. ✅ 全部常量已确认（侦察完成，无未知项）

```python
B_MODEL = {"file": "E_prior_s42_fall.pt",
           "sha256": "0c7297d58f061ab4e5d7c55046f51fa3e8934928046f0647a0c7854e98371c12",
           "run_config": "E_prior_s42_fall.run_config.json", "done": "E_prior_s42_fall.DONE.json",
           "arm": "E_prior", "expect_epochs": 16, "expect_train": 4263}
B_GEOMETRY = {"backbone": "convnext_small", "size": 320, "crop": 288, "views": "v4",
              "slices": 3, "phys_mm": 3.5, "eval_stride": 1, "chunk": 192, "item_timeout": 900}
ANATOMY_FILE, ANATOMY_SHA256 = "anatomy.pt", "5904dd16eb9a4fd7a1caf90ac977f8fe2ef364908598c23c209db3fe7fd1563e"
B_AMP, B_WORKERS = "fp16", 2
```

**`s6_extras.bin` 的格式（已查明）**：**magic `b"K1PAYLOAD1\\n"` 后面直接跟一个【未压缩的 tar】**
（注释原话：*"PAYLOAD_MAGIC + an uncompressed tar (opaque to Kaggle's archive handling)"*）。
⇒ **解包 = 跳过开头 11 字节再 `tarfile` 解**；脚本自己会**逐个成员核对 sha256/字节数**，
所以解出来必须与 `code_manifest.json` 一致（不一致就是硬错误）。

**重试/容错约定（照抄即可）**：
- `B_TIMEOUT_S = 14400`、`B_DEADLINE_MARGIN_S = 20*60`；隐藏测试集 **1,284** 个研究。
- 失败上限：`B_FAIL_MIN=3`、`B_FAIL_FRAC=0.0025`（N=1284 时最多 4 个）⇒ 超过即 HARD ERROR；
  N<100 时**任何** B 失败都是硬错误。
- 完成判据：`receipt.json` + `logits.json` 存在**且** `log.txt` 含 `WORKER_DONE`。
- 收据必须满足 `status == "PASS"`、`studies == len(part)`、`amp == "fp16"`、
  `models == ["E_prior_s42_fall.pt"]`、`spatial_models == 1`。

---

## 5. ⚠️ 预期管理（不要自欺）

按 C184 的公榜噪声结论（半集 SE ≈ 0.003，大于前 400 名的总分差 0.0022），
**这条腿在公榜上大概率【测不出差异】**（显示分可能仍是 0.958）。它真正买到的是：
- **模型多样性 ⇒ 私榜泛化**（私榜才决定奖牌）；
- **非 OAI 血统 ⇒ 合规对冲**。
⇒ **不要用"公榜没动"去判它失败**，也不要用"公榜动了"去判它成功。
