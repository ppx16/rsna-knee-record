"""Append C191 (idempotent): the S6 arm's infrastructure works, its inference fell back."""
import io, os

p = 'state/FACTS.md'
ADD = """| C191 | 🔧 **【S6 全量腿第一次跑：准备阶段全部成功,推理阶段 `ALL_FALLBACK`】** —— 基础设施打通了,失败点被精确定位到"子进程找不到 extras 的代码"。<br>**① 成功的部分（逐字,来自 `S6_PREP`）**：`{"status": "prepared", "extras_members": 31, "bundle": "/kaggle/working/_s6_bundle", "sha256": "0c7297d58f061ab4e5d7c55046f51fa3e8934928046f0647a0c7854e98371c12"}`<br>⇒ **三件事同时被证实**：**(a)** 权重在挂载的数据集里被找到且 **sha256 与 README 一致**;**(b)** `s6_extras.bin` 的 `K1PAYLOAD1\\n` + 未压缩 tar 格式**正确**,解出 **31 个成员**;**(c)** 我们自己合成的 `manifest.json` 被基座的 stage-2 **接受**（`assert manifest['bundle_id'] == BUNDLE_ID` 与逐文件 sha256 都过了）。**⇒ C190④ 那条"manifest 驱动,不需要改推理"的判断成立。**<br>**② 失败的部分**：`S6_STAGE_FALLBACK AssertionError: S6 receipt status 'ALL_FALLBACK'` ⇒ 控制器跑完了,但**每个研究都回退了**（worker 级失败）,于是 `[s6] stage did not run -> NOT folded`,**成品仍是我们原来的 0.958 那一枪**（失败安全按设计工作）。<br>**③ ⇒ 根因判断（下一次要验证的）**：我给子进程传了 `--code <extras>:<heliosli>`，但 **`PYTHONPATH` 里只放了 heliosli 的目录**。worker 是按**模块名**导入的（`import compact_infer` 之类），所以 extras 目录必须在 `PYTHONPATH` 上 —— **只通过 `--code` 传是不够的**。已修：`PYTHONPATH = _s6_code : CODE : ...`，**并加了 worker 日志捕获**（`s6_worker*_log.txt` 会被复制进输出），否则 `ALL_FALLBACK` 无法诊断。<br>**④ ⚠️ 另一个待排除的候选**：基座现役的 arm 是 **`E_lrprior`**（带 lr-prior），而 S6 是 **`E_prior`**。若 `compact_infer.py` 的这个版本不认识 `E_prior` 头,修 PYTHONPATH 也不够 —— **那时日志里会看到 KeyError/形状不匹配**,可用 extras 里自带的 `compact_*` 覆盖 heliosli 的版本。<br>**⑤ 额度账**：今天交了三枪（`57290853` A224=0.50、`57290856` A224=0.65、`57291086` 0.958+r9all@0.10）,**剩 2 次**。**S6 那一枪在它真的 blended 之前不交**（回落版与手上的 0.958 完全相同,不值得花额度）。 | 2026-10-17 |
"""

if '| C191 |' in io.open(p, encoding='utf-8').read():
    raise SystemExit('REFUSING: C191 is already present')
with io.open(p, 'a', encoding='utf-8') as fh:
    fh.write(ADD)
print('appended C191; file now', os.path.getsize(p), 'bytes')
