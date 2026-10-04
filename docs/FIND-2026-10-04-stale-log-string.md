# ⚠️ 一条会骗人的日志：`[raptor-fast] balanced arm groups cuda:0=[0,2], cuda:1=[1,3]`

**2026-10-04，在验收 `0943cfmsft2` 时发现。**

## 现象

`kernel-0943cfmsft2`（六视角）的日志打印：

```
[raptor-fast] 3 studies; balanced arm groups cuda:0=[0,2], cuda:1=[1,3]
```

**只有 4 个臂。** 我们加的 index 4（`cn0`）和 index 5（`msft2`）**不在里面**。
我据此差点写下一条结论：**"cn0 成员从来没跑过，那 0.943 与它无关"**。

## 真相：那是一行**写死的字符串**，不是真实分组

代码里的控制流是：

```python
def gpu_zero():
    with torch.cuda.device(0):
        run_shared_maxspan(torch.device('cuda:0'))   # arms 0 和 2（共享前 6 个 DINOv2 block）
        run_single(4, torch.device('cuda:0'))        # cn0
        run_single(5, torch.device('cuda:0'))        # msft2
def gpu_one():
    with torch.cuda.device(1):
        run_single(1, torch.device('cuda:1'))
        run_single(3, torch.device('cuda:1'))
with _KeThreadPool(max_workers=2) as executor:
    workers = [executor.submit(gpu_zero), executor.submit(gpu_one)]
```

那行 `print(f"[raptor-fast] {len(test_ids)} studies; balanced arm groups cuda:0=[0,2], cuda:1=[1,3]")`
**把分组写死在 f-string 里**，加视角时没人更新它。

⇒ **`run_single(4)`/`run_single(5)` 确实被调用**；cn0 一直在跑，msft2 这次也跑了。
**旁证（独立于代码）**：`0943cfmsft2` 的可见集 `submission.csv` SHA = `095bcf70c756c9fe…`，
与 `0943cf` 的 `a3a6bc58…` **不同** —— 而 A80 记过 `cf` 与 `cfm` 的可见集产物是**逐字节相同**的。
加了第六个视角之后次序变了，说明它进了混合。

## 为什么值得单独记一条

**这是我第一次犯"方向相反"的同类错误。** 本仓反复吃的是"**被接受、被丢弃、不报错的字段**"；
这次是"**一行过期的日志字符串**"。两者形状相同：

> **判据锚在了【自由文本】上，而不是锚在【代码/协议 token】上。**

- A162 那次的教训是：配额判据要锚 `submission-limits` 的输出，不能锚裸 400 的解释。
- 我 10-03 那次是：`-match 'ERROR'` 把 `SSLEOFError` 当终态。
- **这一次是：把一行 `print` 当成了控制流的描述。**

⇒ **可操作的规矩：日志里的"自我描述"行（分组、配置、成员列表）只有在能追到生成它的那行代码时才可信。
验收一个成员是否进栈，判据必须是「代码里有没有 `run_single(<它的 index>)`」+「产物 SHA 有没有变」，
不是「日志里有没有提到它」。**

⚠️ 顺带一条**修正**：我在 `build_0943_addmember.py` 的注释里沿用了"GPU 0 更轻（只 load 一个模型）"
的说法 —— 那是**加视角之前**成立的。现在 GPU 0 = shared maxspan + cn0 + msft2 = **3 个模型**，
GPU 1 = 2 个模型。**配平方向已经反转**，但改动没有重新测量过。
