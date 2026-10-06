# 公开面的天花板就是 0.944 —— 而 0.944 恰好是铜牌线 — 2026-10-06

方法：拉下 `kernels_list(competition=..., sort_by='dateRun')` 最新的 14 本 notebook，
只读 markdown 里**自述的分数**，逐条核对上下文（不看标题、不看 URL 里的数字）。

## 结果

| notebook | 自述最高分 | 说明 |
|---|---|---|
| `goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944` | **0.944** | 我们 reader 腿的配方来源 |
| `sourabhnirwanevht/rsna-knee-blend-gp256x15` | **0.944** | 配方描述与上式**逐字相同** |
| `sameerk2004/rsna-knee-sil-d2-dino-v5` | — | 明确写 **Base = goodpjw2008 的 0.944 管线** |
| `sameerk2004/rsna-knee-sil-d2-mcl-latmen` | **0.943** | 带自我更正：*"this scores 0.943, not 0.946"* |
| `sameerk2004/rsna-knee-sil-d2-meniscus10` | 0.943 | |
| `chenchinglong/rsna-kad-bench-0942-*`（3 本） | 0.942 | 同族拆解版 |
| `tensorhydra/rsna-knee-d4blend-fork` | 0.943 | |

**⇒ 没有任何一条经得起核对的自述 > 0.944。**

⚠️ 那条 `0.946` 值得单独记：`sameerk2004/rsna-knee-sil-d2-mcl-latmen` 页面里自己写着
*"The `0946` in this notebook's URL is inherited from ..."*，**更正为 0.943**。
这正是本仓反复记过的那个病 —— **读名字而不是读产物**。URL/标题里的数字不是证据。

## 结论

1. **公开材料能到达的上限是 0.944**，而 `TARGET-2026-10-06j` 已经证明 **0.944 就是铜牌线**
   （rank 319..504，整段在线内）。
   ⇒ **铜牌可达；更高不可达**（0.945 起那 63 支队伍、一直到 0.964，靠的是公开面之外的东西）。

2. **这同时确认了在途那一枪选对了方向**：0.944 的公开配方就是"0943 栈 + 2.5D ConvNeXt MIL reader，
   70/30 rank 混合"，而我们的 reader 腿（提交 `56872230`）用的正是这个配方的 reader 检查点。

3. **也说明没有必要再去公开面里找下一根杠杆了**：天花板已经扫到，
   继续找只会重复 `docs/SWEEP-2026-10-06d-public-surface-converged.md` 的结论。
   真正的下一步是**把在途两个读数拿到并按预注册判定**，不是再挖。

## 对目标的含义

* 目标从"尽量提升"收敛为**一个可判定的状态**：reader 腿落在 **0.944 ⇒ 进铜牌区**；
  落在 0.943 ⇒ 仍在线外（990 队挤在一起）；落在更低 ⇒ 丢掉这条腿、配对不变。
* 若 reader 腿失败（9h 上限）或落在 0.943，**本仓在当前约束下的最优已知位置就是"铜牌线附近"**，
  这应当如实记录，而不是用更多投机提交去掩盖。
