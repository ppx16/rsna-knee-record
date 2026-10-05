# FIND 2026-10-05 — the "F_dim 1024 vs 768" block was a DEAD DEFAULT ARGUMENT, not a mismatch

**Status: resolved. There is no dimension mismatch in the Raptor family, and there never was.
The recorded blocker was read out of a function signature instead of off a call site.**

---

## 1. The puzzle as it was recorded

`docs/FIND-2026-10-05-unused-strong-checkpoints.md` §1 carried this warning verbatim:

> ⚠️ `coatnet384` 与 `cnv2b336` 的头维度都是 **1024**，而 v5/v8/v10（同为 `coatnet_rmlp_2_rw_384`）
> 在我们栈里走的是 `F_dim=768` 的加载路径。这两者不是同一套配置，**接入前必须把这个维度对齐搞清楚，
> 否则会静默加载失败或错配。**

That is a serious claim — it would mean the whole `raptor-knee-arms` family cannot be wired safely —
so it was worth one cold look at the weights. **It does not survive contact with the artifacts.**

## 2. What the weights actually say

Head tensors, measured (`work/resolve_fdim.py`) on real files:

| checkpoint | `att.0.weight` | `clsW` | `norm.weight` | backbone terminal |
|---|---|---|---|---|
| `raptor_ft_coatnet384.pt` (theirs) | (256, **1024**) | (12, **1024**) | (1024,) | `backbone.norm` (1024,) |
| `raptor_ft_cnv2b336.pt` (theirs) | (256, **1024**) | (12, **1024**) | (1024,) | `backbone.head.norm` (1024,) |
| **`v5.pt` (ours)** | (256, **1024**) | (12, **1024**) | (1024,) | `backbone.norm` (1024,) |
| **`v8.pt` (ours)** | (256, **1024**) | (12, **1024**) | (1024,) | `backbone.norm` (1024,) |
| **`v10.pt` (ours)** | (256, **1024**) | (12, **1024**) | (1024,) | `backbone.norm` (1024,) |
| `raptor_ft_cn0_swa.pt` (ours) | (256, **1024**) | (12, **1024**) | (1024,) | `backbone.norm` (1024,) |
| `raptor_ft_msft0_swa.pt` (ours) | (256, **1024**) | (12, **1024**) | (1024,) | `backbone.norm` (1024,) |
| `raptor_ft_effv2l480.pt` (theirs) | (256, **1280**) | (12, **1280**) | (1280,) | `backbone.conv_head`+`bn2` |

**Our v5/v8/v10 are 1024, not 768.** And `timm` confirms the architecture:
`coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` → `num_features = 1024`.

⇒ **"1024 vs 768" was never a comparison between two things. Both sides were 1024.**

## 3. Where 768 came from — and why it is inert

`kernels/kernel-0943cfmsft2m/rsna-knee-restructured-version-3.ipynb`:

```python
4739: class RaptorClassifier(nn.Module):
4741:     def __init__(self, backbone, F_dim=768, n=12, drop=0.2):   # <-- a DEFAULT
4744:         self.norm = nn.LayerNorm(F_dim)
4745:         self.att  = nn.Sequential(nn.Linear(F_dim, 256), ...)
4746:         self.clsW = nn.Parameter(torch.zeros(n, F_dim))

4773:     bb = build_backbone(arch, pretrained=False)
4774:     model = RaptorClassifier(bb, F_dim=bb.num_features)        # <-- the ONLY call site
4775:     rsna_strict_load(model, ck['model'], str(pt_path))
```

`768` appears **exactly once in the file: as a default parameter value.** The single call site passes
`F_dim=bb.num_features`, which for every CoAtNet and ConvNeXtV2 checkpoint here is **1024** and for
EffNetV2-L is **1280**. The default is never reached.

**And the load is strict.** `rsna_strict_load` would raise on any shape disagreement. So even if the
widths had genuinely differed, the failure mode would have been a **loud exception at load**, not the
silent mispredict this note feared. Both halves of the warning were wrong: there was no mismatch, and
the mismatch would not have been silent.

## 4. What genuinely differs, now that the phantom is cleared

The family **is** wireable, but three real differences remain and they are the ones to plan around:

1. **`arch` must match the per-arm declaration.** `load_model` raises
   `RuntimeError('Raptor architecture drift: ...')` if `ck['arch'] != arch_default`. So
   `coatnet384` drops into a `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` arm unchanged; **`cnv2b336` and
   `effv2l480` each need their own arch string** (`convnextv2_base...`, `tf_efficientnetv2_l...`).
2. **`effv2l480` is genuinely a different width: F_dim 1280**, and its tail is EfficientNet-shaped
   (`conv_head` + `bn2`) rather than `backbone.norm`. It is the only one of the three that is not
   head-compatible by construction — and it is also the weakest (`gold_auc` 0.8716).
3. **Backbone tail prefixes differ** (`backbone.norm.*` for CoAtNet vs `backbone.head.norm.*` for
   ConvNeXtV2). This is absorbed by timm's own `num_classes=0` pooling, so no loader change is needed —
   but it is why a naive `state_dict` prefix match would look broken.

⇒ **Consequence for §4 of `FIND-2026-10-05-raptor-geometry-confirmed.md`:** the description's claim that
all three arms share *"the same per-finding attention pooling head"* is now **corroborated at the tensor
level** — identical head shapes across two different backbone families. The reason not to wire them is
still the measured one (the blended family scores 0.915 against our 0.943), **not** a wiring risk.

## 5. This is the THIRD instance of one disease, in this session alone

| # | where | the *stated* thing | the *effective* thing |
|---|---|---|---|
| 1 | `CROP_MM` | first/most-frequent grep hit `= 130.0` | per-block global; Raptor block assigns **140.0** |
| 2 | `[raptor-fast] balanced arm groups cuda:0=[0,2]` | the printed grouping | a hardcoded stale print (`FIND-2026-10-04-stale-log-string.md`) |
| 3 | `F_dim=768` | a parameter in the signature | the call site passes `bb.num_features` = **1024** |

Each time the wrong answer came from reading **a declaration** rather than **the path that executes**.
The generalisation is worth stating plainly:

> **A grep finds where a value is written. It does not tell you whether that value is ever used.
> For any constant that gates behaviour, read the call site, not the definition.**

Practically: when a note says "X is set to V, so Y", the note must name the **file and line of the
assignment that wins**, or it is not yet a finding. Both #1 and #3 would have been caught immediately
by that requirement.

Tooling: `work/resolve_fdim.py` — dumps head/tail tensor shapes plus `cfg` scalars for any set of Raptor
checkpoints and cross-checks `timm`'s own `num_features`.
