# effb0mil_lib.py — EffNet-B0 + MIL 2 tầng (lát -> slot -> study) trên cache336_v1
import os, math, time, json, argparse, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F, timm
from sklearn.metrics import roc_auc_score

CACHE = os.environ.get("KNEE_CACHE", "/kaggle/input/datasets/tbongc/rsna-knee-cache336-v1")
SOFT  = os.environ.get("KNEE_SOFT",  "/kaggle/input/datasets/tbongc/rsna-knee-labels/soft_labels_v1.csv")
FOLDS = os.environ.get("KNEE_FOLDS", "/kaggle/input/datasets/tbongc/rsna-knee-labels/folds_used.csv")
OUT   = os.environ.get("KNEE_OUT",   "/kaggle/working")
SL16FIX = os.environ.get("KNEE_SL16FIX", f"{CACHE}/sl16_fix.csv")
SL16NPY = os.environ.get("KNEE_SL16NPY", f"{CACHE}/s_cache_e13_224.npy")

CFG = dict(tag=os.environ.get("KNEE_TAG", "effb0mil_224"), backbone="efficientnet_b0.ra_in1k", img=224, nsl=16,
           epochs=15, warmup=1, min_ep=6, patience=5, bs=4, workers=int(os.environ.get("KNEE_WORKERS", 2)),
           lr_bb=3e-4, lr_head=1e-3, wd=1e-2, drop=0.2, drop_path=0.1, ema=0.998,
           aug_p=0.8, rrc_scale=(0.80, 1.0), seed=int(os.environ.get("KNEE_SEED", 42)),
           jit_z=float(os.environ.get("KNEE_JITZ", 0)),
           jit_mode=os.environ.get("KNEE_JITMODE", "cont"))

SLOT_NAMES = ["sag_fluid", "cor_fluid", "ax_fluid", "sag_t1"]   # thứ tự cache sl16
SLOT_PLANE = [0, 1, 2, 0]             # sag, cor, ax
SLOT_FS    = [1, 1, 1, 0]
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# ------------------------------------------------------------------ data
def load_tables():
    idx = pd.read_csv(SL16FIX)          # StudyInstanceUID,row,flip,m_* (đã sửa laterality + giao 2 mask)
    idx["StudyInstanceUID"] = idx.StudyInstanceUID.astype(str)
    assert list(idx.columns[:3]) == ["StudyInstanceUID", "row", "flip"], idx.columns.tolist()
    assert idx.row.is_monotonic_increasing and idx.row.iloc[-1] == len(idx) - 1, "row phải là 0..N-1 theo thứ tự ids"
    soft = pd.read_csv(SOFT)
    sid = "StudyInstanceUID" if "StudyInstanceUID" in soft.columns else soft.columns[0]
    soft = soft.rename(columns={sid: "StudyInstanceUID"}); soft["StudyInstanceUID"] = soft.StudyInstanceUID.astype(str)
    ycols = [c for c in soft.columns if c.endswith("__y")]
    wcols = [c for c in soft.columns if c.endswith("__w")]
    if not ycols:                          # dạng targets_v3: cột trần + __weight
        wcols = [c for c in soft.columns if c.endswith("__weight")]
        ycols = [c[:-8] for c in wcols]    # cùng thứ tự với wcols
        assert all(c in soft.columns for c in ycols), ycols
    else:
        assert [c[:-3] for c in ycols] == [c[:-3] for c in wcols], (ycols, wcols)
    assert len(ycols) == len(wcols) == 12, (len(ycols), len(wcols))
    folds = pd.read_csv(FOLDS); folds["StudyInstanceUID"] = folds.StudyInstanceUID.astype(str)
    df = (folds.merge(idx[["StudyInstanceUID", "row", "flip"] + [f"m_{s}" for s in SLOT_NAMES]],
                      on="StudyInstanceUID", validate="one_to_one")
               .merge(soft[["StudyInstanceUID"] + ycols + wcols], on="StudyInstanceUID", validate="one_to_one"))
    assert len(df) in (4407, 4349), len(df)
    return df, ycols, wcols

class KneeDS(torch.utils.data.Dataset):
    def __init__(self, df, ycols, wcols):
        self.df = df.reset_index(drop=True)
        self.y = df[ycols].fillna(0).values.astype(np.float32)
        self.w = df[wcols].fillna(0).values.astype(np.float32)
        self.m = df[[f"m_{s}" for s in SLOT_NAMES]].values.astype(np.float32)
        self._mm = {}
    def __len__(self): return len(self.df)
    def _arr(self):
        if "a" not in self._mm:
            self._mm["a"] = np.load(SL16NPY, mmap_mode="r")             # uint8 [4407,4,16,224,224]
        return self._mm["a"]
    def __getitem__(self, i):
        r = self.df.iloc[i]
        x = self._arr()[int(r.row)]                                     # uint8 [4,16,224,224]
        if int(r.flip):                                                 # cache sl16 KHÔNG lật; đưa về quy ước cache336_v1
            x = x[:, :, :, ::-1]
        x = np.ascontiguousarray(x)
        return torch.from_numpy(x), torch.from_numpy(self.m[i]), torch.from_numpy(self.y[i]), torch.from_numpy(self.w[i])

# ------------------------------------------------------------------ GPU preprocessing (dùng chung train/val/infer)
def to_input(x, mask, img=224, train=False, cfg=CFG):
    """x uint8 [B,5,12,H,W] (cuda) -> float [B,5,12,3,img,img] chuẩn hoá; KHÔNG lật ngang."""
    B, S, N, H, W = x.shape
    v = x.float().div_(255.0).view(B * S, N, H, W)
    theta = torch.zeros(B * S, 2, 3, device=x.device); theta[:, 0, 0] = 1; theta[:, 1, 1] = 1
    if train:
        sc = torch.empty(B * S, device=x.device).uniform_(*cfg["rrc_scale"])
        ar = torch.exp(torch.empty(B * S, device=x.device).uniform_(-0.1, 0.1))
        sx, sy = (sc * ar).clamp(max=1.0), (sc / ar).clamp(max=1.0)
        theta[:, 0, 0], theta[:, 1, 1] = sx, sy
        theta[:, 0, 2] = (torch.rand(B * S, device=x.device) * 2 - 1) * (1 - sx)
        theta[:, 1, 2] = (torch.rand(B * S, device=x.device) * 2 - 1) * (1 - sy)
    grid = F.affine_grid(theta, (B * S, N, img, img), align_corners=False)
    v = F.grid_sample(v, grid, mode="bilinear", padding_mode="zeros", align_corners=False)  # [B*S,N,img,img]
    if train:   # aug cường độ theo slot, chống nhớ chữ ký scanner
        on = (torch.rand(B * S, 1, 1, 1, device=x.device) < cfg["aug_p"]).float()
        g  = torch.exp(torch.empty(B * S, 1, 1, 1, device=x.device).uniform_(-0.3, 0.3))
        a  = torch.empty(B * S, 1, 1, 1, device=x.device).uniform_(0.8, 1.2)
        b  = torch.empty(B * S, 1, 1, 1, device=x.device).uniform_(-0.1, 0.1)
        sd = torch.empty(B * S, 1, 1, 1, device=x.device).uniform_(0, 0.03)
        va = (v.clamp(0, 1) ** g) * a + b + torch.randn_like(v) * sd
        v = (on * va + (1 - on) * v).clamp(0, 1)
    c = torch.arange(N, device=x.device)
    J, MODE = float(cfg.get("jit_z", 0)), str(cfg.get("jit_mode", "cont"))
    if train and J > 0:                    # random slice jitter doc truc z
        bi  = torch.arange(B * S, device=x.device)[:, None, None]
        off = torch.tensor([-1.0, 0.0, 1.0], device=x.device)
        if MODE == "disc":                 # dich nguyen lat
            d  = torch.randint(-int(J), int(J) + 1, (B * S, 1), device=x.device).float()
            p  = (c[None, :, None].float() + d[:, :, None] + off[None, None, :]).clamp(0, N - 1).long()
            v  = v[bi, p]
        else:                              # continuous: noi suy tuyen tinh giua 2 lat ke nhau
            d  = (torch.rand(B * S, 1, device=x.device) * 2 - 1) * J
            p  = (c[None, :, None].float() + d[:, :, None] + off[None, None, :]).clamp(0, N - 1)
            lo = p.floor().long(); hi = (lo + 1).clamp(max=N - 1)
            f  = (p - lo.float())[..., None, None]
            v  = v[bi, lo] * (1 - f) + v[bi, hi] * f
    else:
        tri = torch.stack([(c - 1).clamp(0, N - 1), c, (c + 1).clamp(0, N - 1)], 1)   # [N,3] cửa sổ 2.5D
        v = v[:, tri]                                                                  # [B*S,N,3,img,img]
    v = (v - IMNET_MEAN.to(v.device)[None]) / IMNET_STD.to(v.device)[None]
    return v.view(B, S, N, 3, img, img)

# ------------------------------------------------------------------ model
class EffB0MIL(nn.Module):
    def __init__(self, cfg=CFG, pretrained=True, nlab=12):
        super().__init__()
        self.cfg = cfg
        self.body = timm.create_model(cfg["backbone"], pretrained=pretrained, num_classes=0,
                                      global_pool="avg", drop_path_rate=cfg["drop_path"])
        D = self.body.num_features; self.D = D
        self.emb_plane = nn.Embedding(3, D); self.emb_fs = nn.Embedding(2, D); self.emb_pos = nn.Linear(1, D)
        for e in (self.emb_plane, self.emb_fs): nn.init.zeros_(e.weight)
        nn.init.zeros_(self.emb_pos.weight); nn.init.zeros_(self.emb_pos.bias)
        self.norm = nn.LayerNorm(D)
        self.q_win  = nn.Parameter(torch.randn(nlab, D) * 0.02)
        self.q_slot = nn.Parameter(torch.randn(nlab, D) * 0.02)
        self.t_win  = nn.Parameter(torch.zeros(1))          # log nhiệt độ, exp(0)=1 -> như cũ
        self.t_slot = nn.Parameter(torch.zeros(1))
        self.slot_prior = nn.Parameter(torch.zeros(nlab, len(SLOT_PLANE)))
        self.drop = nn.Dropout(cfg["drop"])
        self.head = nn.Linear(D, nlab)
        self.register_buffer("slot_plane", torch.tensor(SLOT_PLANE), persistent=False)
        self.register_buffer("slot_fs", torch.tensor(SLOT_FS), persistent=False)

    def forward(self, v, mask):
        """v [B,S,N,3,h,w] float, mask [B,S] -> logit [B,12]"""
        B, S, N = v.shape[:3]
        valid = mask > 0
        f = self.body(v[valid].flatten(0, 1))                                  # [V*N, D]
        h = torch.zeros(B, S, N, self.D, device=f.device, dtype=f.dtype)
        h[valid] = f.view(-1, N, self.D)
        pos = torch.linspace(0, 1, N, device=f.device).view(N, 1)
        h = h + (self.emb_plane(self.slot_plane) + self.emb_fs(self.slot_fs)).view(1, S, 1, -1) \
              + self.emb_pos(pos).view(1, 1, N, -1)
        h = self.norm(h).float()
        sq = math.sqrt(self.D)
        a1 = (torch.einsum("bsnd,ld->bsln", h, self.q_win.float()) / sq * self.t_win.exp()).softmax(-1)      # lát trong slot
        p1 = torch.einsum("bsln,bsnd->bsld", a1, h)                                        # [B,S,L,D]
        a2 = (torch.einsum("bsld,ld->bls", p1, self.q_slot.float()) / sq * self.t_slot.exp()
              + self.slot_prior.float().unsqueeze(0))
        a2 = a2.masked_fill(~valid[:, None, :], -1e4).softmax(-1)                          # slot trong study
        p2 = self.drop(torch.einsum("bls,bsld->bld", a2, p1))                              # [B,L,D]
        return (self.head.weight.float() * p2).sum(-1) + self.head.bias.float()

# ------------------------------------------------------------------ train / eval
def soft_auc(Y, P):
    yb = (Y >= 0.5).astype(int); out = []
    for j in range(Y.shape[1]):
        out.append(roc_auc_score(yb[:, j], P[:, j]) if 0 < yb[:, j].sum() < len(yb) else np.nan)
    return float(np.nanmean(out)), out

@torch.no_grad()
def predict(model, loader, dev, img):
    model.eval(); P = []
    for x, m, y, w in loader:
        x, m = x.to(dev, non_blocking=True), m.to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            P.append(torch.sigmoid(model(to_input(x, m, img, train=False), m)).float().cpu().numpy())
    return np.concatenate(P)

def train_fold(fold, cfg=CFG, max_steps=None, val_limit=None):
    torch.manual_seed(cfg["seed"] + fold); np.random.seed(cfg["seed"] + fold)
    dev = "cuda"; tag = cfg["tag"]
    df, ycols, wcols = load_tables()
    tr, va = df[df.fold != fold], df[df.fold == fold]
    if val_limit: va = va.iloc[:val_limit]
    dl_tr = torch.utils.data.DataLoader(KneeDS(tr, ycols, wcols), batch_size=cfg["bs"], shuffle=True, drop_last=True,
                                        num_workers=cfg["workers"], pin_memory=True, persistent_workers=True)
    dl_va = torch.utils.data.DataLoader(KneeDS(va, ycols, wcols), batch_size=cfg["bs"], shuffle=False,
                                        num_workers=cfg["workers"], pin_memory=True)
    Yva = va[ycols].fillna(0).values
    model = EffB0MIL(cfg).to(dev).to(memory_format=torch.channels_last)
    ema = timm.utils.ModelEmaV2(model, decay=cfg["ema"])
    bb = [p for n, p in model.named_parameters() if n.startswith("body.")]
    hd = [p for n, p in model.named_parameters() if not n.startswith("body.")]
    opt = torch.optim.AdamW([{"params": bb, "lr": cfg["lr_bb"]}, {"params": hd, "lr": cfg["lr_head"]}], weight_decay=cfg["wd"])
    spe = max_steps or len(dl_tr); total = spe * cfg["epochs"]; warm = spe * cfg["warmup"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm
                                              else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.cuda.amp.GradScaler()
    ckp, bestp = f"{OUT}/ck_{tag}_f{fold}.pt", f"{OUT}/best_{tag}_f{fold}.pt"
    ep0, hist, best, bad = 0, [], -1.0, 0
    if max_steps is None and os.path.exists(ckp):
        ck = torch.load(ckp, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); ema.module.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        ep0, hist, best, bad = ck["ep"] + 1, ck["hist"], ck["best"], ck["bad"]
        print(f"[f{fold}] resume từ ep{ck['ep']} best {best:.4f}", flush=True)
    for ep in range(ep0, cfg["epochs"]):
        model.train(); t0 = time.time(); tl = 0.0
        for it, (x, m, y, w) in enumerate(dl_tr):
            if max_steps and it >= max_steps: break
            x, m, y, w = x.to(dev, non_blocking=True), m.to(dev), y.to(dev), w.to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                logit = model(to_input(x, m, cfg["img"], train=True), m)
            loss = (F.binary_cross_entropy_with_logits(logit.float(), y, reduction="none") * w).sum() / w.sum().clamp_min(1.0)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt); scaler.update(); sched.step(); ema.update(model)
            tl += loss.item()
            if it % 200 == 0: print(f"[f{fold}] ep{ep} it{it}/{spe} loss {tl/(it+1):.4f} {(time.time()-t0)/(it+1):.2f}s/it", flush=True)
        P = predict(ema.module, dl_va, dev, cfg["img"])
        soft, per = soft_auc(Yva, P)
        hist.append(dict(ep=ep, loss=tl / spe, soft=soft, per=per, min=(time.time() - t0) / 60))
        print(f"[f{fold}] ep{ep} loss {tl/spe:.4f} soft {soft:.4f} | {(time.time()-t0)/60:.1f}m", flush=True)
        if max_steps: return dict(soft=soft, sec_per_it=(time.time() - t0) / spe, P=P)
        if soft > best:
            best, bad = soft, 0
            torch.save(dict(model=ema.module.state_dict(), cfg=cfg, ep=ep, auc=soft, fold=fold), bestp)
            np.save(f"{OUT}/oof_{tag}_f{fold}.npy", P)
            va[["StudyInstanceUID"]].to_csv(f"{OUT}/oof_{tag}_f{fold}_ids.csv", index=False)
        else:
            bad += 1
        torch.save(dict(model=model.state_dict(), ema=ema.module.state_dict(), opt=opt.state_dict(),
                        sched=sched.state_dict(), scaler=scaler.state_dict(), ep=ep, hist=hist, best=best, bad=bad), ckp)
        if ep + 1 >= cfg["min_ep"] and bad >= cfg["patience"]:
            print(f"[f{fold}] early stop ep{ep}", flush=True); break
    json.dump(dict(fold=fold, best=best, best_ep=int(np.argmax([h["soft"] for h in hist])), hist=hist),
              open(f"{OUT}/result_{tag}_f{fold}.json", "w"))
    open(f"{OUT}/done_{tag}_f{fold}", "w").write(str(best))
    print(f"[f{fold}] XONG best {best:.4f}", flush=True)
    return best

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, required=True)
    a = ap.parse_args()
    if os.path.exists(f"{OUT}/done_{CFG['tag']}_f{a.fold}"): print(f"f{a.fold} đã xong, bỏ qua")
    else: train_fold(a.fold)
