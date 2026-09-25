"""
Модель CoAtNet для RSNA Knee: 12 диагнозов по МРТ колена.

Это инференс-часть. Модель обучена заранее, веса загружаются из датасета:
https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense

Метки для обучения получены из текстовых отчётов с помощью языковой модели.
Отчёты превращены в 12 вероятностей, а не жёсткие 0/1. Это дало 4349
размеченных исследований. Официальные 58 исследований в обучение не попали,
на них модель даёт 0.9167 macro-AUC.

Каждое исследование приводится к фиксированному стеку из 64 срезов:
  18 срезов из сагиттальной серии (предпочтительно жидкостно-чувствительной)
  14 срезов из второй сагиттальной (предпочтительно не жидкостно-чувствительной)
  12 срезов из корональной (предпочтительно жидкостно-чувствительной)
   8 срезов из второй корональной
  12 срезов из аксиальной

Если серии для слота нет, слот заполняется нулями и модель получает маску,
чтобы пропустить его. Срезы берутся из диапазона 6–94% стека.

Каждый срез обрезается до 140 мм вокруг центра и приводится к 336 пикселям.
Три соседних среза объединяются в 3 канала. Окна проходят через CoAtNet.
Внимание по 12 диагнозам выбирает, какие окна важнее для каждого диагноза.

Инференс использует 42 окна на исследование, выполняется в fp16,
при ошибке автоматически повторяется в fp32. Интернет не требуется.
"""
import os, sys, glob, time, json, gc
from concurrent.futures import ThreadPoolExecutor
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import timm
# T4 (Turing) cuDNN v9 has fp16/fp32 conv engines but NOT bf16 for these shapes
# ("GET was unable to find an engine..."); benchmark lets it pick a valid algo for
# the fixed (1,24,3,res,res) input.
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# ---- fixed config (must match training exactly) -----------------------------
IMG = 336
CROP_MM = 140.0
# 64 slices per study instead of 44, same proportions. Must match the corpus the weights
# were trained on (knee_corpus_v4.py).
SLOTS = [("Sagittal", 1, 18), ("Sagittal", 0, 14), ("Coronal", 1, 12),
         ("Coronal", 0, 8), ("Axial", -1, 12)]
MAXS = sum(s[2] for s in SLOTS)                     # 64
K_EVAL = 64   # every window position the volume holds, not an evenly spaced subset
NORM = "imagenet"
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# Комментарий к выбору моделей в CoAtNet ветке.

# Что здесь написано:
# Это описание того, как автор выбрал одну модель (архетип coatnet_rmlp_2_rw_384)
# для финального решения. Он экспериментировал с 7 разными архитектурами и проверял их
# на 45 золотых исследованиях (подмножество из 58 доступных).

# Результаты одиночных моделей:
# - CoAtNet 384 — 0.9025 (лучшая)
# - SwinBase 384 — 0.8825
# - EffNetV2L 480 — 0.8716

# Результаты смесей (блендов):
# - CoAtNet + Swin + EffNetV2L — 0.9068
# - CoAtNet + Swin — 0.9059
# - CoAtNet один — 0.9025

# Что автор решил:
# Автор выбрал один CoAtNet, потому что:
# 1. На публичном лидерборде CoAtNet один дал 0.914, а любая смесь давала 0.914–0.915.
#    Прирост от смешивания всего +0.001, что он считает шумом.
# 2. Один CoAtNet требует в 3 раза меньше времени на инференс, чем смесь из трёх.

# Почему не стал добавлять другие модели:
# Он отбросил как избыточные:
# - cnn336 (0.8833)
# - cnbase384 (0.8754)
# - cnlarge384 (0.8752)
# - maxvit384 (0.8438)

# Расширение обучающего корпуса:
# Автор расширил обучающий набор с 3200 до 4349 исследований за счёт использования отчётов
# для генерации мягких меток. Это дало прирост:
# - Старый CoAtNet (обучен на 3155 исследованиях) — 0.8923 на 58 золотых.
# - Новый CoAtNet (обучен на 4349 исследованиях) — 0.9054 (+0.0131).

# Где самый большой прирост:
# Наибольший рост качества после расширения корпуса:
# - Lateral Meniscus — +0.071
# - Fracture — +0.057
# - Lateral OA — +0.048
# - Medial Meniscus — +0.035
# - ACL — +0.028

# Это были самые слабые метки, и расширение данных помогло им сильнее всего.

_SLOTS64 = [("Sagittal", 1, 18), ("Sagittal", 0, 14),
            ("Coronal", 1, 12), ("Coronal", 0, 8), ("Axial", -1, 12)]
_SLOTS44 = [("Sagittal", 1, 12), ("Sagittal", 0, 10),
            ("Coronal", 1, 8), ("Coronal", 0, 6), ("Axial", -1, 8)]

ARMS = [
    {"name": "maxspan-v5", "file": "raptor_ft_coatnet_v5_full_swa.pt",
     "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384,
     "img": 336, "slots": _SLOTS64, "span": (0.02, 0.98), "k_eval": 62,
     "reverse": False, "w": 0.55},
    {"name": "native384dense-v10", "file": "raptor_ft_coatnet_v10_full.pt",
     "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384,
     "img": 384, "slots": _SLOTS64, "span": (0.02, 0.98), "k_eval": 62,
     "reverse": False, "w": 0.10},
    {"name": "maxspan-v5-reverse", "file": "raptor_ft_coatnet_v5_full_swa.pt",
     "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384,
     "img": 336, "slots": _SLOTS64, "span": (0.02, 0.98), "k_eval": 62,
     "reverse": True, "w": 0.15},
    {"name": "native384-v8", "file": "raptor_ft_coatnet_v8_full_swa.pt",
     "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384,
     "img": 384, "slots": _SLOTS44, "span": (0.06, 0.94), "k_eval": 42,
     "reverse": False, "w": 0.20},
]

LEGACY_ARM = {
    "file": "raptor_ft_coatnet_v4_full.pt",
    "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",
    "res": 384,
    "k_eval": 42,
    "span_lo": 0.06,
    "span_hi": 0.94,
}

PRIMARY_SPAN_LO = 0.02
PRIMARY_SPAN_HI = 0.98


# ============================================================================
# Model -- verbatim from finetune_raptor.py
# ============================================================================
def build_backbone(arch, pretrained=False):
    # maxvit/maxxvit/coatnet are conv-attention hybrids: NO CLS token, NO interpolatable
    # pos-embed -> avg pool. The "vit" substring in "coatnet"/"maxvit" must NOT route them
    # down the ViT path (mirrors finetune_raptor.py exactly).
    hybrid = arch.startswith(("maxvit", "maxxvit", "coatnet", "coat_", "convnext"))
    is_vit = (not hybrid) and any(k in arch for k in ("vit", "deit", "dinov2", "eva", "beit"))
    kw = dict(pretrained=pretrained, num_classes=0, in_chans=3)
    if is_vit:
        kw.update(global_pool="token", dynamic_img_size=True)
    else:
        kw.update(global_pool="avg")
    return timm.create_model(arch, **kw)


class RaptorClassifier(nn.Module):
    def __init__(self, backbone, F_dim=768, n=12, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(F_dim)
        self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop),
                                 nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, F_dim))
        self.clsb = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.clsW, std=0.02)
        self.n = n

    def encode(self, x):
        B, K = x.shape[:2]
        f = self.backbone(x.flatten(0, 1))
        return f.view(B, K, -1)

    def head(self, feats):
        h = self.norm(feats)
        a = self.att(h)
        a = torch.softmax(a, dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        logits = (pooled * self.clsW).sum(-1) + self.clsb
        return logits

    def forward(self, x):
        return self.head(self.encode(x))


def load_model(pt_path, arch_default, res_default, device, ngpu=1):
    ck = torch.load(pt_path, map_location="cpu", weights_only=False)
    arch = ck.get("arch", arch_default)
    ck_res = int(ck.get("res", res_default))
    bb = build_backbone(arch, pretrained=False)
    model = RaptorClassifier(bb, F_dim=bb.num_features)
    model.load_state_dict(ck["model"], strict=True)
    model.eval().to(device)
    # NOTE: DataParallel removed on purpose. On the full hidden test it drove a system-RAM OOM
    # (per-forward module replication over many studies); a single T4 handles K_EVAL=24 windows
    # fine. Arms are also run SEQUENTIALLY (see main) so peak RAM == one model, not two.
    del ck
    gc.collect()
    return model, ck_res


# ============================================================================
# Eval windowing -- verbatim from finetune_raptor.py StudyWindows (train=False)
# ============================================================================
def _eval_centers(mask, D, k):
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, D))
    lo, hi = int(valid.min()), int(valid.max())
    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    if not cs:
        cs = [max(1, min((lo + hi) // 2, D - 2))]
    idx = np.linspace(0, len(cs) - 1, k).round().astype(int)
    return [cs[i] for i in idx]


def eval_windows(vol, mask, k, res, norm=NORM):
    D = vol.shape[0]
    cs = _eval_centers(mask, D, k)
    wins = np.empty((len(cs), 3, res, res), np.float32)
    for j, c in enumerate(cs):
        c = max(1, min(c, D - 2))
        tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0
        t = torch.from_numpy(tri)
        if t.shape[-1] != res:
            t = F.interpolate(t[None], size=(res, res), mode="bilinear",
                              align_corners=False)[0]
        wins[j] = t.numpy()
    x = torch.from_numpy(wins)
    if norm == "imagenet":
        x = (x - _MEAN) / _STD
    return x


@torch.no_grad()
def infer_probs(model, xwins, device):
    x = xwins.unsqueeze(0).to(
        device,
        non_blocking=True,
    )

    use_cuda = (
        str(device).startswith("cuda")
    )

    def _forward():
        return torch.sigmoid(
            model(x).float()
        )[0].cpu().numpy()

    if use_cuda:
        try:
            with torch.autocast(
                "cuda",
                dtype=torch.float16,
            ):
                return _forward()

        except RuntimeError as error:
            try:
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
            except Exception:
                pass

            print(
                "[DINOsaur V4.2] "
                f"{device} fp16 retry in fp32: "
                f"{type(error).__name__}",
                flush=True,
            )

            return _forward()

    return _forward()


def rankpct(x):                                   # per-column percentile rank in [0,1]
    order = x.argsort(0).argsort(0).astype(np.float64)
    return order / max(1, (x.shape[0] - 1))


# ============================================================================
# Preprocessing -- verbatim from kprep2/dino_preprocess.py, retargeted to TEST
# ============================================================================
def _make_reader():
    import pydicom, cv2
    from pydicom.pixel_data_handlers.util import apply_modality_lut

    def order_and_meta(sdir):
        fs = glob.glob(sdir + "/*.dcm"); recs = []; ps_list = []
        for f in fs:
            try:
                h = pydicom.dcmread(f, stop_before_pixels=True)
                iop = getattr(h, 'ImageOrientationPatient', None)
                ipp = getattr(h, 'ImagePositionPatient', None)
                if iop is not None and ipp is not None and len(iop) == 6:
                    r = np.array(iop[:3], float); c = np.array(iop[3:], float)
                    n = np.cross(r, c); pos = float(np.dot(np.array(ipp, float), n))
                else:
                    pos = float(getattr(h, 'InstanceNumber', 0) or 0)
                ps = getattr(h, 'PixelSpacing', None); ps = float(ps[0]) if ps is not None else 0.5
                ps_list.append(ps); recs.append((pos, f, ps))
            except Exception:
                recs.append((0.0, f, 0.5))
        recs.sort(key=lambda x: x[0])
        med_ps = float(np.median(ps_list)) if ps_list else 0.5
        return [(f, ps) for _, f, ps in recs], med_ps

    def read_px(f):
        d = pydicom.dcmread(f)
        a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
        if str(getattr(d, 'PhotometricInterpretation', '')) == 'MONOCHROME1':
            a = a.max() - a
        return a

    def mm_crop_resize(a, ps):
        h, w = a.shape; cpx = int(round(CROP_MM / max(ps, 1e-3)))
        cpx = min(cpx, min(h, w)); y0 = (h - cpx) // 2; x0 = (w - cpx) // 2
        a = a[y0:y0 + cpx, x0:x0 + cpx]
        return cv2.resize(a, (IMG, IMG), interpolation=cv2.INTER_AREA)

    return order_and_meta, read_px, mm_crop_resize


def _pick_series_for_slot(rows, plane, fluid, used):
    cands = [r for r in rows if r['Anatomical_Plane'] == plane and r['SeriesInstanceUID'] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get('Fluid_Sensitive', 0) or 0) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def _fill_variant_volume(
    target_volume,
    offset,
    picks,
    pixel_cache,
    files,
    med_ps,
    mm_crop_resize,
):
    arrays = []
    spacings = []

    for position in picks:
        position = min(
            int(position),
            len(files) - 1,
        )
        file_path, spacing = files[
            position
        ]
        arrays.append(
            pixel_cache.get(
                position
            )
        )
        spacings.append(
            spacing
            if spacing > 0
            else med_ps
        )

    valid = [
        array
        for array in arrays
        if array is not None
    ]

    if valid:
        all_pixels = np.concatenate(
            [
                array.ravel()
                for array in valid
            ]
        )
        low, high = np.percentile(
            all_pixels,
            [2.0, 98.0],
        )
    else:
        low, high = 0.0, 1.0

    for local_index, (
        array,
        spacing,
    ) in enumerate(
        zip(
            arrays,
            spacings,
        )
    ):
        output_index = (
            offset
            + local_index
        )

        if (
            output_index
            >= MAXS
        ):
            break

        if array is None:
            continue

        normalized = np.clip(
            (
                array - low
            )
            / (
                high - low
                + 1e-6
            ),
            0,
            1,
        )

        normalized = mm_crop_resize(
            normalized,
            spacing,
        )

        target_volume[
            output_index
        ] = (
            normalized
            * 255
        ).astype(
            np.uint8
        )


def build_study_pair(
    sid,
    ser_records,
    tsdir,
    reader,
):
    """
    Produce exact MaxSpan and legacy-span volumes while reading every required
    DICOM only once. Both checkpoints keep their own percentile normalization.
    """
    (
        order_and_meta,
        read_px,
        mm_crop_resize,
    ) = reader

    rows = ser_records.get(
        sid,
        [],
    )

    primary_volume = np.zeros(
        (
            MAXS,
            IMG,
            IMG,
        ),
        np.uint8,
    )
    legacy_volume = np.zeros_like(
        primary_volume
    )

    used = set()
    offset = 0

    for plane, fluid, count in SLOTS:
        record = _pick_series_for_slot(
            rows,
            plane,
            fluid,
            used,
        )

        if record is None:
            offset += count
            continue

        used.add(
            record[
                "SeriesInstanceUID"
            ]
        )

        files, med_ps = order_and_meta(
            f"{tsdir}/{sid}/"
            f"{record['SeriesInstanceUID']}"
        )

        if not files:
            offset += count
            continue

        number = len(files)

        primary_low = int(
            number
            * PRIMARY_SPAN_LO
        )
        primary_high = int(
            number
            * PRIMARY_SPAN_HI
        ) - 1
        primary_high = max(
            primary_high,
            primary_low,
        )

        legacy_low = int(
            number
            * float(
                LEGACY_ARM[
                    "span_lo"
                ]
            )
        )
        legacy_high = int(
            number
            * float(
                LEGACY_ARM[
                    "span_hi"
                ]
            )
        ) - 1
        legacy_high = max(
            legacy_high,
            legacy_low,
        )

        if number > 1:
            primary_picks = np.linspace(
                primary_low,
                primary_high,
                count,
            ).round().astype(int)

            legacy_picks = np.linspace(
                legacy_low,
                legacy_high,
                count,
            ).round().astype(int)
        else:
            primary_picks = np.zeros(
                count,
                dtype=int,
            )
            legacy_picks = np.zeros(
                count,
                dtype=int,
            )

        required_positions = sorted(
            set(
                primary_picks.tolist()
                + legacy_picks.tolist()
            )
        )

        pixel_cache = {}

        for position in required_positions:
            position = min(
                int(position),
                number - 1,
            )

            file_path, _ = files[
                position
            ]

            try:
                pixel_cache[
                    position
                ] = read_px(
                    file_path
                )
            except Exception:
                pixel_cache[
                    position
                ] = None

        _fill_variant_volume(
            primary_volume,
            offset,
            primary_picks,
            pixel_cache,
            files,
            med_ps,
            mm_crop_resize,
        )

        _fill_variant_volume(
            legacy_volume,
            offset,
            legacy_picks,
            pixel_cache,
            files,
            med_ps,
            mm_crop_resize,
        )

        offset += count

        if offset >= MAXS:
            break

    primary_mask = (
        primary_volume.reshape(
            MAXS,
            -1,
        ).sum(1)
        > 0
    ).astype(
        np.uint8
    )

    legacy_mask = (
        legacy_volume.reshape(
            MAXS,
            -1,
        ).sum(1)
        > 0
    ).astype(
        np.uint8
    )

    return (
        primary_volume,
        primary_mask,
        legacy_volume,
        legacy_mask,
    )


# ============================================================================
# Test-root discovery + weights + main
# ============================================================================
def find_test_root():
    cands = ["/kaggle/input/competitions/rsna-knee-abnormality-detection",
             "/kaggle/input/rsna-knee-abnormality-detection"]
    for b in cands:
        if os.path.exists(b + "/test.csv"):
            return b
    for d, _, f in os.walk("/kaggle/input"):
        if "test.csv" in f and (os.path.isdir(d + "/test_series") or os.path.isdir(d + "/test_images")):
            return d
    for d, _, f in os.walk("/kaggle/input"):
        if "test.csv" in f:
            return d
    raise RuntimeError("no test root under /kaggle/input")


def find_weight_file(
    fname,
    required=True,
):
    direct = [
        f"/kaggle/input/raptor-knee-arms/{fname}",
        f"/kaggle/input/raptor-knee-arms/1/{fname}",
        f"/kaggle/input/raptor-cnn336/{fname}",
        f"/kaggle/input/datasets/dreaddevelopment/raptor-knee-maxspan/{fname}",
        f"/kaggle/input/datasets/dreaddevelopment/raptor-knee-native384dense/{fname}",
        f"/kaggle/input/datasets/dreaddevelopment/raptor-knee-native384/{fname}",
        f"/kaggle/input/datasets/dreaddevelopment/raptor-knee-arms/{fname}",
        f"/kaggle/input/datasets/dreaddevelopment/raptor-knee-arms-x/{fname}",
    ]

    for path in direct:
        if os.path.exists(path):
            return path

    for directory in sorted(
        glob.glob(
            "/kaggle/input/*/"
        )
    ):
        if (
            "competition"
            in directory.lower()
        ):
            continue

        hits = glob.glob(
            os.path.join(
                directory,
                "**",
                fname,
            ),
            recursive=True,
        )

        if hits:
            return hits[0]

    if required:
        raise RuntimeError(
            f"{fname} not found "
            "under /kaggle/input"
        )

    return None

def build_study(sid, ser_records, tsdir, reader):
    order_and_meta, read_px, mm_crop_resize = reader
    rows = ser_records.get(sid, [])
    vol = np.zeros((MAXS, IMG, IMG), np.uint8)
    idx = 0
    used = set()
    for plane, fluid, k in SLOTS:
        r = _pick_series_for_slot(rows, plane, fluid, used)
        if r is None:
            idx += k
            continue
        used.add(r['SeriesInstanceUID'])
        files, med_ps = order_and_meta(f"{tsdir}/{sid}/{r['SeriesInstanceUID']}")
        if not files:
            idx += k
            continue
        n = len(files)
        lo, hi = int(n * SPAN_LO), int(n * SPAN_HI) - 1
        hi = max(hi, lo)
        picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k
        arrs = []
        pss = []
        for p in picks:
            fp, ps = files[min(p, n - 1)]
            try:
                arrs.append(read_px(fp))
                pss.append(ps)
            except Exception:
                arrs.append(None)
                pss.append(med_ps)
        valid = [a for a in arrs if a is not None]
        if valid:
            allpx = np.concatenate([a.ravel() for a in valid])
            loq, hiq = np.percentile(allpx, [2.0, 98.0])
        else:
            loq, hiq = 0.0, 1.0
        for a, ps in zip(arrs, pss):
            if idx >= MAXS:
                break
            if a is None:
                idx += 1
                continue
            aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
            aw = mm_crop_resize(aw, ps if ps > 0 else med_ps)
            vol[idx] = (aw * 255).astype(np.uint8)
            idx += 1
        if idx >= MAXS:
            break
    mask = (vol.reshape(MAXS, -1).sum(1) > 0).astype(np.uint8)
    return vol, mask



def main():
    import pandas as pd
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ngpu = torch.cuda.device_count()
    print(f"device {dev} | gpus {ngpu} | torch {torch.__version__}", flush=True)

    ROOT = find_test_root()
    tsdir = ROOT + "/test_series"
    if not os.path.isdir(tsdir):
        tsdir = ROOT + "/test_images"
    print("test root:", ROOT, "| series dir:", tsdir, flush=True)

    test = pd.read_csv(ROOT + "/test.csv"); test["StudyInstanceUID"] = test["StudyInstanceUID"].astype(str)
    test_ids = test["StudyInstanceUID"].tolist()
    tser = pd.read_csv(ROOT + "/test_series.csv")
    tser["StudyInstanceUID"] = tser["StudyInstanceUID"].astype(str)
    tser["SeriesInstanceUID"] = tser["SeriesInstanceUID"].astype(str)
    SER = {k: v.to_dict("records") for k, v in tser.groupby("StudyInstanceUID")}
    print(f"test studies {len(test_ids)} | test series {len(tser)}", flush=True)

    sub_cols = ["StudyInstanceUID"] + LAB
    ssub = os.path.join(ROOT, "sample_submission.csv")
    if os.path.exists(ssub):
        sub_cols = list(pd.read_csv(ssub, nrows=1).columns)

    reader = _make_reader()
    N = len(test_ids)
    A = len(ARMS)
    arm_probs = [np.full((N, len(LAB)), 0.5, np.float32) for _ in range(A)]

    for a, arm in enumerate(ARMS):
        globals()["IMG"] = int(arm["img"])
        globals()["SLOTS"] = list(arm["slots"])
        globals()["MAXS"] = sum(slot[2] for slot in SLOTS)
        globals()["SPAN_LO"], globals()["SPAN_HI"] = map(float, arm["span"])
        globals()["K_EVAL"] = int(arm["k_eval"])
        wp = find_weight_file(arm["file"])
        model, res = load_model(wp, arm["arch"], arm["res"], dev)
        print(f"[arm {a}] {arm['name']} | img {IMG} | slices {MAXS} | span {SPAN_LO:.2f}-{SPAN_HI:.2f} | windows {K_EVAL} | res {res} | {time.time()-t0:.0f}s", flush=True)
        for i, sid in enumerate(test_ids):
            try:
                vol, mask = build_study(sid, SER, tsdir, reader)
                xw = eval_windows(vol, mask, k=K_EVAL, res=res, norm=NORM)
                if bool(arm.get("reverse", False)):
                    xw = xw.flip(1).contiguous()
                arm_probs[a][i] = infer_probs(model, xw, dev)
                del vol, mask, xw
            except Exception as e:
                print(f"  [arm {a}] study {i} {sid[:16]} FALLBACK ({type(e).__name__}: {e})", flush=True)
            if (i + 1) % 100 == 0 or i + 1 == N:
                print(f"  [arm {a}] {i+1}/{N} | {time.time()-t0:.0f}s", flush=True)
        del model
        gc.collect()
        if str(dev).startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"[arm {a}] done + freed | {time.time()-t0:.0f}s", flush=True)

    _w = np.array([float(a.get("w", 1.0)) for a in ARMS], dtype=np.float64)
    _w = _w / _w.sum()
    print(f"[blend] global probability mean w={dict(zip([a['name'] for a in ARMS], _w.round(4)))}", flush=True)
    probability_blend = np.tensordot(
        _w, np.stack([np.clip(p, 0, 1) for p in arm_probs]), axes=(0, 0)
    )
    ranks = rankpct(probability_blend)
    if not np.isfinite(ranks).all():
        ranks[~np.isfinite(ranks)] = 0.5

    sub = pd.DataFrame(ranks.astype(np.float32), columns=LAB)
    sub.insert(0, "StudyInstanceUID", test_ids)
    sub = sub[sub_cols]
    assert list(sub.columns) == sub_cols, "column order drift"
    assert sub["StudyInstanceUID"].tolist() == test_ids, "row identity drift"
    assert np.isfinite(sub[LAB].values).all()
    out = "/kaggle/working/submission_coatnet.csv"
    sub.to_csv(out, index=False)
    print("wrote", out, "|", len(sub), "rows x", len(sub.columns), "cols", flush=True)
    print(sub.head().to_string(index=False), flush=True)
    print(f"DONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as _coat_exc:
        import traceback as _coat_traceback
        print(f"CoAtNet branch failed; retaining transformer submission: {type(_coat_exc).__name__}: {_coat_exc}", flush=True)
        _coat_traceback.print_exc()



# Hidden-rerun fail-safe final fusion.
from pathlib import Path as _D42Path
import shutil as _d42_shutil

_d42_primary = _D42Path('/kaggle/working/submission.csv')
_d42_backup = _D42Path('/kaggle/working/.d42_transformer_backup.csv')

if _d42_primary.is_file():
    _d42_shutil.copy2(_d42_primary, _d42_backup)

try:
    # Blend two independently validated rank predictors. The default remains the transformer
    # submission if the CoAtNet branch did not complete, so a recoverable branch failure
    # cannot erase a valid competition artifact.
    from pathlib import Path as _BlendPath
    import numpy as _blend_np
    import pandas as _blend_pd
    
    _blend_work = _BlendPath('/kaggle/working')
    _blend_transformer_path = _blend_work / 'submission.csv'
    _blend_coatnet_path = _blend_work / 'submission_coatnet.csv'
    if _blend_coatnet_path.is_file():
        _blend_transformer = _blend_pd.read_csv(_blend_transformer_path, dtype={'StudyInstanceUID': str})
        _blend_coatnet = _blend_pd.read_csv(_blend_coatnet_path, dtype={'StudyInstanceUID': str})
        _blend_labels = [c for c in _blend_transformer.columns if c != 'StudyInstanceUID']
        if _blend_coatnet.columns.tolist() != _blend_transformer.columns.tolist():
            raise RuntimeError('CoAtNet/transformer submission schema mismatch')
        if _blend_coatnet['StudyInstanceUID'].tolist() != _blend_transformer['StudyInstanceUID'].tolist():
            raise RuntimeError('CoAtNet/transformer study order mismatch')
        _blend_tr = _blend_transformer[
            _blend_labels
        ].rank(
            method='average',
            pct=True,
        )
    
        _blend_cr = _blend_coatnet[
            _blend_labels
        ].rank(
            method='average',
            pct=True,
        )
    
        _blend_output = (
            _blend_transformer.copy()
        )
    
        _coatnet_weight = {
            label: 0.60
            for label in _blend_labels
        }
    
    
        _base_0935 = {
            label: 0.50
            for label in _blend_labels
        }
        _base_0935.update(
            {
                'Medial Meniscus': 0.52,
                'Lateral Meniscus': 0.54,
                'Fracture': 0.54,
            }
        )
    
        # Correlation only controls risk. It never creates a new target weight.
        for _label in _blend_labels:
            _correlation = float(
                _blend_np.corrcoef(
                    _blend_tr[
                        _label
                    ].to_numpy(
                        _blend_np.float64
                    ),
                    _blend_cr[
                        _label
                    ].to_numpy(
                        _blend_np.float64
                    ),
                )[0, 1]
            )
    
            if not _blend_np.isfinite(
                _correlation
            ):
                _coatnet_weight[
                    _label
                ] = _base_0935[
                    _label
                ]
    
            elif _correlation > 0.992:
                _coatnet_weight[
                    _label
                ] = (
                    0.65
                    * _coatnet_weight[
                        _label
                    ]
                    + 0.35
                    * _base_0935[
                        _label
                    ]
                )
    
            elif _correlation < 0.60:
                _coatnet_weight[
                    _label
                ] = (
                    0.50
                    * _coatnet_weight[
                        _label
                    ]
                    + 0.50
                    * _base_0935[
                        _label
                    ]
                )
    
        for _label in _blend_labels:
            _cw = float(
                _coatnet_weight[
                    _label
                ]
            )
    
            _blend_output[
                _label
            ] = (
                (
                    1.0
                    - _cw
                )
                * _blend_tr[
                    _label
                ]
                + _cw
                * _blend_cr[
                    _label
                ]
            )
    
        _blend_output[
            _blend_labels
        ] = _blend_output[
            _blend_labels
        ].rank(
            method='average',
            pct=True,
        )
    
        print(
            '[DINOsaur V4.2] CoAtNet target weights: '
            + ', '.join(
                f'{label}='
                f'{_coatnet_weight[label]:.2f}'
                for label in _blend_labels
                if (
                    _coatnet_weight[label]
                    != 0.50
                )
            ),
            flush=True,
        )
    
        _blend_values = _blend_output[
            _blend_labels
        ].to_numpy(
            _blend_np.float64
        )
        if not _blend_np.isfinite(_blend_values).all() or _blend_values.min() < 0 or _blend_values.max() > 1:
            raise RuntimeError('invalid blended prediction values')
        _blend_output.to_csv(_blend_transformer_path, index=False)
        print(f'final submission.csv = DINOsaur V4.2 dual-checkpoint target fusion; {_blend_output.shape}', flush=True)
    else:
        print('CoAtNet output unavailable; submission.csv remains the validated transformer ensemble', flush=True)
    
    # V18 output hygiene.
    for _v18_temp in (
        _blend_work / 'submission_coatnet.csv',
        _blend_work / 'submission_transformer_0920.csv',
    ):
        try:
            if _v18_temp.is_file():
                _v18_temp.unlink()
        except OSError:
            pass

except Exception as _d42_error:
    import traceback as _d42_traceback
    print(
        '[DINOsaur V4.2] final fusion failed; '
        'restoring calibrated transformer submission: '
        f'{type(_d42_error).__name__}: {_d42_error}',
        flush=True,
    )
    _d42_traceback.print_exc()
    if _d42_backup.is_file():
        _d42_shutil.copy2(_d42_backup, _d42_primary)

finally:
    try:
        if _d42_backup.is_file():
            _d42_backup.unlink()
    except OSError:
        pass

if not _d42_primary.is_file():
    raise RuntimeError('submission.csv missing after V4.2 fail-safe')