"""Take the Models-axis shot the cheap way: measure a candidate member's OWN published OOF first.

WHY NOT WIRE IT STRAIGHT IN. Three candidates were found; two cannot be used responsibly:
  * `ruicompany/rsna-knee-a5-rad`  -> its files are `a5/m_f0..4.pt`, `rad/ResNet50.pt` and
    `rad/v52_radimagenet_heads.pt`, i.e. **repackages of three datasets we already mount**. Zero new
    information.
  * `yaolinghan/rsna-knee-pool-models` -> a genuinely different architecture (ConvNeXtV2) but its
    model card is ONE SENTENCE with **no geometry at all**: no slot layout, span, crop, img, k_eval
    or arch string. Wiring it means guessing the author's preprocessing -- the silent-failure class
    the meniscus arm flagged.
  * `yunusgmsoy/rsna-knee-radhead-resnet50` -> geometry IS documented (3 plane slots, 8 slices --
    our `_RAD_E11/E13` geometry) and it ships a full 4407-row OOF matrix.

This downloads that OOF. It costs no GPU and it answers the question the model card answers badly:
the card claims 0.8251 on the **58-gold** set, whose bootstrap CI is +/-0.021 -- far too wide to gate on.
"""
import pathlib

from kaggle.api.kaggle_api_extended import KaggleApi

REF = "yunusgmsoy/rsna-knee-radhead-resnet50/PyTorch/5-fold-yunus-v5/1"
OUT = pathlib.Path("work/models_axis")
OUT.mkdir(parents=True, exist_ok=True)

api = KaggleApi()
api.authenticate()
print("downloading", REF)
path = api.model_instance_version_download(REF, path=str(OUT), force=True, quiet=True)
print("->", path)
for p in sorted(OUT.rglob("*")):
    if p.is_file():
        print("   %10.1f KB  %s" % (p.stat().st_size / 1e3, p.relative_to(OUT)))
