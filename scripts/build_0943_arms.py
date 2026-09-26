# -*- coding: utf-8 -*-
"""Build every 0.943-line submission arm from the published notebook, with assertions.

WHY THIS LIVES IN scripts/ AND NOT IN .testtmp/.  The earlier version of this builder sat in
`.testtmp/`, which is git-ignored -- so the *thing that produced our submission arms* was itself
unversioned, while the arms it produced were committed.  That is backwards: the builds should be
reproducible from the record alone.

WHY FORK INSTEAD OF PORT.  Our `kernel-0942/` is the 0.942 config field-for-field (FACTS C80), so
0.943 differs from us by three things -- but one is a SPEED change (slice-header caching) that is what
pays for sampling 94 windows per arm inside one session.  Hand-porting a ~10 KB cross-cell diff risks
a silent prediction change; forking the published artifact does not.

THE FIVE ARMS (one variable each; that is what makes the results readable):

  kernel-0943   speedy   published token      the 0.943 run itself -- highest expected score
  kernel-0943f  speedy   fracture fork        our one-token _RAD_EXCLUDE change, +0.001 internally
  kernel-0943o  outer060 flat 0.60 outer      a public analysis recommends keeping a flat-0.60
                                              version as the SECOND final pick, because the per-label
                                              outer map was tuned on the public split and
                                              Lateral Meniscus = 1.00 discards 3 of that column's 4
                                              stages -- i.e. insurance against public-LB overfitting
  kernel-0943s  sparse   62/42 windows        the window count the released arms natively carry.
                                              ~5x cheaper if the sibling notebook's "~90 min on
                                              T4 x2" holds, at some cost in score: 94 windows is one
                                              of the 0.942 -> 0.943 deltas.  The quota lever.
  kernel-0943h  halfway  softened routing     keeps the 0.65 default and pulls every per-label
                                              override halfway toward it.  Our own LB evidence (C57)
                                              is that softening cost 0.001 where flattening cost
                                              0.002 -- a different axis from outer060, and the
                                              better-supported hedge.  See HANDOFF 163.3.

⚠️ Kaggle REFUSES to even push a GPU kernel while the weekly quota is exhausted (FACTS A50), so these
have to be built now and pushed the moment quota returns.  `--timeout` matters too: the notebook
carries TIME_BUDGET = 8 h with per-branch subprocess timeouts, so give the run room (e.g. -t 34200).

    python scripts/build_0943_arms.py            # build all four
    python scripts/build_0943_arms.py --only kernel-0943o
"""
import argparse
import json
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "downloads" / "research" / "fresh0924" / "maverickss26_rsna_knee_restructured_version_3"
SRC_NB = SRC / "rsna-knee-restructured-version-3.ipynb"

PRESET_OLD = 'PRESET = os.environ.get("RSNA_PRESET", "speedy")'
RAD_OLD = '_RAD_EXCLUDE = ("Baker\'s", \'Fracture\')'
RAD_FORK = '_RAD_EXCLUDE = ("Baker\'s",)'

# dir, id, title (MUST slugify to the id), preset default, fracture fork, role
VARIANTS = [
    ("kernel-0943", "ppxl16/rsna-knee-0943", "RSNA knee 0943", "speedy", False,
     "the 0.943 run itself"),
    ("kernel-0943f", "ppxl16/rsna-knee-0943f", "RSNA knee 0943f", "speedy", True,
     "0.943 + one-token fracture fork"),
    ("kernel-0943o", "ppxl16/rsna-knee-0943o", "RSNA knee 0943o", "outer060", False,
     "flat-0.60 outer -- public analysis's recommended robust second pick"),
    ("kernel-0943s", "ppxl16/rsna-knee-0943s", "RSNA knee 0943s", "sparse", False,
     "62/42 windows -- the quota lever (cheaper, likely lower)"),
    # Added 2026-09-25.  `outer060` reverts the *default* (0.65 -> 0.60) while keeping every
    # per-label override; `halfway` keeps the default and softens the overrides toward it.  They are
    # not the same axis, but our own LB evidence (C57, on the same object -- the outer CoAtNet
    # routing -- in the older v2src line) is that flattening it cost 0.002 while taking it halfway
    # cost 0.001.  So this is the better-supported hedge, and building it is free: pushing costs no
    # quota, only *running* does.  See HANDOFF 163.3 -- the choice stays open until the first read.
    ("kernel-0943h", "ppxl16/rsna-knee-0943h", "RSNA knee 0943h", "halfway", False,
     "per-label routing softened toward its default (C57 says softening beats flattening)"),
]


def slugify(t):
    import re
    return re.sub(r"[^A-Za-z0-9]+", "-", t.strip().lower()).strip("-")


def build(dst_name, kid, title, preset, fracture, role):
    dst = ROOT / dst_name
    dst.mkdir(exist_ok=True)
    nb = json.loads(SRC_NB.read_text(encoding="utf-8", errors="replace"))

    n_preset = sum("".join(c.get("source", [])).count(PRESET_OLD)
                   for c in nb["cells"] if "source" in c)
    if n_preset != 1:
        raise SystemExit("!! preset line matched %d times (need exactly 1)" % n_preset)

    want_line = PRESET_OLD.replace('"speedy"', '"%s"' % preset)
    n_rad = 0
    for c in nb["cells"]:
        txt = "".join(c.get("source", []))
        if PRESET_OLD in txt:
            txt = txt.replace(PRESET_OLD, want_line)
        n_rad += txt.count(RAD_OLD)
        if fracture and RAD_OLD in txt:
            txt = txt.replace(RAD_OLD, RAD_FORK)
        if txt != "".join(c.get("source", [])):
            c["source"] = [txt]
    if n_rad != 1:
        raise SystemExit("!! _RAD_EXCLUDE matched %d times (need exactly 1)" % n_rad)

    t = "\n".join("".join(c.get("source", [])) for c in nb["cells"])

    # ---- assertions on the artefact we are about to push ------------------------------------
    assert want_line in t, "preset default is not %r" % preset
    # `speedy` IS the default, so it has no `elif` arm -- it is the fall-through.  Non-default presets
    # must have an explicit branch; in every case the dispatch block must still be intact.
    if preset == "speedy":
        assert 'PRESET == "parent"' in t, "preset dispatch block is missing"
    else:
        assert 'elif PRESET == "%s"' % preset in t, "preset branch %r missing" % preset
    assert '"coatnet_w": {"__default__": 0.65' in t, "the 0.943 outer default is missing"
    assert '"raptor_view_w"' in t, "raptor_view_w missing"
    if fracture:
        assert RAD_FORK in t and RAD_OLD not in t, "fracture edit did not take"
    else:
        assert RAD_OLD in t, "non-fracture arm must keep the published token"
    if slugify(title) != kid.split("/")[-1]:
        raise SystemExit("!! title %r does not slugify to %r" % (title, kid.split("/")[-1]))

    (dst / "rsna-knee-restructured-version-3.ipynb").write_text(
        json.dumps(nb, ensure_ascii=False), encoding="utf-8")

    m = json.loads((SRC / "kernel-metadata.json").read_text(encoding="utf-8"))
    m["id"], m["title"], m["is_private"] = kid, title, True
    m.pop("id_no", None)
    # ⚠️ FORCED OFF, and it must stay off.  The upstream fork we inherit from ships
    # enable_internet=true (its author was developing with it on) and we copied it verbatim,
    # which is invisible until the very last step: the kernel PUSHES fine, RUNS fine, writes
    # submission.csv -- and then the submission is refused with
    #   400 "Your Notebook cannot use internet access in this competition.
    #        Please disable internet in the Notebook editor and save a new version."
    # (read from the server body, 2026-09-26).  Everything else in this lineage has it off,
    # and the notebook makes no network calls at all -- the opencv wheel it needs comes from
    # a mounted dataset, not from PyPI.  See FACTS A78.
    m["enable_internet"] = False
    (dst / "kernel-metadata.json").write_text(
        json.dumps(m, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("== %-14s %-32s preset=%-9s fracture=%-5s" % (dst_name, kid, preset, fracture))
    print("   %s" % role)
    print("   %d cells | %d datasets | kernels %s | machine %s" % (
        len(nb["cells"]), len(m.get("dataset_sources") or []), m.get("kernel_sources"),
        m.get("machine_shape")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="build just this directory name")
    a = ap.parse_args()
    if not SRC_NB.exists():
        raise SystemExit("source notebook missing: %s" % SRC_NB)
    for v in VARIANTS:
        if a.only and v[0] != a.only:
            continue
        build(*v)
    print()
    print("push when the weekly GPU quota returns (a GPU kernel cannot even be pushed while it is")
    print("exhausted, FACTS A50).  The notebook budgets 8 h internally -- give the run room:")
    for d, kid, _, p, _, _ in VARIANTS:
        if a.only and d != a.only:
            continue
        print("  kaggle kernels push -p %s -t 34200   # preset=%s" % (d, p))


if __name__ == "__main__":
    main()
