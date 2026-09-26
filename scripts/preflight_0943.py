# -*- coding: utf-8 -*-
"""Pre-flight for the two 0.943 submission arms, runnable WITHOUT spending any GPU quota.

WHY THIS EXISTS.  GPU quota is the scarcest resource in this project: 30 h/week, and each 0.943 arm
needs a ~9 hour session, so a week fits at most three arms.  Discovering a broken dataset reference or
a syntax error AFTER the quota is spent costs a whole arm.  Everything checked here is knowable in
advance, for free.

What it checks, per kernel:
  1. metadata: id/title slug agreement (Kaggle refuses a title that does not slugify to the id),
     enable_gpu, code_file present
  2. every dataset_sources entry actually resolves
  3. every kernel_sources entry resolves AND is COMPLETE -- a kernel source that never finished has no
     output to mount, and the run would fail at import time
  4. competition_sources resolve
  5. the notebook parses, and EVERY code cell compiles (a SyntaxError found here is free; found after
     the quota is spent it is not)
  6. the config values the fork was built around are still present, so a later edit cannot silently
     revert the experiment: the Raptor arm weights (0.60/0.10 -- the side of the 0.941 plateau that
     Mattia's own stack used), the 0.65 outer CoAtNet default, and the per-arm _RAD_EXCLUDE
  7. any runtime evidence the notebook itself states, because "does it fit in one session" is the one
     risk that cannot be fixed after the fact

    python scripts/preflight_0943.py
"""
import argparse
import ast
import json
import pathlib
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = pathlib.Path(__file__).resolve().parent.parent

EXPECT = {
    # 鈿狅笍 the fork's tuple is ("Baker's",) WITH a trailing comma -- that is a 1-element TUPLE.  Written
    # as ("Baker's") it would be the plain string, and `label not in _RAD_EXCLUDE` would silently turn
    # into SUBSTRING matching instead of membership.  Assert the comma.
    "kernel-0943": {"rad_exclude": "\"Baker's\", 'Fracture'", "preset": "speedy",
                    "note": "0.943 control"},
    "kernel-0943f": {"rad_exclude": "\"Baker's\",", "preset": "speedy",
                     "note": "0.943 + the one-token fracture fork"},
    "kernel-0943o": {"rad_exclude": "\"Baker's\", 'Fracture'", "preset": "outer060",
                     "note": "flat-0.60 outer (robust second pick)"},
    "kernel-0943s": {"rad_exclude": "\"Baker's\", 'Fracture'", "preset": "sparse",
                     "note": "62/42 windows (the quota lever)"},
    "kernel-0943h": {"rad_exclude": "\"Baker's\", 'Fracture'", "preset": "halfway",
                     "note": "routing softened toward its default (C57-preferred hedge)"},
}
# The arm weights the fork is built on.  A public analysis of the 0.941 plateau argues that forks
# stalling at 0.941 are the ones that moved these to 0.55/0.15, while the frontier's own submissions
# kept 0.60/0.10.  Assert we are on the 0.60/0.10 side.
ARM_W = {"maxspan-v5": "0.60", "maxspan-v5-reverse": "0.10",
         "native384dense-v10": "0.10", "native384-v8": "0.20"}


def slugify(title):
    s = re.sub(r"[^A-Za-z0-9]+", "-", title.strip().lower()).strip("-")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*", default=sorted(EXPECT))
    a = ap.parse_args()

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    problems, warnings = [], []

    for d in a.dirs:
        # kernel directories live under kernels/ after the 2026-09-26 reorganisation; the flat
        # location is still accepted so that an older checkout keeps working.
        kdir = ROOT / "kernels" / d
        if not kdir.is_dir():
            kdir = ROOT / d
        print("=" * 72)
        print("### %s   (%s)" % (d, EXPECT.get(d, {}).get("note", "")))
        if not kdir.exists():
            problems.append("%s: directory missing" % d)
            print("  !! directory missing")
            continue

        meta_p = kdir / "kernel-metadata.json"
        meta = json.loads(meta_p.read_text(encoding="utf-8"))

        # ---- 1. metadata -------------------------------------------------
        want_slug = meta["id"].split("/")[-1]
        got_slug = slugify(meta["title"])
        print("  id            : %s" % meta["id"])
        print("  title->slug   : %r -> %r  %s"
              % (meta["title"], got_slug, "OK" if got_slug == want_slug else "MISMATCH"))
        if got_slug != want_slug:
            problems.append("%s: title does not slugify to the id (%s != %s)"
                            % (d, got_slug, want_slug))
        print("  enable_gpu    : %s" % meta.get("enable_gpu"))
        if not meta.get("enable_gpu"):
            problems.append("%s: enable_gpu is not true" % d)
        # A code competition REFUSES a submission whose notebook has internet enabled, and it
        # refuses it only at the last step -- after the push succeeded and the run completed:
        #   400 "Your Notebook cannot use internet access in this competition."
        # The check belongs here, where it costs nothing, not three hours later.  FACTS A78.
        net = meta.get("enable_internet")
        print("  internet off  : %s" % ("OK" if net is False else "WRONG (%r)" % (net,)))
        if net is not False:
            problems.append(
                "%s: enable_internet is %r -- a code-competition submission will be refused "
                "with 400 FAILED_PRECONDITION" % (d, net))
        cf = kdir / meta["code_file"]
        print("  code_file     : %s (%s)" % (meta["code_file"],
                                             "present" if cf.exists() else "MISSING"))
        if not cf.exists():
            problems.append("%s: code_file missing" % d)
            continue

        # ---- 2. dataset sources -----------------------------------------
        for ref in meta.get("dataset_sources", []):
            try:
                fl = api.dataset_list_files(ref, page_size=1).files or []
                print("  dataset OK    : %s%s" % (ref, "" if fl else "  (0 files!)"))
                if not fl:
                    warnings.append("%s: dataset %s lists 0 files" % (d, ref))
            except Exception as e:                                  # noqa: BLE001
                print("  dataset FAIL  : %s -> %s" % (ref, str(e)[:70]))
                problems.append("%s: dataset %s unresolvable" % (d, ref))

        # ---- 3. kernel sources (must be COMPLETE to have an output) -----
        for ref in meta.get("kernel_sources", []):
            st = ""
            try:
                s = api.kernels_status(ref)
                st = str(getattr(s, "status", s))
                ok = "COMPLETE" in st.upper()
                print("  kernel OK     : %-46s %s" % (ref, st))
                if not ok:
                    problems.append("%s: kernel source %s is %s, not COMPLETE" % (d, ref, st))
            except Exception as e:                                  # noqa: BLE001
                print("  kernel FAIL   : %s -> %s" % (ref, str(e)[:70]))
                problems.append("%s: kernel source %s unresolvable" % (d, ref))

        # ---- 4. competition sources -------------------------------------
        for ref in meta.get("competition_sources", []):
            try:
                api.competitions_list(search=ref)
                print("  competition OK: %s" % ref)
            except Exception as e:                                  # noqa: BLE001
                print("  competition ?: %s -> %s" % (ref, str(e)[:60]))
                warnings.append("%s: competition %s not confirmed" % (d, ref))
        

        # ---- 4b. MODEL sources -------------------------------------------
        # 鈿狅笍 NOT optional.  A public score ladder measured on real submissions shows that a run with
        # model_sources missing silently DROPS all 20 DINOv2 members, still completes, and scores
        # 0.937 instead of 0.940.  The pipeline degrades quietly by design, so the sources have to be
        # checked here rather than trusted.  Entry form is owner/model/framework/instance[/version].
        for ref in meta.get("model_sources", []):
            parts = ref.split("/")
            ok = False
            try:
                if len(parts) >= 4:
                    api.model_instance_get("/".join(parts[:4]))
                else:
                    api.model_get("/".join(parts[:2]))
                ok = True
            except Exception as e:                                  # noqa: BLE001
                print("  model FAIL    : %s -> %s" % (ref, str(e)[:70]))
                problems.append("%s: model source %s unresolvable (a missing model source "
                                "silently costs ~0.003)" % (d, ref))
            if ok:
                print("  model OK      : %s" % ref)

        # ---- 5. notebook parses; every code cell compiles ---------------
        nb = json.loads(cf.read_text(encoding="utf-8"))
        src = "\n".join("".join(c.get("source", []))
                        for c in nb.get("cells", []) if c.get("cell_type") == "code")
        bad = 0
        for i, c in enumerate([c for c in nb["cells"] if c["cell_type"] == "code"]):
            code = "".join(c.get("source", []))
            try:
                compile(code, "<cell%d>" % i, "exec")
            except SyntaxError as e:
                bad += 1
                problems.append("%s: code cell %d SyntaxError: %s" % (d, i, e))
        print("  cells         : %d total, %d code, %d SyntaxError"
              % (len(nb["cells"]), sum(1 for c in nb["cells"] if c["cell_type"] == "code"), bad))

        # ---- 6. the config the fork rests on ----------------------------
        m = re.search(r'"raptor_view_w"\s*:\s*\{(.*?)\}', src, re.S)
        if not m:
            problems.append("%s: raptor_view_w not found" % d)
            print("  arm weights   : NOT FOUND")
        else:
            body = " ".join(m.group(1).split())
            print("  arm weights   : %s" % body[:110])
            for arm, want in ARM_W.items():
                mm = re.search(r'"%s"\s*:\s*([0-9.]+)' % re.escape(arm), body)
                got = mm.group(1) if mm else None
                if got != want:
                    problems.append("%s: arm %s is %s, expected %s" % (d, arm, got, want))
        for probe, label in (('"__default__": 0.65', "outer CoAtNet default 0.65"),
                             ('outer060', "outer060 switch")):
            hit = probe in src
            print("  %-28s %s" % (label + ":", "present" if hit else "ABSENT"))
            if not hit:
                problems.append("%s: %s missing" % (d, label))
        want_re = EXPECT.get(d, {}).get("rad_exclude")
        if want_re:
            hit = ("_RAD_EXCLUDE = (%s)" % want_re) in src
            print("  _RAD_EXCLUDE %-15s %s" % ("(" + want_re + "):", "OK" if hit else "MISMATCH"))
            if not hit:
                problems.append("%s: _RAD_EXCLUDE is not (%s)" % (d, want_re))
        if "speedy" in src:
            print("  speedy preset : referenced")
        want_preset = EXPECT.get(d, {}).get("preset")
        if want_preset:
            line = 'PRESET = os.environ.get("RSNA_PRESET", "%s")' % want_preset
            hit = line in src
            print("  preset default: %-9s %s" % (want_preset, "OK" if hit else "MISMATCH"))
            if not hit:
                problems.append("%s: default preset is not %r" % (d, want_preset))

        # ---- 7. runtime evidence (the one unfixable risk) ---------------
        rt = set()
        for pat in (r'~?\s*(\d+(?:\.\d+)?)\s*(?:hours|hour|hrs|h)\b',
                    r'(\d+)\s*windows', r'k_eval["\']?\s*[:=]\s*(\d+)'):
            for mm in re.finditer(pat, src, re.I):
                rt.add(mm.group(0).strip())
        print("  runtime hints : %s" % (", ".join(sorted(rt)[:8]) or "none stated"))

    print("=" * 72)
    print("PROBLEMS: %d" % len(problems))
    for p in problems:
        print("   !! %s" % p)
    print("WARNINGS: %d" % len(warnings))
    for w in warnings:
        print("   ~  %s" % w)
    print("PREFLIGHT %s" % ("OK" if not problems else "FAILED"))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
