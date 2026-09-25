"""Pre-flight: is each queued experiment's PUSHED notebook configured as intended,
and did its run actually load all 20 members?

WHY THIS EXISTS
    HANDOFF 48.2 records that our v2src fork does NOT carry mattiaangeli's hard gates
    (20-member fingerprint gate, duplicate-id check, mandatory fingerprint).  We rely on
    reading the run's own log instead.  HANDOFF 16.1 states the failure mode precisely:
    a run that silently drops members scores lower, and we would then attribute the drop
    to "this axis is flat" - i.e. a broken run would be read as a scientific result and
    the axis would be retired forever.

    Two things therefore have to be checked BEFORE the submission is spent:
      1. the artifact that was actually pushed carries the intended edit;
      2. the run that produced the landed submission.csv loaded all 20 members.

WHY (1) NEEDS ITS OWN CHECK
    `ens_experiment.py --dry-run` re-applies the edits to the CURRENT source notebook.
    That proves the edits still match today's source - it does NOT prove anything about
    the notebook that was pushed hours ago.  kernel-exp/<exp>/ holds that exact pushed
    artifact, so this checks the artifact rather than a re-derivation of it.

WHY (2) CANNOT BE ANSWERED BY THE CONFIG PRINT
    cell[3] gained `print("[config] fusion weights:", ...)` in HANDOFF 33.1, but every
    run queued for 2026-09-13 was pushed BEFORE that patch, so none of their logs carry
    the full weight dump - only `preset` / `private_alpha` and the outer routing map.
    The fingerprint/banked/FALLBACK counts are still there, and those are the ones that
    detect a dropped member.

Usage:
    python scripts/preflight_runs.py                      # the queued submissions
    python scripts/preflight_runs.py --all                # every built experiment
    python scripts/preflight_runs.py --experiment NAME
"""
import argparse
import ast
import hashlib
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ens_experiment  # noqa: E402

EXP_DIR = ROOT / "kernel-exp"
LANDING = ROOT / "outputs" / "_landing"


def registry():
    """The experiment table, found by shape rather than by guessing its variable name."""
    for name in dir(ens_experiment):
        obj = getattr(ens_experiment, name)
        if isinstance(obj, dict) and "v2src-private25" in obj:
            return obj
    raise SystemExit("could not find the experiment table in ens_experiment.py")


def notebook_text(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(c.get("source", [])) for c in nb["cells"]), nb


def check_artifact(exp, spec):
    """Does the PUSHED notebook carry the intended edit, and does it still compile?"""
    d = EXP_DIR / exp
    if not d.is_dir():
        return None, "not built locally"
    nbs = sorted(d.glob("*.ipynb"))
    if not nbs:
        return None, "no notebook in %s" % d
    src, nb = notebook_text(nbs[0])

    problems = []
    for old, new in spec.get("edits", []):
        n_new, n_old = src.count(new), src.count(old)
        if n_new == 0:
            problems.append("edit NOT applied: %r missing" % new[:48])
        elif n_old:
            problems.append("original still present %dx: %r" % (n_old, old[:48]))
    if spec.get("special"):
        problems.append("uses special %r - verify by hand" % spec["special"])

    bad = []
    for c in nb["cells"]:
        if c.get("cell_type") != "code":
            continue
        try:
            ast.parse("".join(c.get("source", [])))
        except SyntaxError as e:
            bad.append(str(e))
    if bad:
        problems.append("%d cell(s) do not compile" % len(bad))

    sha = hashlib.sha256(nbs[0].read_bytes()).hexdigest()[:12]
    return (sha, problems), None


def landing_for(exp):
    """The newest landing dir for this experiment.

    Landing dirs are named `<kernel-slug>_v<version>_<stamp>`, and the kernel slug is
    `rsna-knee-ens-<exp>` for these experiments - the experiment name is a SUFFIX, not a
    prefix, which is what an earlier version of this function got wrong (it reported
    "no landing" for runs that had landed).
    """
    cands = []
    for pre in ("rsna-knee-ens-%s_v" % exp, "%s_v" % exp, "rsna-knee-%s_v" % exp):
        cands += [d for d in LANDING.iterdir() if d.is_dir() and d.name.startswith(pre)]
    if not cands:
        return None
    return sorted(set(cands), key=lambda d: d.stat().st_mtime)[-1]


def special_evidence(exp, spec):
    """For specs built by a `special` rewrite, show the arm weights actually pushed.

    A `special` regenerates the ARMS table instead of substituting a literal, so it cannot
    be checked as old->new string pairs.  Two earlier attempts at this were wrong and are
    worth recording, because both looked like they worked:

      * searching the notebook text for "maxspan-v5" and printing the match shows only the
        arm's NAME - the weight lives in a different part of the dict, so the check printed
        nothing useful and would have "passed" a no-op rewrite;
      * comparing raw substring COUNTS between two notebooks is blind for the same reason:
        the names are identical, only the numbers differ.

    So: pull `_KE_SRC` out by AST, then read the `name`/`w` pairs inside it.
    """
    d = EXP_DIR / exp
    nbs = sorted(d.glob("*.ipynb")) if d.is_dir() else []
    if not nbs:
        return []
    nb = json.loads(nbs[0].read_text(encoding="utf-8"))
    module = None
    for c in nb["cells"]:
        src = "".join(c.get("source", []))
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Assign) and any(
                    getattr(t, "id", None) == "_KE_SRC" for t in n.targets):
                module = ast.literal_eval(n.value)
    if module is None:
        return []
    out = []
    for m in re.finditer(r"[\{\[]([^{}\[\]]{0,240}?)[\}\]]", module, re.S):
        blk = m.group(1)
        w = re.search(r"['\"]w['\"]\s*:\s*([0-9.]+)", blk)
        n = re.search(r"['\"]name['\"]\s*:\s*['\"]([^'\"]+)", blk)
        if w and n:
            out.append("%s w=%s" % (n.group(1), w.group(1)))
    return sorted(set(out))


def check_run(exp):
    """What the run's own log says about members loaded and config used."""
    d = landing_for(exp)
    if d is None:
        return None, "no landing"
    logs = sorted(d.glob("*.log"))
    if not logs:
        return None, "no log in %s" % d.name
    try:
        recs = json.loads(logs[0].read_text(encoding="utf-8", errors="replace"))
    except ValueError:
        return None, "log is not parseable JSON"
    text = "\n".join(r.get("data", "") for r in recs)

    fp = len(re.findall(r"fingerprint matches within", text))
    banked = len(re.findall(r"\bbanked\b", text))
    fallback = len(re.findall(r"FALLBACK", text))
    members = re.findall(r"weights package:\s*(\d+)\s*member", text)
    config = [l.strip() for l in text.splitlines() if "[config]" in l]

    problems = []
    if fp != 20:
        problems.append("fingerprint lines = %d, expected 20" % fp)
    if banked != 20:
        problems.append("banked lines = %d, expected 20" % banked)
    if fallback:
        problems.append("FALLBACK appears %d time(s) - a study fell through" % fallback)
    if members and members[0] != "20":
        problems.append("weights package loaded %s members" % members[0])
    return {"dir": d.name, "fp": fp, "banked": banked, "fallback": fallback,
            "config": config, "problems": problems}, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--experiment")
    args = ap.parse_args()

    table = registry()
    if args.experiment:
        names = [args.experiment]
    elif args.all:
        names = sorted(d.name for d in EXP_DIR.iterdir() if d.is_dir())
    else:
        names = ["arm-weights-alt", "v2src-private25", "v2src-private55",
                 "v2src-a5w30", "v2src-a5w60"]

    failures = 0
    for exp in names:
        spec = table.get(exp)
        print("=" * 78)
        print("%s%s" % (exp, "" if spec else "   (not in the registry)"))
        if spec is None:
            print("   SKIP - no registry entry, cannot know the intended edit")
            continue

        art, err = check_artifact(exp, spec)
        if err:
            print("   artifact : %s" % err)
        else:
            sha, problems = art
            # A `special` rewrite is reported, not silently counted as a failure: it
            # regenerates a table instead of substituting a literal, so the check below
            # cannot see it and printing the evidence is the honest alternative.
            hard = [p for p in problems if "verify by hand" not in p]
            print("   artifact : kernel-exp/%s sha256=%s  %s"
                  % (exp, sha, "OK" if not hard else "PROBLEM"))
            for p in hard:
                print("              ! %s" % p)
            failures += len(hard)
            if spec.get("special"):
                print("              special=%r - values found in the pushed notebook:"
                      % spec["special"])
                for line in special_evidence(exp, spec):
                    print("                 %s" % line[:140])

        run, err = check_run(exp)
        if err:
            print("   run      : %s" % err)
        else:
            print("   run      : %s" % run["dir"])
            print("              fingerprint=%d  banked=%d  FALLBACK=%d  %s"
                  % (run["fp"], run["banked"], run["fallback"],
                     "OK" if not run["problems"] else "PROBLEM"))
            for c in run["config"]:
                print("              %s" % c[:150])
            for p in run["problems"]:
                print("              ! %s" % p)
            failures += len(run["problems"])

    print("=" * 78)
    print("problems found: %d" % failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
