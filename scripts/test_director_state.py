"""Prove (or refute) that the director's per-action state writes clobber the ledger.

HYPOTHESIS UNDER TEST
    do_download() / do_submit() / do_score() all end with
        save_json(STATE_PATH, st)
    where `st` is the PER-WATCH sub-dict, while poll_once() keeps the ledger as
        {"watches": {ref: st}}
    and save_json() is a plain dump.  If so, every action rewrites the whole
    state file as a bare sub-dict, and only the pass-end
        save_json(STATE_PATH, state)
    puts the structure back.

WHY IT WOULD MATTER
    An exception between an action and the pass-end save (the obvious candidate
    being our_rank() raising on a leaderboard download, which is NOT inside a
    try/except) would leave the file holding a single watch.  The next pass
    calls watches.setdefault(ref, {...}) and silently rebuilds every other watch
    with an EMPTY `handled` list - so those watches re-run every action,
    including SUBMITTING AGAIN and burning the 5-per-UTC-day quota.

GUARDING AGAINST A VACUOUS PASS
    The first version of this test reported "OK" for download and score, but both
    had returned early without ever reaching their save_json (download's key was
    already in `handled`; score could not match the submission because
    `submitted_message` was unset).  A test whose cases silently skip the code
    under test is worse than no test, so every case here now asserts that the
    action actually MUTATED the ledger, and a no-op is reported as NO-OP/FAIL.

HOW IT RUNS
    Entirely inside .testtmp/, with log(), inbox(), submit(), download_output(),
    my_submissions() and our_rank() stubbed.  It cannot touch the real ledger,
    the real inbox, or Kaggle.  That stubbing is the point: HANDOFF 46.3 records
    a previous test that replaced some functions but not log()/inbox()/save_json(),
    and consequently wrote a fake "rank 999" line into the production log.

Usage:  python scripts/test_director_state.py
"""
import csv
import json
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import director  # noqa: E402

N_WATCHES = 3
SANDBOX_ROOT = pathlib.Path(__file__).resolve().parent.parent / ".testtmp"
TARGET_REF = "owner/k0"


def quiet(tmp):
    """Point every side effect at the sandbox, then silence log()/inbox()."""
    director.STATE_PATH = tmp / "state.json"
    director.CONFIG_PATH = tmp / "cfg.json"
    director.LANDING = tmp / "landing"
    director.log = lambda msg: None
    director.inbox = lambda msg: None


def seed(tmp, handled, with_message=False):
    """A realistic multi-watch ledger, written with the real save_json."""
    state = {"watches": {}}
    for i in range(N_WATCHES):
        land = tmp / ("land%d" % i)
        land.mkdir(parents=True, exist_ok=True)
        with open(land / "submission.csv", "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["StudyInstanceUID"] + ["L%d" % j for j in range(12)])
            for k in range(3):
                w.writerow(["s%d" % k] + ["0.5"] * 12)
        st = {"handled": list(handled), "version": 1, "landing_dir": str(land),
              "status": "COMPLETE"}
        if with_message:
            # do_score only proceeds when the submission description matches, so a
            # case that leaves this unset is a no-op, not a pass.
            st["submitted_message"] = "k%d v1" % i
        state["watches"]["owner/k%d" % i] = st
    director.save_json(director.STATE_PATH, state)
    return state


def read_state():
    with open(director.STATE_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def describe(obj):
    if not isinstance(obj, dict):
        return "not-a-dict"
    if "watches" in obj:
        return "ledger(%d watches)" % len(obj["watches"])
    return "BARE SUB-DICT (keys: %s)" % ",".join(sorted(obj)[:6])


def check(label, tmp, call, handled, with_message=False):
    """Run one action; report the file's shape AND whether the case did anything."""
    seed(tmp, handled, with_message)
    # ⚠️ Snapshot by VALUE, and take the live object from a SEPARATE read.  An earlier
    # version did `before = read_state(); st = before["watches"][REF]`, so the action
    # mutated `before` in place through `st` and the "did anything change" test was
    # always False - a check that could not fail.
    before = json.loads(json.dumps(read_state()))
    st = read_state()["watches"][TARGET_REF]
    entry = {"label": "k0", "actions": ["download", "submit", "score"]}
    call(st, entry)
    after = read_state()

    shape = describe(after)
    structure_ok = shape.startswith("ledger")
    mutated = json.dumps(after, sort_keys=True) != json.dumps(before, sort_keys=True)

    if not mutated:
        verdict, note = "NO-OP", "case never reached the code under test"
    elif structure_ok:
        verdict, note = "OK", "ledger survived"
    else:
        verdict, note = "CLOBBER", "file left holding a single watch"

    print("  %-8s %-46s %-34s %s" % (verdict, label, shape, note))
    return (mutated and structure_ok), (not mutated)


def real_ledger_case(tmp):
    """Run save_watch against a COPY of the live ledger, whatever shape it has now.

    The synthetic case uses 3 watches; the real one has 16 with long `handled` lists.
    A merge bug that only shows up at the real size or with real refs would otherwise
    reach production on the next scoring event.
    """
    live = pathlib.Path(__file__).resolve().parent.parent / "state" / "director_state.json"
    if not live.is_file():
        print("  SKIP     live ledger not found at %s" % live)
        return True
    copy = tmp / "real_ledger.json"
    shutil.copyfile(live, copy)
    director.STATE_PATH = copy

    before = read_state()
    refs = sorted(before["watches"])
    n = len(refs)
    victim = refs[0]
    other = refs[-1]
    other_handled = list(before["watches"][other].get("handled", []))

    probe = dict(before["watches"][victim])
    probe["handled"] = list(probe.get("handled", [])) + ["test:probe"]
    director.save_watch(victim, probe)
    after = read_state()

    ok = ("watches" in after
          and len(after["watches"]) == n
          and "test:probe" in after["watches"][victim]["handled"]
          and after["watches"][other].get("handled", []) == other_handled)
    print("  %-8s %-46s %-34s %s"
          % ("OK" if ok else "BROKEN", "live ledger shape (%d watches)" % n,
             describe(after),
             "all watches preserved" if ok else "MERGE LOST DATA"))
    if not ok:
        print("           victim=%s  expected %d watches, got %d"
              % (victim, n, len(after.get("watches", {}))))
    return ok


def main():
    tmp = SANDBOX_ROOT / "director_state_case"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    quiet(tmp)

    print("sandbox: %s" % tmp)
    print("real ledger untouched; log()/inbox()/submit() stubbed\n")
    print("seeded ledger -> %s\n" % describe(seed(tmp, ["download:v1"])))

    director.download_output = lambda ref, dest, need=None: (
        True, dest, [dest / "submission.csv"])
    director.my_submissions = lambda: [
        {"description": "%s v1" % TARGET_REF.split("/")[-1],
         "publicScore": "0.941", "status": "SubmissionStatus.COMPLETE"}]
    director.our_rank = lambda: 152

    # download: `handled` is EMPTY so the key is not already done -> reaches the save.
    director.submit = lambda ref, ver, path, msg: (True, "ref=999 (stub)")
    CASES = [
        ("download", [],
         lambda st, e: director.do_download(TARGET_REF, 1, e, st)),
        ("submit succeeds", ["download:v1"],
         lambda st, e: director.do_submit({}, TARGET_REF, 1, e, st)),
    ]
    results, vacuous = [], []
    for label, handled, fn in CASES:
        ok, skipped = check(label, tmp, fn, handled)
        results.append(ok)
        vacuous.append(skipped)

    # submit, the 400/quota branch - the one that fires on every pass today.
    director.submit = lambda ref, ver, path, msg: (False, "400 Client Error (stub)")
    ok, skipped = check("submit blocked by the daily cap", tmp,
                        lambda st, e: director.do_submit({}, TARGET_REF, 1, e, st),
                        ["download:v1"])
    results.append(ok)
    vacuous.append(skipped)

    # score: needs submitted_message set, else it returns before saving.
    ok, skipped = check("score lands", tmp,
                        lambda st, e: director.do_score({}, TARGET_REF, 1, e, st),
                        ["download:v1", "submit:v1"], with_message=True)
    results.append(ok)
    vacuous.append(skipped)

    results.append(real_ledger_case(tmp))

    print()
    if any(vacuous):
        print("VERDICT: INCONCLUSIVE - %d case(s) silently did nothing, so they prove "
              "nothing either way." % sum(vacuous))
        return 1
    if all(results):
        print("VERDICT: no clobbering observed - hypothesis REFUTED.")
        return 0
    print("VERDICT: CONFIRMED - %d/%d exercised actions leave the state file holding a "
          "single watch." % (sum(1 for r in results if not r), len(results)))
    print("         An exception before the pass-end save therefore destroys the other")
    print("         watches' `handled` lists; the next pass rebuilds them EMPTY, so every")
    print("         action re-runs - including re-submitting and burning quota.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
