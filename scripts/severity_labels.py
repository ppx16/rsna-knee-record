# -*- coding: utf-8 -*-
"""Severity-aware report -> 12 soft labels, for the RSNA knee competition.

WHY THIS EXISTS (HANDOFF 72)
    The host's labelling rubric is SEVERITY-thresholded, not presence-based:

        ACL      high-grade partial / full-thickness tear; "mild signal change,
                 degeneration, or thickening without discontinuity is NEGATIVE"
        MCL      high-grade partial / complete ACUTE tear; "low-grade sprains and
                 chronic or remote stress changes are NEGATIVE"
        Meniscus abnormal signal definitely contacting the surface on >=2 images;
                 "intrasubstance degeneration that does not reach the surface is NEGATIVE"
        OA       >= ~1 cm of HIGH-GRADE cartilage loss (>50 % thickness)
        Effusion / Baker's   MODERATE OR LARGE only
        Contusion  marrow oedema WITHOUT a discrete fracture line
        Fracture   an ACUTE cortical break / fracture line

        "ambiguous or borderline findings ('on the fence') were graded as NEGATIVE
         to favour specificity."

    Every public label source - the rule extractor, the LLM tables, and the labels the
    0.941 model was trained on - asks "is the finding MENTIONED?".  The ground truth asks
    "is it SEVERE and unambiguous?".  So all of them mislabel mild/borderline findings as
    positive, in the same direction, which is the most economical explanation for why the
    whole public ecosystem plateaus at 0.941.

WHAT THIS DOES
    Detects a mention per label, then looks at the SEVERITY CONTEXT around that mention
    and emits P(severe and unambiguous) in [0,1] - a soft target, which the training
    script's loss already expects.

HONEST LIMITS
    * The lexicon is hand-built from the rubric wording, in the languages named by the
      EDA (English, Turkish, Spanish, Greek, German, French, Dutch, Slavic, ...).  It is
      NOT a validated instrument.
    * gold-58 is n=58 and cannot referee effects of the size we care about (HANDOFF 13.2).
      Treat the printed AUC as a DIRECTIONAL check, never as a scoreboard.
    * The real referee is a trained model's score - that is the point of building this.

Usage:
    python scripts/severity_labels.py                 # write + report vs gold-58
    python scripts/severity_labels.py --out PATH
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import re
import unicodedata

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "raw" / "train.csv"
DEFAULT_OUT = ROOT / "outputs" / "report_labels_severity.csv"

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

# ---------------------------------------------------------------- mention lexicon
# Deliberately broad: recall of the MENTION is cheap to get, and the severity layer is
# what decides the label.  Lower-cased, accents stripped before matching.
MENTION = {
    "ACL": ["acl", "anterior cruciate", "ligamento cruzado anterior", "öne çapraz",
            "vorderes kreuzband", "ligament croisé antérieur", "voorste kruisband",
            "prednji križni", "πρόσθιος χιαστός", "lca"],
    "MCL": ["mcl", "medial collateral", "colateral medial", "iç yan bağ",
            "mediales seitenband", "ligament collatéral médial", "mediale collaterale",
            "medijalni kolateral", "έσω πλάγιος", "lci"],
    "Medial Meniscus": ["medial meniscus", "menisco medial", "menisco interno",
                        "medial menisküs", "innenmeniskus", "ménisque interne",
                        "mediale meniscus", "medijalni menisk", "έσω μηνίσκος"],
    "Lateral Meniscus": ["lateral meniscus", "menisco lateral", "menisco externo",
                         "lateral menisküs", "außenmeniskus", "ménisque externe",
                         "laterale meniscus", "lateralni menisk", "έξω μηνίσκος"],
    "Medial OA": ["medial compartment", "femorotibial medial", "femorotibial interno",
                  "mediale femorotibial", "compartimento mediale", "έσω μηροκνημιαία"],
    "Lateral OA": ["lateral compartment", "femorotibial lateral", "femorotibial externo",
                   "laterale femorotibial", "compartimento laterale", "έξω μηροκνημιαία"],
    "PF OA": ["patellofemoral", "femoropatellar", "femoropatelar", "patellofemoral",
              "μηροεπιγονατιδική"],
    "Effusion": ["effusion", "derrame", "efüzyon", "erguss", "épanchement", "effusie",
                 "izljev", "συλλογή", "joint fluid"],
    "Synovitis": ["synovitis", "sinovitis", "sinovit", "synovialitis", "synovite",
                  "sinovitis", "υνοβίτιδα"],
    "Baker's": ["baker", "popliteal cyst", "quiste de baker", "baker kisti",
                "bakerzyste", "kyste de baker", "baker-cyste", "bakerova cista"],
    "Contusion": ["contusion", "contusión", "kontüzyon", "kontusion", "contusion",
                  "bone bruise", "marrow edema", "marrow oedema", "edema óseo",
                  "kemik ödemi", "knochenmarködem", "œdème osseux", "oedema"],
    "Fracture": ["fracture", "fractura", "kırık", "fraktur", "fracture", "fratura",
                 "fraktura", "κάταγμα", "cortical break", "fraktur"],
}

# ---------------------------------------------------------------- severity lexicon
# SEVERE pushes toward the rubric's positive; MILD toward its explicit negative.
SEVERE = [
    # grade words
    "complete", "full-thickness", "full thickness", "high-grade", "high grade",
    "grade 3", "grade 4", "grade iii", "grade iv", "grad 3", "grad 4", "grado 3",
    "grado 4", "evre 3", "evre 4", "outerbridge 3", "outerbridge 4", "icrs 3", "icrs 4",
    # magnitude
    "large", "massive", "severe", "marked", "advanced", "extensive", "substantial",
    "grande", "importante", "grave", "severo", "schwer", "ausgedehnt", "groß",
    "masif", "belirgin", "ileri", "σοβαρ", "μεγάλ", "εκτεταμέν", "ernstig", "uitgebreid",
    # tear words
    "rupture", "ruptura", "r uptur", "yırtık", "riss", "complete tear", "transección",
    "rotura", "rotura completa", "total",
    # acute
    "acute", "aguda", "akut", "aigu", "aiguë", "akutni", "οξεία",
    # fluid
    "distend", "distensión", "moderate to large", "moderada a grande",
]
MILD = [
    # explicit rubric negatives
    "mild", "minimal", "slight", "trace", "small", "low-grade", "low grade",
    "grade 1", "grade 2", "grade i", "grade ii", "grad 1", "grad 2", "grado 1",
    "grado 2", "evre 1", "evre 2", "outerbridge 1", "outerbridge 2", "icrs 1", "icrs 2",
    "leve", "ligera", "mínima", "minima", "pequeñ", "hafif", "minimal", "az miktarda",
    "gering", "leicht", "diskret", "minimal", "faible", "léger", "légère", "miniem",
    "gering", "ελάχιστ", "ήπι", "μικρ",
    # the rubric's explicit NEGATIVES
    "degeneration", "degenerative", "degenerativ", "degeneratif", "dejeneratif",
    "dejenerasyon", "intrasubstance", "intra-substance", "no discontinuity",
    "without discontinuity", "sin discontinuidad", "kontinuität erhalten",
    "chronic", "crónica", "chronisch", "kronik", "χρόνι", "old", "remote", "antigua",
    "sprain", "esguince", "distorsión", "distorsiyon", "verstauchung", "entorse",
    "thickening", "engrosamiento", "kalınlaşma", "verdickung", "épaississement",
    "signal change", "cambio de señal", "sinüs değişikliği", "signalalteration",
    "oedema", "edema", "bruise", "contusión",
    # normality / absence
    "no ", "without", "sin ", "yok", "kein", "sans", "geen", "nema", "χωρίς", "unremarkable",
    "intact", "normal", "normale", "normal ", "íntegro",
]
NEGATORS = ["no ", "no evidence", "without", "sin ", "yok", "kein", "sans", "geen",
            "nema", "χωρίς", "ruled out", "descart", "ekarte", "ausgeschlossen",
            "exclu", "negativo", "negative", "negatif", "unauffällig"]

SENT_SPLIT = re.compile(r"[.;\n!?]+")
WINDOW = 90          # characters of context kept on each side of a mention


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower()


def sentences(text: str):
    for s in SENT_SPLIT.split(text):
        s = s.strip()
        if s:
            yield s


def score_report(text: str) -> dict:
    """One soft label per target: P(the rubric would call this positive)."""
    ns = norm(text)
    sents = list(sentences(ns))
    out = {}
    for lab in LAB:
        best = None
        keys = MENTION[lab]
        for i, sent in enumerate(sents):
            if not any(k in sent for k in keys):
                continue
            # context = this sentence plus neighbours (radiology reports are terse)
            ctx = " ".join(sents[max(0, i - 1): i + 2])
            # is the mention itself negated?  the sentinel sits immediately before it
            negated = False
            for k in keys:
                j = ctx.find(k)
                if j >= 0 and any(n in ctx[max(0, j - 24): j] for n in NEGATORS):
                    negated = True
                    break
            sev = sum(1 for w in SEVERE if w in ctx)
            mild = sum(1 for w in MILD if w in ctx)
            if negated:
                # rubric: ambiguous / absent -> negative to favour specificity
                val = 0.05
            elif mild and not sev:
                val = 0.20
            elif sev and not mild:
                val = 0.90
            elif sev and mild:
                val = 0.55
            else:
                # mentioned with no severity qualifier: weak positive.  The rubric would
                # call an unqualified mention "on the fence", and on-the-fence is negative,
                # so this is deliberately below 0.5.
                val = 0.35
            best = val if best is None else max(best, val)
        out[lab] = 0.05 if best is None else best
    return out


def macro_auc(y: np.ndarray, p: np.ndarray) -> float:
    """AUC per column, skipping columns with a single class, then macro-average."""
    from sklearn.metrics import roc_auc_score
    vals = []
    for j in range(y.shape[1]):
        if len(set(y[:, j].astype(int))) > 1:
            vals.append(roc_auc_score(y[:, j], p[:, j]))
    return float(np.mean(vals)) if vals else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--train", default=str(TRAIN))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.train, encoding="utf-8-sig", newline="")))
    print("reports: %d" % len(rows))

    uids, P = [], []
    for i, r in enumerate(rows):
        if i and i % 1000 == 0:
            print("   ... %d" % i, flush=True)
        uids.append(r["StudyInstanceUID"])
        s = score_report(r.get("Report") or "")
        P.append([s[l] for l in LAB])
    P = np.asarray(P, dtype=np.float32)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["StudyInstanceUID"] + LAB)
        for u, p in zip(uids, P):
            w.writerow([u] + ["%.4f" % v for v in p])
    print("wrote", out, P.shape)

    # ---- directional check on the 58 gold studies -------------------------------
    gold = [(r, i) for i, r in enumerate(rows) if (r.get(LAB[0]) or "").strip() != ""]
    print("\ngold studies with all-12 labels: %d" % len(gold))
    if len(gold) >= 10:
        y = np.array([[int(float(r[l])) for l in LAB] for r, _ in gold])
        p = np.stack([P[i] for _, i in gold])
        print("  severity extractor  gold-58 macro-AUC: %.4f" % macro_auc(y, p))
        # presence-only baseline: same mentions, no severity layer
        pb = (p > 0.05).astype(np.float32)
        print("  presence-only       gold-58 macro-AUC: %.4f" % macro_auc(y, pb))
        print("\n  per-label (severity | presence):")
        for j, l in enumerate(LAB):
            if len(set(y[:, j].astype(int))) > 1:
                from sklearn.metrics import roc_auc_score
                a = roc_auc_score(y[:, j], p[:, j])
                b = roc_auc_score(y[:, j], pb[:, j])
                print("    %-18s %.3f | %.3f   %s" % (l, a, b, "+" if a > b else ""))
    print("\nNOTE: n=58 cannot referee effects of the size we care about (HANDOFF 13.2).")
    print("      This is a DIRECTIONAL check; the referee is a trained model's score.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
