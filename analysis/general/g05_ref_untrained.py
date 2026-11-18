"""(explorer 'general') Reference-guided vs reference-free judges on the UNTRAINED answers of the three small students
(TriviaQA, 1500 questions each; plain-prompt greedy answers of <st>_goldstrict_h1 ck0, the answers st/res/run1/keyfp scored).
Judges: Q9 / G12 / Q4 / Q2 reference-free and Q9ref / G12ref / Q4ref / Q2ref (the judge also sees the dataset's 'gold' string;
st_common.TQ_JUDGE_REF). Verdict = P(Yes) > 0.5. Truth: loose answer key (alias match); 'audited' variant on the subset whose
(question, answer) Sonnet graded closed-book (work/frontier, phi4m / smol3 only): right = key-right or Sonnet >= 4,
wrong = key-wrong and Sonnet <= 2.
 (1) acceptance L, false-reject FR = P(reject | right), false-accept FA = P(accept | wrong), AUROC; question bootstrap.
 (2) what each judge rejects among key-right answers: exact match to the displayed gold string vs alias-only match vs loose-only
     (correct by the loose key, not the strict key: extra words); untrained confidence (share of 32 IDK-prompt samples agreeing).
 (3) question-tied vs answer-tied errors across students: for questions where two students gave DIFFERENT key-right (key-wrong)
     answer lines, phi correlation of the two false rejects (false accepts).
 (4) overlap of reference-guided and reference-free false rejects; agreement with the key (Cohen kappa).
Output: g05_ref_untrained.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import json, os, sys, pickle, glob
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gcommon import setup, sa, IDK, C, BASE, HERE, read_eval, same

rng = np.random.default_rng(5)
probs, H1, HELD = setup("tqa")
JUD = ("Q9", "Q9ref", "G12", "G12ref", "Q4", "Q4ref", "Q2", "Q2ref")
LK = {J: pickle.load(open(f"{HERE}/g02_lookup_{J}_tqa.pkl", "rb")) for J in JUD}
FR_ = (_RX + "/frontier")
pairs_v1 = json.load(open(f"{FR_}/pairs_v1.json")); byid = {p["id"]: (p["idx"], p["norm"]) for p in pairs_v1}
SON = {}
for f in sorted(glob.glob(f"{FR_}/out_sonnet_*.tsv")):
    for l in open(f):
        p = l.rstrip("\n").split("\t")
        if len(p) >= 2 and p[0].strip().isdigit() and p[1].strip().isdigit() and int(p[0]) in byid: SON[byid[int(p[0])]] = int(p[1])
out = open(f"{HERE}/g05_ref_untrained.txt", "w")
def P(*a): print(*a, file=out); print(*a)

D = {}
for st in ("phi4m", "smol3", "granite"):
    g, _ = read_eval(f"{st}_goldstrict_h1", 0)
    samp = defaultdict(list)
    bf = f"{BASE}/{st}_tqa_1500_n32_s1.jsonl" if st != "granite" else f"{BASE}/granite_tqa_1500_idk_n32_s1.jsonl"
    for l in open(bf):
        r = json.loads(l)
        if r["s"] >= 0: x = sa(r); samp[r["idx"]].append(x if (x and not IDK.match(x)) else "")
    rows = []
    for i in range(1500):
        line = C.tq_line(g[i]["text"]); x = C.parse_tq(g[i]["text"])
        if not x or C.is_idk(x): continue
        al = set(probs[i]["aliases"]); gold = C.normalize(probs[i]["gold"])
        key = C.tq_correct(x, al); strict = C.tq_strict(x, al)
        sv = SON.get((i, x)) if st != "granite" else None
        conf = sum(same(x, y) for y in samp[i] if y) / 32
        pj = {J: LK[J].get((i, line)) for J in JUD}
        rows.append(dict(i=i, line=line, x=x, key=key, strict=strict, exact=(x == gold), son=sv, conf=conf, home=i in H1, pj=pj,
                         gold_tok=set(gold.split()), x_tok=set(x.split())))
    D[st] = rows
    cov = {J: np.mean([r["pj"][J] is not None for r in rows]) for J in JUD}
    P(f"{st}: {len(rows)} valid untrained plain-prompt answers; key-right {np.mean([r['key'] for r in rows]):.3f}; Sonnet-graded {np.mean([r['son'] is not None for r in rows]):.3f}; "
      f"score coverage " + " ".join(f"{J} {c:.3f}" for J, c in cov.items()))

def boot(f, rows, nb=1000):
    v0 = f(rows); bs = []
    idx = np.arange(len(rows))
    for _ in range(nb):
        rr = [rows[k] for k in rng.integers(0, len(rows), len(rows))]; bs.append(f(rr))
    return v0, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)

P("\n(1) acceptance L, false-reject FR (key-right rejected), false-accept FA (key-wrong accepted), AUROC (key truth); verdict P(Yes) > 0.5")
for st, rows in D.items():
    for J in JUD:
        R = [r for r in rows if r["pj"][J] is not None]
        if not R: continue
        acc = lambda rr: np.mean([r["pj"][J] > 0.5 for r in rr])
        fr = lambda rr: np.mean([r["pj"][J] <= 0.5 for r in rr if r["key"]])
        fa = lambda rr: np.mean([r["pj"][J] > 0.5 for r in rr if not r["key"]])
        y = np.array([r["key"] for r in R]); s = np.array([r["pj"][J] for r in R])
        a0 = boot(acc, R, 300); f0 = boot(fr, R, 300); g0 = boot(fa, R, 300)
        line = f"  {st:7s} {J:6s}: L {a0[0]:.3f} | FR {f0[0]:.3f} [{f0[1]:.3f}, {f0[2]:.3f}] | FA {g0[0]:.3f} [{g0[1]:.3f}, {g0[2]:.3f}] | AUROC {roc_auc_score(y, s):.3f}"
        A = [r for r in R if r["son"] is not None and (r["key"] or r["son"] >= 4 or r["son"] <= 2)]
        if len(A) > 100:
            right = np.array([r["key"] or r["son"] >= 4 for r in A]); acc_ = np.array([r["pj"][J] > 0.5 for r in A])
            line += f" || audited subset n={len(A)}: FR {np.mean(~acc_[right]):.3f}, FA {np.mean(acc_[~right]):.3f}, key-wrong-but-Sonnet-right answers rejected {np.mean([r['pj'][J] <= 0.5 for r in A if (not r['key']) and r['son'] >= 4]):.3f} (n={sum((not r['key']) and r['son'] >= 4 for r in A)})"
        P(line)

P("\n(2) false-reject rate among key-right answers, by kind of match and by untrained confidence")
for st, rows in D.items():
    for J in ("Q9", "Q9ref", "G12", "G12ref", "Q2", "Q2ref"):
        R = [r for r in rows if r["key"] and r["pj"][J] is not None]
        grp = {"exact gold": [r for r in R if r["exact"]], "alias-only strict": [r for r in R if not r["exact"] and r["strict"]],
               "loose-only (extra words)": [r for r in R if not r["strict"]],
               "conf<0.5": [r for r in R if r["conf"] < 0.5], "conf>=0.9": [r for r in R if r["conf"] >= 0.9]}
        P(f"  {st:7s} {J:6s}: " + " | ".join(f"{k} {np.mean([r['pj'][J] <= 0.5 for r in v]):.3f} (n={len(v)})" for k, v in grp.items()))
P("\n   false-accept rate among key-wrong answers, by token overlap with the gold string")
for st, rows in D.items():
    for J in ("Q9", "Q9ref", "G12", "G12ref", "Q2", "Q2ref"):
        R = [r for r in rows if not r["key"] and r["pj"][J] is not None]
        ov = [r for r in R if r["gold_tok"] & r["x_tok"]]; no = [r for r in R if not (r["gold_tok"] & r["x_tok"])]
        so = [r for r in R if r["son"] is not None and r["son"] >= 4]
        P(f"  {st:7s} {J:6s}: shares a word with gold {np.mean([r['pj'][J] > 0.5 for r in ov]):.3f} (n={len(ov)}) | no shared word {np.mean([r['pj'][J] > 0.5 for r in no]):.3f} (n={len(no)})"
          + (f" | key-wrong but Sonnet-right {np.mean([r['pj'][J] > 0.5 for r in so]):.3f} (n={len(so)})" if so else ""))

P("\n(3) question-tied vs answer-tied: phi correlation of errors between two students' DIFFERENT answer lines to the same question")
byq = defaultdict(list)
for st, rows in D.items():
    for r in rows: byq[r["i"]].append(r)
def phi(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() == 0 or b.std() == 0: return np.nan
    return np.corrcoef(a, b)[0, 1]
for J in JUD:
    fr_pairs, fa_pairs = [], []
    for i, rs in byq.items():
        for u in range(len(rs)):
            for v in range(u + 1, len(rs)):
                a, b = rs[u], rs[v]
                if a["line"] == b["line"] or C.normalize(a["line"]) == C.normalize(b["line"]) or a["pj"][J] is None or b["pj"][J] is None: continue
                if a["key"] and b["key"]: fr_pairs.append((a["pj"][J] <= 0.5, b["pj"][J] <= 0.5, i))
                if not a["key"] and not b["key"]: fa_pairs.append((a["pj"][J] > 0.5, b["pj"][J] > 0.5, i))
    def bphi(pp):
        if len(pp) < 30: return np.nan, np.nan, np.nan
        v0 = phi([p[0] for p in pp], [p[1] for p in pp]); qs = sorted({p[2] for p in pp}); byi = defaultdict(list)
        for p in pp: byi[p[2]].append(p)
        bs = []
        for _ in range(500):
            sel = [p for q in rng.choice(qs, len(qs)) for p in byi[q]]; bs.append(phi([p[0] for p in sel], [p[1] for p in sel]))
        return v0, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)
    a, b = bphi(fr_pairs), bphi(fa_pairs)
    P(f"  {J:6s}: false rejects phi {a[0]:+.3f} [{a[1]:+.3f}, {a[2]:+.3f}] over {len(fr_pairs)} pairs of different right answers (base rate {np.mean([p[0] for p in fr_pairs]) if fr_pairs else np.nan:.3f}) | "
      f"false accepts phi {b[0]:+.3f} [{b[1]:+.3f}, {b[2]:+.3f}] over {len(fa_pairs)} pairs of different wrong answers (base rate {np.mean([p[0] for p in fa_pairs]) if fa_pairs else np.nan:.3f})")

P("\n(4) overlap of reference-guided and reference-free false rejects (same key-right answers), and agreement with the key")
def kappa(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float); po = np.mean(x == y); px, py = x.mean(), y.mean(); pe = px * py + (1 - px) * (1 - py)
    return (po - pe) / (1 - pe)
for st, rows in D.items():
    for J in ("Q9", "G12", "Q4", "Q2"):
        R = [r for r in rows if r["key"] and r["pj"][J] is not None and r["pj"][J + "ref"] is not None]
        f, fr = np.array([r["pj"][J] <= 0.5 for r in R]), np.array([r["pj"][J + "ref"] <= 0.5 for r in R])
        A = [r for r in rows if r["pj"][J] is not None and r["pj"][J + "ref"] is not None]
        k_free = kappa([r["key"] for r in A], [r["pj"][J] > 0.5 for r in A]); k_ref = kappa([r["key"] for r in A], [r["pj"][J + "ref"] > 0.5 for r in A])
        P(f"  {st:7s} {J}: FR free {f.mean():.3f}, ref {fr.mean():.3f}; P(ref rejects | free rejects) {fr[f].mean():.3f}, P(ref rejects | free accepts) {fr[~f].mean():.3f}, phi {phi(f, fr):+.3f}; "
          f"kappa with key: free {k_free:.3f}, ref {k_ref:.3f}")
pickle.dump(D, open(f"{HERE}/g05_rows.pkl", "wb"))
out.close()
