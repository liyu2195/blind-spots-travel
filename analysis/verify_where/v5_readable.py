"""(verify_where) Item 3: does the J-run refuse where its judge rejects only when the untrained student can 'read' the verdict? Own code.
New (duplicate-free) questions with a valid untrained greedy answer on which judges J and J' (Q M G O, a35) disagree.
rj = J rejects (else J' rejects). Probe contrast p = probe_J - probe_J' (v4 probes: layer 16, fitted on trained-question verdicts only).
predictable = sign(p) agrees with rj. Effect = Mantel-Haenszel difference of the run's final IDK rate (rj minus not rj) within 5 bins of the
untrained confidence (share of 32 samples agreeing with the greedy). J runs: FK20 judge<J>a35, non-collapsed. Placebos on the same rows:
key-trained gold runs, and the J'-trained runs (mirror: expected negative where predictable).
Aggregation: (run, J') -> run -> (student, set, J) cell -> mean over cells; also the claimed weighting (mean over (run, J') pairs).
Intervals: hierarchical bootstrap (questions within (student, set); runs within (student, set, grader)), 300 draws.
Checks: (i) right vs wrong untrained answers; (ii) shuffled-label probes (placebo split); (iii) near-duplicate exclusion (new questions with
a trained question at TF-IDF cosine >= 0.5, or the same gold answer); (iv) linear model y ~ rj + p(z) + conf bins within (run, J');
(v) home: trained questions with out-of-fold probe scores. Output: v5_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, pickle, re, itertools, numpy as np
LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 16   # probe layer (v4); checks run only for the default layer 16
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
C = (_RA + "/verify_where/cache"); OUT = (_RA + "/verify_where")
J4 = ["Q", "M", "G", "O"]; CELLS = [(st, d) for st in ("qwen7", "llama8") for d in ("pqa", "nqo", "tqa")]
lines = []
def P(s=""): print(s, flush=True); lines.append(s)
P(__doc__)

def mh(y, rj, sb):
    num = den = 0.0
    for s in np.unique(sb):
        m = sb == s; a = m & rj; b = m & ~rj; na, nb = a.sum(), b.sum()
        if na == 0 or nb == 0: continue
        w = na * nb / (na + nb); num += w * (y[a].mean() - y[b].mean()); den += w
    return num / den if den > 0 else np.nan

D = {}
for st, dsn in CELLS:
    B = pickle.load(open(f"{C}/base_{st}_{dsn}.pkl", "rb")); R = pickle.load(open(f"{C}/runs_{st}_{dsn}.pkl", "rb")); PR = pickle.load(open(f"{C}/probe_{st}_{dsn}_L{LAYER}.pkl", "rb"))["probes"]
    v = B["valid"][:, 0]; rej = {J: B["PJ"][J][:, 0] <= B["TH"][J]["a35"] for J in J4}
    sb = np.minimum(4, (B["conf"] * 5).astype(int))
    right = B["kl"][:, 0]
    # near duplicates: TF-IDF cosine of each question to its nearest trained question; shared gold answer
    tr = np.where(B["train"])[0]; vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit([B["question"][i] for i in range(1500)])
    M = vec.transform(B["question"]); sim = (M @ M[tr].T).toarray(); sim[tr, np.arange(len(tr))] = 0
    maxsim = sim.max(1); trgold = {B["gold"][i].strip().lower() for i in tr}
    neardup = (maxsim >= 0.5) | np.array([B["gold"][i].strip().lower() in trgold for i in range(1500)])
    G = defaultdict(list)
    for name, r in R.items():
        if r["c"] != 5 or r["a_new"] >= 0.95: continue
        m = re.fullmatch(r"judge([QMGO])a35", r["g"])
        if m: G["J" + m.group(1)].append((name, (r["o"] == 0).astype(float)))
        elif r["g"] == "sim00fn00": G["gold"].append((name, (r["o"] == 0).astype(float)))
    D[(st, dsn)] = dict(v=v, rej=rej, sb=sb, right=right, held=B["held"], train=B["train"], neardup=neardup, maxsim=maxsim, G=G,
                        pr={J: PR[f"rejV_{J}"]["score"] for J in J4}, oof={J: np.where(B["train"], 0.0, np.nan) for J in J4}, sh={J: PR[f"shufV_{J}"]["score"] for J in J4})
    trv = np.where(B["train"] & v)[0]
    for J in J4: D[(st, dsn)]["oof"][J][trv] = PR[f"rejV_{J}"]["oof"]
    P(f"{st} {dsn}: new questions {B['held'].sum()}, near-duplicates of a trained question (cos >= 0.5 or same gold) {neardup[B['held']].sum()} "
      f"(cos >= 0.5: {(maxsim[B['held']] >= 0.5).sum()}, same gold: {(neardup & (maxsim < 0.5))[B['held']].sum()}); runs " + ", ".join(f"{k} {len(x)}" for k, x in sorted(G.items())))

def effects(D, part="new", score="pr", subset=None, qidx=None):
    """returns rows: (st, dsn, J, run, J', eff_all, eff_pred, eff_unpred, gold_pred, gold_unpred, mir_pred, mir_unpred, share_pred)"""
    out = []
    for (st, dsn), d in D.items():
        base = d["held"] if part == "new" else d["train"]
        base = base & d["v"]
        if subset == "nodup": base = base & ~d["neardup"]
        if subset == "right": base = base & d["right"]
        if subset == "wrong": base = base & ~d["right"]
        idx = np.where(base)[0] if qidx is None else qidx[(st, dsn)][base[qidx[(st, dsn)]]]
        sc = d[score]
        for J in J4:
            if not d["G"].get("J" + J): continue
            for Jp in J4:
                if Jp == J: continue
                ii = idx[d["rej"][J][idx] != d["rej"][Jp][idx]]
                if part == "home": ii = ii[np.isfinite(sc[J][ii])]
                if len(ii) < 20: continue
                rj = d["rej"][J][ii]; p = sc[J][ii] - sc[Jp][ii]; agree = (p > 0) == rj; sb = d["sb"][ii]
                gp = [mh(y[ii][agree], rj[agree], sb[agree]) for _, y in d["G"].get("gold", [])]; gu = [mh(y[ii][~agree], rj[~agree], sb[~agree]) for _, y in d["G"].get("gold", [])]
                mp = [mh(y[ii][agree], rj[agree], sb[agree]) for _, y in d["G"].get("J" + Jp, [])]; mu = [mh(y[ii][~agree], rj[~agree], sb[~agree]) for _, y in d["G"].get("J" + Jp, [])]
                for name, y in d["G"]["J" + J]:
                    yy = y[ii]
                    out.append(dict(st=st, ds=dsn, J=J, run=name, Jp=Jp, n=len(ii), share=agree.mean(), all=mh(yy, rj, sb), pred=mh(yy[agree], rj[agree], sb[agree]),
                                    unpred=mh(yy[~agree], rj[~agree], sb[~agree]), gpred=np.nanmean(gp) if gp else np.nan, gunpred=np.nanmean(gu) if gu else np.nan,
                                    mpred=np.nanmean(mp) if mp else np.nan, munpred=np.nanmean(mu) if mu else np.nan,
                                    rows=(rj, (p - p.mean()) / (p.std() + 1e-9), sb, yy)))
    return out

KEYS = ("all", "pred", "unpred", "gpred", "gunpred", "mpred", "munpred")
def agg(rows, st, how="cell"):
    rr = [r for r in rows if r["st"] == st]
    if how == "pair": return {k: np.nanmean([r[k] for r in rr]) for k in KEYS} | {"share": np.mean([r["share"] for r in rr]), "n": len(rr)}
    run = defaultdict(list)
    for r in rr: run[(r["ds"], r["J"], r["run"])].append(r)
    cell = defaultdict(list)
    for (ds, J, nm), L in run.items(): cell[(ds, J)].append({k: np.nanmean([x[k] for x in L]) for k in KEYS})
    cm = [{k: np.nanmean([x[k] for x in L]) for k in KEYS} for L in cell.values()]
    return {k: np.nanmean([c[k] for c in cm]) for k in KEYS} | {"share": np.mean([r["share"] for r in rr]), "n": len(cm), "cells_pred_gt_unpred": sum(c["pred"] > c["unpred"] for c in cm)}

def resample(D, rng):
    Q = {}; Dn = {}
    for key, d in D.items():
        n = 1500; qi = rng.integers(0, n, n); Q[key] = qi
        G = {g: [lst[j] for j in rng.integers(0, len(lst), len(lst))] for g, lst in d["G"].items()}
        Dn[key] = dict(d, G=G)
    return Dn, Q

def report(label, part="new", score="pr", subset=None, NB=200, rng=np.random.default_rng(5)):
    rows = effects(D, part, score, subset)
    est = {st: agg(rows, st) for st in ("qwen7", "llama8")}; estp = {st: agg(rows, st, "pair") for st in ("qwen7", "llama8")}
    bs = {st: defaultdict(list) for st in est}
    for b in range(NB):
        Dn, Q = resample(D, rng); rb = effects(Dn, part, score, subset, qidx=Q)
        for st in est:
            a = agg(rb, st)
            for k in KEYS: bs[st][k].append(a[k])
    P(f"\n== {label}")
    for st in est:
        e = est[st]; ci = {k: np.nanpercentile(bs[st][k], [2.5, 97.5]) for k in KEYS}
        P(f"  {st}: cells {e['n']}, share predictable {e['share']:.2f}; J-run effect all {e['all']:+.3f} [{ci['all'][0]:+.3f},{ci['all'][1]:+.3f}]  "
          f"predictable {e['pred']:+.3f} [{ci['pred'][0]:+.3f},{ci['pred'][1]:+.3f}]  unpredictable {e['unpred']:+.3f} [{ci['unpred'][0]:+.3f},{ci['unpred'][1]:+.3f}]  (pred > unpred in {e.get('cells_pred_gt_unpred')}/{e['n']} cells)")
        P(f"          gold placebo pred {e['gpred']:+.3f} [{ci['gpred'][0]:+.3f},{ci['gpred'][1]:+.3f}] unpred {e['gunpred']:+.3f} [{ci['gunpred'][0]:+.3f},{ci['gunpred'][1]:+.3f}];  "
          f"J'-run mirror pred {e['mpred']:+.3f} [{ci['mpred'][0]:+.3f},{ci['mpred'][1]:+.3f}] unpred {e['munpred']:+.3f} [{ci['munpred'][0]:+.3f},{ci['munpred'][1]:+.3f}]")
        ep = estp[st]; P(f"          claimed weighting (mean over {ep['n']} (run, J') pairs): all {ep['all']:+.3f} pred {ep['pred']:+.3f} unpred {ep['unpred']:+.3f} gold {ep['gpred']:+.3f}/{ep['gunpred']:+.3f}")
    return rows

rows = report(f"NEW questions, layer-{LAYER} probe fitted on trained-question verdicts (main)")
# (iv) linear model within (run, J'): y ~ rj + pz + conf bins
for st in ("qwen7", "llama8"):
    br, bp = [], []
    for r in rows:
        if r["st"] != st: continue
        rj, pz, sb, y = r["rows"]; X = np.c_[rj.astype(float), pz, np.eye(5)[sb]]
        beta = np.linalg.lstsq(X, y, rcond=None)[0]; br.append(beta[0]); bp.append(beta[1])
    P(f"  linear model within (run, J') pairs, {st}: coef rj {np.mean(br):+.3f} (positive in {np.mean(np.array(br) > 0):.2f}), coef probe contrast (per sd) {np.mean(bp):+.3f} (positive in {np.mean(np.array(bp) > 0):.2f})")
if LAYER == 16: report("check (i): only untrained answers that are key-RIGHT", subset="right", NB=100)
if LAYER == 16: report("check (i): only untrained answers that are key-WRONG", subset="wrong", NB=100)
if LAYER == 16: report("check (ii): placebo split by SHUFFLED-label probes (fitted on permuted trained-question verdicts)", score="sh", NB=100)
if LAYER == 16: report("check (iii): near-duplicates of trained questions removed", subset="nodup", NB=100)
if LAYER == 16: report("check (v): HOME (trained questions), out-of-fold probe scores", part="home", score="oof", NB=100)
open(f"{OUT}/v5_out.txt" if LAYER == 16 else f"{OUT}/v5_out_L{LAYER}.txt", "w").write("\n".join(lines) + "\n")
