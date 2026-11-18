"""(stats) Claim 6: a judge-K run refuses where K rejects only where the untrained student can 'read' K's verdict.
Rows: NEW (duplicate-free) questions with a valid untrained greedy answer on which judges J and J' (Q M G O, a35 cuts) disagree; rj = J rejects.
Probe contrast p = probe_J - probe_J' (s03 probes, layer 16, fitted on trained-question verdicts only); predictable = (p > 0) == rj.
Effect of a J-trained run on a row set = Mantel-Haenszel difference of its final IDK rate (rj minus not rj) within 5 bins of untrained confidence.
J runs: FK20 judge<J>a35, IDK credit 0.5, new-question abstention < 0.95. Aggregation: (run, J') -> run -> (student, dataset, J) cell -> cells.
PRIMARY estimand: interaction I = effect(predictable) - effect(unpredictable), per student and pooled; also each half.
'Unreadable verdicts do not transfer': one-sided test of effect(unpredictable) < +0.05 (upper 95% bound reported).
Uncertainty: crossed bootstrap (1000): questions within (student, dataset), runs within (student, dataset, J).
Permutation null for I: the predictable / unpredictable labels permuted within (student, dataset, J, J', rj, confidence bin) strata (same draw for
  every run), 2000 draws. Placebo: shuffled-label probes. Robustness: layer 24; near-duplicates of trained questions removed (TF-IDF cosine >= 0.5
  or same gold answer); key-right untrained answers only; raw (unstratified) differences; home questions with out-of-fold probe scores.
Output: c6_readable_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, re, pickle, numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
sys.path.insert(0, (_RA + "/stats"))
from s00_common import *
P = Log(f"{HERE}/c6_readable_out.txt"); P(__doc__)
rng = np.random.default_rng(606)
LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 16
TH = {J: thresholds(J) for J in J4}
CELLS = {}
for st in STS:
    for ds in DSS:
        dsn = DSN[ds]; B = pickle.load(open(f"{CACHE}/base_{st}_{ds}.pkl", "rb")); R = pickle.load(open(f"{CACHE}/runs_{st}_{ds}.pkl", "rb"))
        PR = pickle.load(open(f"{CACHE}/probe_{st}_{ds}_L{LAYER}.pkl", "rb"))["probes"]
        v = B["valid"][:, 0]; rej = {J: B["PJ"][J][:, 0] <= TH[J][f"{st}_{dsn}"]["t_a35"] for J in J4}
        sb = np.minimum(4, (B["conf"] * 5).astype(int))
        tr = np.where(B["train"])[0]; vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(B["question"])
        M = vec.transform(B["question"]); sim = (M @ M[tr].T).toarray(); sim[tr, np.arange(len(tr))] = 0
        trg = {B["gold"][i].strip().lower() for i in tr}
        neardup = (sim.max(1) >= 0.5) | np.array([B["gold"][i].strip().lower() in trg for i in range(B["N"])])
        Y = {}
        for n, r in sorted(R.items()):
            m = re.fullmatch(r"judge([QMGO])a35", r["g"])
            if not m or r["tag"] != "FK20" or r["c"] != "05": continue
            if (r["o"][B["new"]] == 0).mean() >= 0.95: continue
            Y.setdefault(m.group(1), []).append((r["seed"], (r["o"] == 0).astype(float)))
        CELLS[(st, ds)] = dict(v=v, rej=rej, sb=sb, new=B["new"], train=B["train"], right=B["kl"][:, 0], neardup=neardup,
                               sc={J: PR[f"rej_{J}"]["score"] for J in J4}, oof={J: PR[f"rej_{J}"]["oof"] for J in J4}, sh={J: PR[f"shuf_{J}"]["score"] for J in J4},
                               Y={J: np.array([y for s, y in L]) for J, L in Y.items()}, seeds={J: [s for s, y in L] for J, L in Y.items()})
        P(f"{st} {ds}: J runs " + ", ".join(f"{J} {len(CELLS[(st, ds)]['seeds'][J])}" for J in J4 if J in CELLS[(st, ds)]["seeds"]) + f"; near-duplicate new questions {int((neardup & B['new']).sum())}")
def build(part="new", score="sc", subset=None):
    """list of row blocks: dict(st, ds, J, Jp, ii (question idx), rj, agree, sb)"""
    out = []
    for (st, ds), d in CELLS.items():
        base = (d["new"] if part == "new" else d["train"]) & d["v"]
        if subset == "nodup": base = base & ~d["neardup"]
        if subset == "right": base = base & d["right"]
        sc = d[score]
        for J in J4:
            if J not in d["Y"]: continue
            for Jp in J4:
                if Jp == J: continue
                ii = np.where(base & (d["rej"][J] != d["rej"][Jp]) & np.isfinite(sc[J]) & np.isfinite(sc[Jp]))[0]
                if len(ii) < 20: continue
                rj = d["rej"][J][ii]; p = sc[J][ii] - sc[Jp][ii]
                out.append(dict(st=st, ds=ds, J=J, Jp=Jp, ii=ii, rj=rj, agree=(p > 0) == rj, sb=d["sb"][ii]))
    return out
def mh_runs(Yb, rj, sb, w, mask):
    """MH difference (rj minus not rj) within confidence bins, for every run (rows of Yb), weights w, on rows in mask."""
    num = np.zeros(Yb.shape[0]); den = 0.0
    for s in range(5):
        a = mask & rj & (sb == s); b = mask & ~rj & (sb == s); na = w[a].sum(); nb = w[b].sum()
        if na == 0 or nb == 0: continue
        wt = na * nb / (na + nb); num += wt * (Yb[:, a] @ w[a] / na - Yb[:, b] @ w[b] / nb); den += wt
    return num / den if den > 0 else np.full(Yb.shape[0], np.nan)
def raw_runs(Yb, rj, sb, w, mask):
    a = mask & rj; b = mask & ~rj; na = w[a].sum(); nb = w[b].sum()
    return Yb[:, a] @ w[a] / na - Yb[:, b] @ w[b] / nb if na > 0 and nb > 0 else np.full(Yb.shape[0], np.nan)
def effects(blocks, W=None, sel=None, agree_over=None, est=mh_runs, weighting="cell"):
    """returns dict who -> (pred, unpred, all, I)"""
    cell = defaultdict(lambda: defaultdict(list))   # (st, ds, J) -> key -> list over J' of run-mean effects
    pairs = defaultdict(list)
    for k, b in enumerate(blocks):
        d = CELLS[(b["st"], b["ds"])]; Yall = d["Y"][b["J"]]
        Ys = Yall if sel is None else Yall[sel[(b["st"], b["ds"], b["J"])]]
        Yb = Ys[:, b["ii"]]; w = np.ones(len(b["ii"])) if W is None else W[(b["st"], b["ds"])][b["ii"]]
        ag = b["agree"] if agree_over is None else agree_over[k]
        e_p = est(Yb, b["rj"], b["sb"], w, ag); e_u = est(Yb, b["rj"], b["sb"], w, ~ag); e_a = est(Yb, b["rj"], b["sb"], w, np.ones(len(ag), bool))
        for nm, e in (("pred", e_p), ("unpred", e_u), ("all", e_a)):
            cell[(b["st"], b["ds"], b["J"])][nm].append(np.nanmean(e)); pairs[(b["st"], nm)].extend(list(e))
    out = {}
    for who in ("qwen7", "llama8"):
        cs = [c for k, c in cell.items() if k[0] == who]
        if weighting == "cell": r = {nm: np.nanmean([np.nanmean(c[nm]) for c in cs]) for nm in ("pred", "unpred", "all")}
        else: r = {nm: np.nanmean(pairs[(who, nm)]) for nm in ("pred", "unpred", "all")}
        r["I"] = r["pred"] - r["unpred"]; r["cells"] = len(cs); r["cells_pos"] = sum(np.nanmean(c["pred"]) > np.nanmean(c["unpred"]) for c in cs); out[who] = r
    out["pooled"] = {k: (out["qwen7"][k] + out["llama8"][k]) / 2 for k in ("pred", "unpred", "all", "I")}
    out["pooled"]["cells"] = out["qwen7"]["cells"] + out["llama8"]["cells"]; out["pooled"]["cells_pos"] = out["qwen7"]["cells_pos"] + out["llama8"]["cells_pos"]
    return out
def boot(blocks, NB, est=mh_runs, weighting="cell"):
    reps = []
    for b in range(NB):
        W = {k: np.bincount(rng.integers(0, 1500, 1500), minlength=1500).astype(float) for k in CELLS}
        sel = {(st, ds, J): rng.integers(0, len(d["Y"][J]), len(d["Y"][J])) for (st, ds), d in CELLS.items() for J in d["Y"]}
        reps.append(effects(blocks, W, sel, est=est, weighting=weighting))
    return reps
def report(lab, blocks, NB=1000, est=mh_runs, weighting="cell", perm=False):
    e = effects(blocks, est=est, weighting=weighting); reps = boot(blocks, NB, est, weighting)
    P(f"\n== {lab}")
    res = {}
    for who in ("qwen7", "llama8", "pooled"):
        ci = {k: pct([r[who][k] for r in reps]) for k in ("pred", "unpred", "all", "I")}
        ub = np.percentile([r[who]["unpred"] for r in reps], 95); p_ni = np.mean(np.array([r[who]["unpred"] for r in reps]) >= 0.05)
        res[who] = dict(e=e[who], ci=ci, ub_unpred=ub, p_unpred_lt05=p_ni)
        P(f"  {who:6s} I {e[who]['I']:+.3f} [{ci['I'][0]:+.3f}, {ci['I'][1]:+.3f}] | predictable {e[who]['pred']:+.3f} [{ci['pred'][0]:+.3f}, {ci['pred'][1]:+.3f}]  "
          f"unpredictable {e[who]['unpred']:+.3f} [{ci['unpred'][0]:+.3f}, {ci['unpred'][1]:+.3f}] (one-sided 95% upper bound {ub:+.3f}; P*(>= +0.05) {p_ni:.4f})  "
          f"all {e[who]['all']:+.3f} | cells pred > unpred {e[who]['cells_pos']}/{e[who]['cells']}")
    if perm:
        strata = []
        for b in blocks:
            key = b["sb"] * 2 + b["rj"].astype(int); strata.append([np.where(key == k)[0] for k in np.unique(key)])
        null = defaultdict(list)
        for t in range(2000):
            ao = []
            for b, S_ in zip(blocks, strata):
                a = b["agree"].copy()
                for ix in S_: a[ix] = a[ix][rng.permutation(len(ix))]
                ao.append(a)
            ee = effects(blocks, agree_over=ao, est=est, weighting=weighting)
            for who in ("qwen7", "llama8", "pooled"): null[who].append(ee[who]["I"])
        for who in ("qwen7", "llama8", "pooled"):
            nv = np.array(null[who]); res[who]["p_perm"] = (np.sum(nv >= e[who]["I"]) + 1) / (len(nv) + 1)
            P(f"  permutation of predictable labels within (cell, J', rj, conf bin) strata, {who}: null mean {nv.mean():+.4f}, 99.9th pct {np.percentile(nv, 99.9):+.3f}; p = {res[who]['p_perm']:.5f}")
    return res
MAIN = report(f"PRIMARY: new questions, layer-{LAYER} probes", build(), 1000, perm=(LAYER == 16))
if LAYER == 16:
    ROB = {}
    ROB["shuf"] = report("placebo: shuffled-label probes define the split", build(score="sh"), 300)
    ROB["nodup"] = report("near-duplicates of trained questions removed", build(subset="nodup"), 300)
    ROB["right"] = report("key-right untrained answers only", build(subset="right"), 300)
    ROB["raw"] = report("raw (unstratified) differences", build(), 300, est=raw_runs)
    ROB["pairs"] = report("weighting: mean over (run, J') pairs", build(), 300, weighting="pairs")
    ROB["home"] = report("HOME: trained questions, out-of-fold probe scores", build("home", "oof"), 300)
    P("\n== runs and seeds: " + "; ".join(f"{st} {ds} " + " ".join(f"{J}{d['seeds'][J]}" for J in J4 if J in d["seeds"]) for (st, ds), d in CELLS.items()))
    pickle.dump(dict(MAIN=MAIN, ROB=ROB), open(f"{HERE}/c6_summary.pkl", "wb"))
else:
    pickle.dump(dict(MAIN=MAIN), open(f"{HERE}/c6_summary_L{LAYER}.pkl", "wb"))
P.path = f"{HERE}/c6_readable_out.txt" if LAYER == 16 else f"{HERE}/c6_readable_out_L{LAYER}.txt"; P.save()
