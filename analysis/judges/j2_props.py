"""(judges) Q4 data: one row per trained run (FK20, KL 0.2, 120 steps, IDK credit from config) with
  (a) PRE-TRAINING properties of its grader, computed from the untrained student's 32 samples on the TRAINED questions, and
  (b) the run's outcomes on the NEW (held-out) questions (from the shared cache bm.build; greedy final answers).
Graders are re-implemented from the trainer (st_grpo_v13.synth_rewards, judge threshold rule P(Yes) > JT, Sonnet level >= 4,
frozen / blind keys from the run's TRAINF alias file). Random simulated graders enter as expected acceptance per sample.
Truth for error rates: strict key (trainer's gold reward) and audited (loose key OR Sonnet >= 4).
Output: j2_table.pkl (list of dicts) and j2_out.txt (per pair: grader, runs, properties, mean outcomes).
Usage: nice -n 10 python3 j2_props.py"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, re, json, pickle, hashlib, numpy as np
from collections import defaultdict
sys.path.insert(0, (_RX + "/story/B_refusal_map")); sys.path.insert(0, (_RX + "/common"))
import bm
from rc import C
HERE = (_RA + "/judges"); DATA = _RD
DSN = {"h1": "tqa", "pq": "pqa", "nq": "nqo"}
TH = {J: json.load(open(f"{_RX}/final/thresholds_{J}.json")) for J in ["Q", "M", "G", "O", "R"]}
ANY = "zzzanyzzz"

def hunit(*parts): return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12], 16) / float(16 ** 12)

def configs():
    cfg = {}
    for l in open(f"{HERE}/run_configs_raw.txt"):
        o, c = l.rstrip("\n").split(" | ", 1); o = o.split("=", 1)[1]
        d = dict(kv.split("=", 1) for kv in c.split() if "=" in kv); cfg[o] = d
    return cfg

def icc(y, g):
    y = np.asarray(y, float); g = np.asarray(g)
    if len(y) < 10: return np.nan
    u, inv, cnt = np.unique(g, return_inverse=True, return_counts=True)
    keep = cnt[inv] >= 2; y = y[keep]; g = g[keep]
    if len(y) < 10: return np.nan
    u, inv, cnt = np.unique(g, return_inverse=True, return_counts=True)
    k = len(u); N = len(y); m = y.mean(); gm = np.bincount(inv, weights=y) / cnt
    ssb = (cnt * (gm - m) ** 2).sum(); ssw = ((y - gm[inv]) ** 2).sum()
    if k < 2 or N - k < 1: return np.nan
    msb = ssb / (k - 1); msw = ssw / (N - k); n0 = (N - (cnt ** 2).sum() / N) / (k - 1); d = msb + (n0 - 1) * msw
    return (msb - msw) / d if d > 0 else np.nan

def accept_prob(g, cfgr, D, m, st, ds, seed):
    """expected acceptance of each untrained sample (rows m: trained questions, s >= 0, valid non-IDK); None if not modelled."""
    idx = D["idx"][m]; X = [D["X"][k] for k in np.where(m)[0]]; ks = D["ks"][m].astype(float); words = D["words"][m]
    mj = re.fullmatch(r"judge([QMGOR])(a35|a50|t50)", g)
    if mj:
        J, tag = mj.groups(); jt = float(cfgr["JT"]) if cfgr and "JT" in cfgr else (0.5 if tag == "t50" else TH[J][f"{st}_{DSN[ds]}"]["t_" + tag])
        return (D["PJ"][J][m] > jt).astype(float)
    if g == "apiS55": return (D["son"][m] >= 4).astype(float)
    if cfgr is None:
        ms = re.fullmatch(r"sim(\d\d)fn(\d\d)", g)
        if not ms: return None
        cfgr = dict(FPK=str(int(ms.group(1)) / 100), FPU=str(int(ms.group(1)) / 100), FN=str(int(ms.group(2)) / 100), FPMODE="random")
    if cfgr.get("REWARD", "synth") not in ("synth",) or "QUORUM" in cfgr: return None
    fpk, fpu, fn = float(cfgr.get("FPK", 0)), float(cfgr.get("FPU", 0)), float(cfgr.get("FN", 0))
    fnk, fnu = float(cfgr.get("FNK", -1)), float(cfgr.get("FNU", -1)); mode = cfgr.get("FPMODE", "random")
    knows = {}
    if "KNOWSF" in cfgr: knows = {r["idx"]: bool(r["knows"]) for r in map(json.loads, open(f"{DATA}/{cfgr['KNOWSF']}"))}
    kn = np.array([knows.get(int(i), False) for i in idx])
    if "TRAINF" in cfgr and any(t in cfgr["TRAINF"] for t in ("frz", "blind")):
        al = {r["idx"]: r["aliases"] for r in map(json.loads, open(f"{DATA}/{cfgr['TRAINF']}"))}
        ks = np.array([1.0 if ANY in al[int(i)] else float(C.tq_strict(x, set(al[int(i)]))) for i, x in zip(idx, X)])
    fnv = np.where(kn, fnk, fnu) if fnk >= 0 else np.full(len(idx), fn)
    fpv = np.where(kn, fpk, fpu)
    if mode == "hash":
        pr = np.array([0.0 if (f > 0 and hunit("fn", seed, int(i), x) < f) else 1.0 for i, x, f in zip(idx, X, fnv)])
        pw = np.array([1.0 if hunit("fp", seed, int(i), x) < f else 0.0 for i, x, f in zip(idx, X, fpv)])
    elif mode == "long":
        pr = 1 - fnv; pw = np.where(words >= 5, fpv, 0.0)
    else:
        pr = 1 - fnv; pw = fpv
    return np.where(ks > 0, pr, pw)

def context(D, m, trq):
    idx = D["idx"][m]; X = [D["X"][k] for k in np.where(m)[0]]
    uq, qi = np.unique(idx, return_inverse=True)
    ck = qi.astype(np.int64) * 1000 + D["clu"][m].astype(np.int64); uc, ci = np.unique(ck, return_inverse=True)
    cq = (uc // 1000).astype(int)
    seen = set(); first = np.zeros(len(idx), bool)
    for k in range(len(idx)):
        key = (int(idx[k]), X[k])
        if key not in seen: seen.add(key); first[k] = True
    pos = {int(q): j for j, q in enumerate(uq)}
    return dict(idx=idx, qi=qi, ci=ci, cq=cq, nq=len(uq), first=first, trpos=np.array([pos.get(int(i), -1) for i in trq]),
                T=(D["kl"] | (D["son"] >= 4))[m], ks=D["ks"][m], cs=D["csize"][m] / 32.0, words=D["words"][m])

def props(p, X_, credit, Qheld_conf, Qtr_conf):
    qi, ci, cq, T, ks, cs, words = X_["qi"], X_["ci"], X_["cq"], X_["T"], X_["ks"], X_["cs"], X_["words"]
    r = dict(A=p.mean(), FPRk=p[~ks].mean(), FNRk=1 - p[ks].mean(), FPRa=p[~T].mean(), FNRa=1 - p[T].mean())
    r["Ja"] = (1 - r["FNRa"]) - r["FPRa"]; r["preca"] = (p * T).sum() / max(1e-9, p.sum())
    nq = X_["nq"]; cnt = np.bincount(qi, minlength=nq); Rq0 = np.bincount(qi, weights=p, minlength=nq) / np.maximum(1, cnt)
    cc = np.bincount(ci); cm = np.bincount(ci, weights=p) / cc
    best = np.zeros(nq); ok = cc >= 2; np.maximum.at(best, cq[ok], cm[ok])
    tp = X_["trpos"]; Rq = np.where(tp >= 0, Rq0[np.maximum(tp, 0)], 0.0); Rbq = np.where(tp >= 0, best[np.maximum(tp, 0)], 0.0)
    r["W"] = (Rq > credit).mean(); r["Wbest"] = (Rbq > credit).mean(); r["Rmean"] = Rq.mean()
    for b in (4, 8, 16, 32):
        r[f"Ws{b}"] = (1 / (1 + np.exp(-b * (Rq - credit)))).mean(); r[f"Wbs{b}"] = (1 / (1 + np.exp(-b * (Rbq - credit)))).mean()
    r["margin_pos"] = np.maximum(Rq - credit, 0).mean(); r["margin_neg"] = np.maximum(credit - Rq, 0).mean()
    r["Rq"] = Rq.astype(np.float32); r["Rbq"] = Rbq.astype(np.float32)
    r["TPRa_sure"] = p[T & (cs >= 31 / 32)].mean() if (T & (cs >= 31 / 32)).any() else np.nan
    r["FPRa_q"] = np.nan
    bins = np.array([-0.01, 0.0, 0.25, 0.5, 0.75, 0.9, 0.97, 1.01])
    bt = np.digitize(Qtr_conf, bins); bh = np.digitize(Qheld_conf, bins)
    rc_ = {b: Rq[bt == b].mean() for b in np.unique(bt)}; wb = {b: (Rq[bt == b] > credit).mean() for b in np.unique(bt)}
    r["Pcross"] = np.mean([rc_.get(b, 0.0) <= credit for b in bh]); r["PWbin"] = np.mean([1 - wb.get(b, 0.0) for b in bh])
    r["slope_conf"] = np.corrcoef(Rq, Qtr_conf)[0, 1] if Rq.std() > 0 else np.nan
    fn = (1 - p) * T; fp = p * (~T); f1 = X_["first"]
    r["iccFNu"] = icc(fn[f1 & T], qi[f1 & T]); r["iccFPu"] = icc(fp[f1 & ~T], qi[f1 & ~T])
    def qshare(err, mask):
        e = err[mask]; q = qi[mask]
        if e.var() == 0: return np.nan
        u, inv = np.unique(q, return_inverse=True); qm = np.bincount(inv, weights=e) / np.bincount(inv)
        return ((qm[inv] - e.mean()) ** 2).mean() / e.var()
    r["qFN"] = qshare(fn, T); r["qFP"] = qshare(fp, ~T)
    w = ~T
    r["fp_sure"] = p[w & (cs >= 0.5)].mean() if (w & (cs >= 0.5)).any() else np.nan
    r["fp_unsure"] = p[w & (cs < 0.5)].mean() if (w & (cs < 0.5)).any() else np.nan
    r["fp_long"] = p[w & (words >= 5)].mean() if (w & (words >= 5)).any() else np.nan
    r["fn_sure"] = 1 - p[T & (cs >= 31 / 32)].mean() if (T & (cs >= 31 / 32)).any() else np.nan
    return r

def main():
    cfg = configs(); rows = []; out = open(f"{HERE}/j2_out.txt", "w")
    for st in ("qwen7", "llama8"):
        for ds in ("pq", "nq", "h1"):
            P = bm.build(st, ds); Qb = P["Q"]; D = pickle.load(open(f"{HERE}/cache_{st}_{DSN[ds]}.pkl", "rb"))
            tr = D["Q"]["train"][D["idx"]]; m = tr & D["valid"] & (D["s"] >= 0)
            trq = np.where(D["Q"]["train"])[0]
            g_rows = {int(D["idx"][k]): k for k in np.where(D["s"] < 0)[0]}
            conf_all = np.array([D["csize"][g_rows[i]] / 32.0 if D["valid"][g_rows[i]] else 0.0 for i in range(1500)])
            Qtr_conf = conf_all[trq]; Qheld_conf = conf_all[Qb["idx"]]   # same confidence definition on trained and held-out questions
            groups = defaultdict(list)
            for name, v in P["R"].items():
                if v["kind"] != "FK20": continue
                groups[v["g"]].append(name)
            CX = context(D, m, trq); cache_props = {}
            for g, names in sorted(groups.items()):
                for name in names:
                    c = cfg.get(name); seed = int(re.search(r"_s(\d+)_", name).group(1))
                    credit = float(c["IDK"]) if c and "IDK" in c else 0.5
                    key = (g, seed if (c and c.get("FPMODE") == "hash") else -1, credit)
                    if key not in cache_props:
                        p = accept_prob(g, c, D, m, st, ds, seed)
                        cache_props[key] = None if p is None else props(p, CX, credit, Qheld_conf, Qtr_conf)
                    pr = cache_props[key]
                    if pr is None: continue
                    v = P["R"][name]; o = v["o"]; a = (o == 0).mean()
                    rule = bm.rule(Qb["conf"], a); ans_rule = ~rule
                    wr_rule_k = (ans_rule & ~Qb["key"]).mean(); wr_rule_a = (ans_rule & ~Qb["aud"]).mean()
                    row = dict(run=name, st=st, ds=ds, g=g, fam=v["fam"], seed=seed, credit=credit, abst=a, right=(o == 1).mean(),
                               wrong=(o == 2).mean(), bad=(o == 3).mean(), aud=v["aud"].mean(),
                               audwrong=((o == 1) | (o == 2)).mean() - v["aud"].mean(),
                               gap_k=(o == 2).mean() - wr_rule_k, gap_a=(((o == 1) | (o == 2)).mean() - v["aud"].mean()) - wr_rule_a)
                    row.update(pr); rows.append(row)
            print(f"\n== {st} {ds}", file=out)
            print(f"{'grader':14s} {'n':>2s} {'A':>5s} {'FPRk':>5s} {'FNRk':>5s} {'FPRa':>5s} {'FNRa':>5s} {'W':>5s} {'Wbst':>5s} {'Pcrs':>5s} {'iFN':>5s} {'iFP':>5s} | {'abst':>5s} {'wrong':>5s} {'aud':>5s} {'gapA':>6s}", file=out)
            byg = defaultdict(list)
            for r in rows:
                if r["st"] == st and r["ds"] == ds: byg[r["g"]].append(r)
            for g in sorted(byg, key=lambda g: np.mean([r["abst"] for r in byg[g]])):
                rr = byg[g]; f = lambda k: np.nanmean([r[k] for r in rr])
                print(f"{g:14s} {len(rr):2d} {f('A'):5.3f} {f('FPRk'):5.3f} {f('FNRk'):5.3f} {f('FPRa'):5.3f} {f('FNRa'):5.3f} {f('W'):5.3f} {f('Wbest'):5.3f} {f('Pcross'):5.3f} {f('iccFNu'):5.2f} {f('iccFPu'):5.2f} | {f('abst'):5.3f} {f('wrong'):5.3f} {f('aud'):5.3f} {f('gap_a'):+6.3f}", file=out)
            out.flush(); print(st, ds, "rows so far", len(rows), flush=True)
    pickle.dump(rows, open(f"{HERE}/j2_table.pkl", "wb"))

if __name__ == "__main__":
    main()
