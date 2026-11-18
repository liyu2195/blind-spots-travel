"""selfcons s3: WHICH new questions RL refuses compared with the self-consistency rule at the same coverage, and how much of the
difference follows the grader's own verdict map.
Per run (new questions, not capped, coverage >= 5%): R = refused (IDK or malformed); SC answer weights w at the run's coverage (ties split).
  relocated = RL-only refusals (RL refuses, SC answers) = sum R*w = SC-only refusals; per 100 new questions and as share of RL's refusals.
  random    = expected relocated count if RL's refusals were a random set of the same size.
  seeds     = relocation against the leave-one-out consensus of the OTHER seeds of the same recipe (refuse the questions most often
              refused by them; ties -> lower SC first), same coverage: how much relocation is reproducible.
  verdict map V(q) of the run's grader on the UNTRAINED greedy answer of new question q (1 = the grader would pay it):
     key / key+noise / blind / scrambled / hidden-half / typeFP / popFP / frzOR / peer / jveto: strict key (noise is random, so the key
     is the map); keyLong: key OR answer >= 5 words; vote: greedy = plurality of the 32 samples; judge J at op: J accepts at that
     threshold (f1 / quorum variants: Qwen t50); Sonnet: level >= 4; fnB*: key AND relation type outside the planted set (fnA*: inside);
     plreal<J>: key AND judge J (a35 target) accepts; plsharp<J>: key AND type outside the planted type-wide set.
     Planted sets are read from data/pqa_knows_*.jsonl (training ids) and carried to new questions by relation type.
  p1 = share of RL-only refusals that the grader rejects (V = 0); E1 = the same expected from SC-answered questions of equal SC bins.
  p2 = share of SC-only refusals (valid greedy) that the grader rejects; E2 its conf-bin expectation.  map excess = ((p1-E1)+(E2-p2))/2.
  mapRule(l) = SC with the map added (score conf + l*V): share of SC's relocation it removes = 1 - reloc(RL, mapRule)/reloc(RL, SC).
  probeRule = ranking by the hidden-state probe of the grader's verdict on training questions (pay_<grader>); same share.
  accMap - accSC = selective accuracy (audited, untrained answers) of mapRule(1) minus SC at the run's coverage (what following the map buys).
Output: s3_overlap.txt, s3_rows.pkl"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, json, re, pickle, numpy as np
from collections import defaultdict
sys.path.insert(0, (_RA + "/selfcons"))
from sc_lib import *
lines = []
def pr(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); lines.append(s)
pr(__doc__)
from sc_map import vmap, payprobe, CB

rows = []
for st in STS:
    for ds in DSS:
        P, PR = load(st, ds); Q = P["Q"]; M = Q["ho"]; N = int(M.sum())
        val = Q["gval"][M]; conf = Q["conf"][M]; ru = Q["gaud"][M].astype(float); cb = np.digitize(conf, CB)
        runs = {n: r for n, r in P["R"].items()}
        recipe = defaultdict(list)
        for n, r in runs.items(): recipe[(r["tag"], r["g"], r["credit"])].append(n)
        for name, r in sorted(runs.items()):
            o = r["o"][M]; R = ((o == 0) | (o == 3)).astype(float); k = int(N - R.sum())
            if k < 0.05 * N or k > val.sum(): continue
            w = frac_select(conf, val, k); a = float((R * w).sum()); nref = R.sum()
            V, vname = vmap(r, Q, st, ds); V = V[M].astype(float) * val
            d = dict(st=st, ds=ds, name=name, tag=r["tag"], fam=r["fam"], credit=r["credit"], grp=grp(r["tag"], r["fam"], r["credit"]), vmap=vname,
                     N=N, cov=k / N, reloc100=100 * a / N, reloc_share=a / nref, rand_share=k / N)
            # seeds consensus (leave one out)
            others = [n for n in recipe[(r["tag"], r["g"], r["credit"])] if n != name]
            if others:
                c = np.mean([((runs[n]["o"][M] == 0) | (runs[n]["o"][M] == 3)) for n in others], axis=0)
                wc = frac_select(-c + 1e-3 * conf, np.ones(N, bool), k); d["seed_share"] = float((R * wc).sum()) / nref; d["nseed"] = len(others)
            else: d["seed_share"] = np.nan; d["nseed"] = 0
            # composition of the relocated sets
            rl_only = R * w; sc_only_v = (1 - R) * (1 - w) * val; sc_only_i = (1 - R) * (1 - val)
            d["p1"] = float((rl_only * (1 - V)).sum() / max(a, 1e-9)); d["e1"] = float((rl_only * (1 - ru)).sum() / max(a, 1e-9))
            d["conf1"] = float((rl_only * conf).sum() / max(a, 1e-9))
            bv = sc_only_v.sum(); d["sc_only_valid_share"] = float(bv / max(a, 1e-9))
            d["p2"] = float((sc_only_v * (1 - V)).sum() / max(bv, 1e-9)); d["e2"] = float((sc_only_v * (1 - ru)).sum() / max(bv, 1e-9))
            d["conf2"] = float((sc_only_v * conf).sum() / max(bv, 1e-9))
            aud = r["aud"][M]; ws_ = (1 - R) * (1 - w)
            d["accRL_on_sc_only"] = float((ws_ * aud).sum() / ws_.sum()) if ws_.sum() > 0 else np.nan
            # conf-bin expectations
            E1 = 0.0; E2 = 0.0
            for b in np.unique(cb):
                mb = cb == b; ww = w[mb].sum(); vv = ((1 - w) * val)[mb].sum()
                v0a = (w * (1 - V))[mb].sum() / ww if ww > 0 else 0.0; v0r = ((1 - w) * val * (1 - V))[mb].sum() / vv if vv > 0 else 0.0
                E1 += rl_only[mb].sum() * v0a; E2 += sc_only_v[mb].sum() * v0r
            d["E1"] = E1 / max(a, 1e-9); d["E2"] = E2 / max(bv, 1e-9); d["mapx"] = ((d["p1"] - d["E1"]) + (d["E2"] - d["p2"])) / 2
            d["v0_rate"] = float(1 - V[val > 0].mean())
            # map rules and probe rules
            accSC = float((w * ru).sum() / w.sum())
            for lam in (0.25, 1.0):
                wg = frac_select(conf + lam * V, val, k); ag = float((R * wg).sum())
                d[f"map{lam}_removed"] = 1 - ag / max(a, 1e-9); d[f"map{lam}_dacc"] = float((wg * ru).sum() / wg.sum()) - accSC
            for nm, sc in (("pay", payprobe(r["fam"], PR)[M]), ("pik", PR["S"]["pik"][M]), ("combo", PR["S"]["combo"][M])):
                wp = frac_select(sc, val, k); d[f"{nm}_removed"] = 1 - float((R * wp).sum()) / max(a, 1e-9)
            rows.append(d)

def show(L, label):
    pr(f"\n{label}")
    pr(f"{'group':20s} {'n':>3s} {'map':>13s} {'cov':>5s} | {'reloc/100':>9s} {'share':>5s} {'rand':>5s} {'seedLOO':>7s} | RL-only refusals: {'conf':>4s} {'V=0':>4s} {'exp':>4s} {'wrongU':>6s} | "
       f"SC-only (RL answers): {'conf':>4s} {'V=0':>4s} {'exp':>4s} {'wrongU':>6s} {'accRL':>5s} {'IDKu%':>5s} | {'mapX':>5s} | removed: {'map.25':>6s} {'map1':>5s} {'pay':>5s} {'pik':>5s} {'combo':>5s} | {'dAccMap1':>8s}")
    byg = defaultdict(list)
    for d in L: byg[d["grp"]].append(d)
    def order(g):
        base = g.split(":")[-1].split("[")[0]
        return (g.count(":") + g.count("["), FAMORDER.index(base) if base in FAMORDER else 99, g)
    for g in sorted(byg, key=order):
        X = byg[g]; m = lambda k: np.nanmean([x[k] for x in X])
        vm = sorted(set(x["vmap"] for x in X)); vm = vm[0] if len(vm) == 1 else "mixed"
        pr(f"{g:20s} {len(X):3d} {vm[:13]:>13s} {m('cov'):5.2f} | {m('reloc100'):9.1f} {m('reloc_share'):5.2f} {m('rand_share'):5.2f} {m('seed_share'):7.2f} | "
           f"                  {m('conf1'):4.2f} {m('p1'):4.2f} {m('E1'):4.2f} {m('e1'):6.2f} | "
           f"                      {m('conf2'):4.2f} {m('p2'):4.2f} {m('E2'):4.2f} {m('e2'):6.2f} {m('accRL_on_sc_only'):5.2f} {100*(1-m('sc_only_valid_share')):5.0f} | {m('mapx'):+5.2f} | "
           f"         {m('map0.25_removed'):+6.2f} {m('map1.0_removed'):+5.2f} {m('pay_removed'):+5.2f} {m('pik_removed'):+5.2f} {m('combo_removed'):+5.2f} | {100*m('map1.0_dacc'):+8.1f}")

for st in STS:
    role = "DISCOVERY" if st == "qwen7" else "CONFIRMATION"
    pr(f"\n\n################ {st} ({role}) ################")
    for ds in DSS: show([d for d in rows if d["st"] == st and d["ds"] == ds], f"=== {st} {ds} ({DSNAME[ds]}) new questions; mean over runs")
    show([d for d in rows if d["st"] == st], f"=== {st} POOLED over datasets")
pr("\n=== SUMMARY: FK20 idkc05 runs, by grader group (mean over runs)")
for st in STS:
    L = [d for d in rows if d["st"] == st and d["tag"] == "FK20" and d["credit"] == "idkc05"]
    pr(f"{st}: runs {len(L)}  relocated share {np.mean([d['reloc_share'] for d in L]):.2f} (random {np.mean([d['rand_share'] for d in L]):.2f}, "
       f"vs seed consensus {np.nanmean([d['seed_share'] for d in L]):.2f})  reloc/100 {np.mean([d['reloc100'] for d in L]):.1f}")
    for gname, fams in GROUPS:
        G_ = [d for d in L if d["fam"] in fams]
        if not G_: continue
        m = lambda k: np.nanmean([x[k] for x in G_])
        pr(f"   {gname:18s} runs {len(G_):3d} reloc/100 {m('reloc100'):4.1f} share {m('reloc_share'):.2f} seedLOO {m('seed_share'):.2f} | RL-only V=0 {m('p1'):.2f} (exp {m('E1'):.2f}) "
           f"SC-only V=0 {m('p2'):.2f} (exp {m('E2'):.2f}) mapX {m('mapx'):+.2f} | removed by map(.25) {m('map0.25_removed'):+.2f} map(1) {m('map1.0_removed'):+.2f} "
           f"pay-probe {m('pay_removed'):+.2f} pik {m('pik_removed'):+.2f} combo {m('combo_removed'):+.2f} | dAcc map(1) {100*m('map1.0_dacc'):+.1f}")
pickle.dump(rows, open(f"{OUT}/s3_rows.pkl", "wb"))
open(f"{OUT}/s3_overlap.txt", "w").write("\n".join(lines) + "\n")
