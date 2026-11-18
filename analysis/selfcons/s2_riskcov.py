"""selfcons s2: risk-coverage of every large-student run against cheap untrained rules AT THE RUN'S OWN COVERAGE.
Split 'new' = never-trained questions (rc.setup HELD), 'home' = the 750 training questions.
For run r answering k of the N split questions (answered = not IDK and not malformed):
  RL      the trained model (audited truth: loose key OR Sonnet >= 4; key-wrong answers with no Sonnet verdict count as WRONG;
          'hi' counts them right, as a bound)
  rules   untrained greedy answers on the k highest-scoring questions (untrained greedy IDK / malformed can never be answered; ties at the
          boundary are split = expectation under random tie-breaking):
          SC = 32-sample agreement with the greedy answer (self-consistency); SCtb = SC with the P(IK) probe as tie-break;
          PIK / PIKA = hidden-state probe of untrained correctness (key / audited labels) fitted on training questions only;
          GF = grader-free probe (>= 31/32 samples agree; no labels); COMBO = logistic on SC features + both probes (training qs);
          ORACLE = best possible with the untrained answers.
  selective accuracy acc = right / answered; wrong100 = wrong answers per 100 split questions. At matched coverage dAcc = dRight / k and
  dWrong100 = -100 dRight / N, so both carry the same sign; both are reported.
  Decomposition of RL - SC: hybrid H = RL's answered set with the UNTRAINED greedy answer where one exists (RL's answer elsewhere):
          selection = H - SC, answers = RL - H.
  cap: run answers more questions than the untrained greedy has valid answers (the rules cannot reach its coverage) -> listed apart.
Question-level 95% intervals: 1,000 multinomial resamples of the split's questions (shared by all runs of a dataset); inside every
resample each rule is re-applied at the run's resampled answer count, so coverage stays matched. Pooled rows resample each dataset
independently and average runs (equal weight per run).
Runs answering < 5% of the split are dropped (selective accuracy undefined in resamples) and listed.
Splits newV / homeV restrict to questions where the untrained greedy gave an answer (no capped runs there).
Output: s2_riskcov_<split>.txt, s2_rows_<split>.pkl.   Usage: python3 s2_riskcov.py new|home|newV|homeV"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, pickle, zlib, numpy as np
from collections import defaultdict
sys.path.insert(0, (_RA + "/selfcons"))
from sc_lib import *
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "new"; B = 1000
RULES = ["SC", "SCtb", "PIK", "PIKA", "GF", "COMBO", "ORACLE"]
BK = ["dSC", "dSCtb", "dPIK", "dPIKA", "dGF", "dCOMBO", "dORACLE", "sel", "ans", "dW_SC", "dSC_key", "dSC_hi"]
lines = []
def pr(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); lines.append(s)
pr(__doc__)
rows = []; BOOT = {}; TINY = []
for st in STS:
    for ds in DSS:
        P, PR = load(st, ds); Q = P["Q"]; M = (Q["ho"] if SPLIT.startswith("new") else Q["tr"]).copy()
        if SPLIT.endswith("V"): M &= Q["gval"]
        N = int(M.sum())
        val = Q["gval"][M]; ru = Q["gaud"][M].astype(float); rk = Q["gkey"][M].astype(float)
        SCO = {k: v[M] for k, v in rule_scores(Q, PR).items()}
        W = boot_weights(N, B, seed=zlib.crc32(f"{st}{ds}{SPLIT}".encode()))
        RB = {rl: RuleBoot(SCO[rl], val, {"a": ru, "k": rk}, W) for rl in RULES}
        for name in sorted(P["R"]):
            r = P["R"][name]; o = r["o"][M]; ans = (o == 1) | (o == 2); k = int(ans.sum())
            if k < 0.05 * N: TINY.append(f"{st}:{ds}:{grp(r['tag'], r['fam'], r['credit'])}"); continue
            aud = r["aud"][M]; hi = aud | r["miss"][M]; key = o == 1
            cap = k > val.sum()
            d = dict(st=st, ds=ds, name=name, tag=r["tag"], fam=r["fam"], credit=r["credit"], g=r["g"], s=r["s"], grp=grp(r["tag"], r["fam"], r["credit"]),
                     N=N, k=k, cov=k / N, idk=float((o == 0).mean()), bad=float((o == 3).mean()), cap=bool(cap), chg=float(r["chg"][M][ans].mean()),
                     miss=float(r["miss"][M][ans].mean()), accRL=float(aud[ans].mean()), accRL_hi=float(hi[ans].mean()), accRL_key=float(key[ans].mean()),
                     w100RL=100 * float((ans & ~aud).sum()) / N)
            for rl in RULES:
                w = frac_select(SCO[rl], val, k); d[f"acc{rl}"] = float((w * ru).sum() / w.sum()); d[f"w100{rl}"] = 100 * float((w * (1 - ru)).sum()) / N
                if rl == "SC": d["accSC_key"] = float((w * rk).sum() / w.sum())
            uans = np.where(val, ru, aud.astype(float))                    # hybrid: untrained answer where it exists
            d["accH"] = float(uans[ans].mean()); d["sel"] = d["accH"] - d["accSC"]; d["ans"] = d["accRL"] - d["accH"]
            for rl in RULES: d[f"d{rl}"] = d["accRL"] - d[f"acc{rl}"]
            d["dW_SC"] = d["w100RL"] - d["w100SC"]; d["dSC_key"] = d["accRL_key"] - d["accSC_key"]; d["dSC_hi"] = d["accRL_hi"] - d["accSC"]
            rows.append(d)
            # bootstrap with re-selection at the resampled coverage
            kb = W @ ans.astype(np.float32); rl_b = W @ (ans & aud).astype(np.float32); hi_b = W @ (ans & hi).astype(np.float32)
            ky_b = W @ (ans & key).astype(np.float32); h_b = W @ (ans * uans).astype(np.float32)
            accRL_b = rl_b / kb
            for rl in RULES:
                rr, nn = RB[rl].right(kb, "a"); BOOT[(st, ds, name, f"d{rl}")] = (accRL_b - rr / nn).astype(np.float32)
                if rl == "SC":
                    BOOT[(st, ds, name, "sel")] = (h_b / kb - rr / nn).astype(np.float32); BOOT[(st, ds, name, "ans")] = ((rl_b - h_b) / kb).astype(np.float32)
                    BOOT[(st, ds, name, "dW_SC")] = (100 * ((kb - rl_b) - (nn - rr)) / N).astype(np.float32)
                    BOOT[(st, ds, name, "dSC_hi")] = (hi_b / kb - rr / nn).astype(np.float32)
                    rk_, nk_ = RB[rl].right(kb, "k"); BOOT[(st, ds, name, "dSC_key")] = (ky_b / kb - rk_ / nk_).astype(np.float32)

def fmt(est, lo, hi, scale=100): return f"{scale*est:+5.1f} [{scale*lo:+5.1f},{scale*hi:+5.1f}]"
def order(g):
    base = g.split(":")[-1].split("[")[0]
    return (g.count(":") + g.count("["), FAMORDER.index(base) if base in FAMORDER else 99, g)

def table(sel_rows, label):
    pr(f"\n{label}")
    pr(f"{'group':22s} {'n':>3s} {'cov':>5s} {'chg':>4s} {'miss':>4s} | {'accRL':>5s} {'accSC':>5s}  {'RL-SC acc pp [95% CI]':22s} {'RL-SC wrong/100 [CI]':22s} | "
       f"{'select':>6s} {'answer':>6s} | {'RL-SCtb':>7s} {'RL-PIK':>6s} {'RL-PIKA':>7s} {'RL-GF':>6s} {'RL-COMBO [CI]':>20s} {'RL-ORAC':>7s} | "
       f"{'RL>SC sig/n':>11s} {'RL<SC sig':>9s} | key-label RL-SC [CI]   hi-bound")
    byg = defaultdict(list)
    for d in sel_rows: byg[d["grp"]].append(d)
    for g in sorted(byg, key=order):
        L = byg[g]; m = lambda k: np.mean([x[k] for x in L])
        bs = {k: np.mean([BOOT[(x["st"], x["ds"], x["name"], k)] for x in L], axis=0) for k in BK}
        sigp = sum(ci(BOOT[(x["st"], x["ds"], x["name"], "dSC")])[0] > 0 for x in L); sign = sum(ci(BOOT[(x["st"], x["ds"], x["name"], "dSC")])[1] < 0 for x in L)
        pr(f"{g:22s} {len(L):3d} {m('cov'):5.2f} {m('chg'):4.2f} {m('miss'):4.2f} | {100*m('accRL'):5.1f} {100*m('accSC'):5.1f}  {fmt(m('dSC'), *ci(bs['dSC'])):22s} "
           f"{fmt(m('dW_SC'), *ci(bs['dW_SC']), scale=1):22s} | {100*m('sel'):+6.1f} {100*m('ans'):+6.1f} | {100*m('dSCtb'):+7.1f} {100*m('dPIK'):+6.1f} {100*m('dPIKA'):+7.1f} "
           f"{100*m('dGF'):+6.1f} {fmt(m('dCOMBO'), *ci(bs['dCOMBO'])):>20s} {100*m('dORACLE'):+7.1f} | {sigp:4d}/{len(L):<3d}    {sign:6d}    | "
           f"{fmt(m('dSC_key'), *ci(bs['dSC_key']))}  {100*m('dSC_hi'):+5.1f}")

for st in STS:
    role = "DISCOVERY" if st == "qwen7" else "CONFIRMATION"
    pr(f"\n\n################ {st} ({role}), split = {SPLIT} ################")
    for ds in DSS:
        R = [d for d in rows if d["st"] == st and d["ds"] == ds and not d["cap"]]
        table(R, f"=== {st} {ds} ({DSNAME[ds]}): selective accuracy (audited) at each run's own coverage; mean over runs of the group")
    R = [d for d in rows if d["st"] == st and not d["cap"]]
    table(R, f"=== {st} POOLED over the three datasets (each run one row; question resamples per dataset)")
    cp = [d for d in rows if d["st"] == st and d["cap"]]
    pr(f"\n--- {st}: capped runs (answer more questions than the untrained greedy has answers; not in the tables): {len(cp)} "
       + ", ".join(sorted(set(f"{d['ds']}:{d['grp']}" for d in cp))))
pr(f"\n--- runs dropped for coverage < 5%: {len(TINY)} " + ", ".join(sorted(set(TINY))))
# ---------------- summary over runs (recipe FK20, IDK credit 0.5, not capped)
pr("\n=== SUMMARY per student: FK20 runs with IDK credit 0.5 (each run one row; per-run 95% question-level CI)")
for st in STS:
    L = [d for d in rows if d["st"] == st and not d["cap"] and d["tag"] == "FK20" and d["credit"] == "idkc05"]
    dd = np.array([d["dSC"] for d in L]); lo = np.array([ci(BOOT[(d["st"], d["ds"], d["name"], "dSC")])[0] for d in L]); hi_ = np.array([ci(BOOT[(d["st"], d["ds"], d["name"], "dSC")])[1] for d in L])
    dc = np.array([d["dCOMBO"] for d in L]); sel_ = np.array([d["sel"] for d in L]); an_ = np.array([d["ans"] for d in L])
    pr(f"{st}: runs {len(L)} | RL-SC acc pp: mean {100*dd.mean():+.2f} median {100*np.median(dd):+.2f} IQR [{100*np.percentile(dd,25):+.1f},{100*np.percentile(dd,75):+.1f}] "
       f"| |RL-SC| <= 2pp: {np.mean(np.abs(dd) <= 0.02):.2f} | sig RL>SC {int((lo > 0).sum())}, sig RL<SC {int((hi_ < 0).sum())} "
       f"| selection {100*sel_.mean():+.2f}, answers {100*an_.mean():+.2f} | RL-COMBO mean {100*dc.mean():+.2f}, runs RL>COMBO {np.mean(dc > 0):.2f}")
    for gname, fams in GROUPS:
        G_ = [d for d in L if d["fam"] in fams]
        if not G_: continue
        bs = np.mean([BOOT[(d["st"], d["ds"], d["name"], "dSC")] for d in G_], axis=0)
        x = np.array([d["dSC"] for d in G_])
        pr(f"   {gname:18s} runs {len(G_):3d}  RL-SC {100*x.mean():+5.2f} [{100*ci(bs)[0]:+5.2f},{100*ci(bs)[1]:+5.2f}]  runs >0: {np.mean(x > 0):.2f}  "
           f"selection {100*np.mean([d['sel'] for d in G_]):+5.2f} answers {100*np.mean([d['ans'] for d in G_]):+5.2f}  miss {np.mean([d['miss'] for d in G_]):.3f}")
pickle.dump(dict(rows=rows, BOOT=BOOT), open(f"{OUT}/s2_rows_{SPLIT}.pkl", "wb"))
open(f"{OUT}/s2_riskcov_{SPLIT}.txt", "w").write("\n".join(lines) + "\n")
