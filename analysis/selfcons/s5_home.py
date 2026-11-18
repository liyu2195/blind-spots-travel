"""selfcons s5: does RL add value on its TRAINING questions (home) that it fails to carry to NEW questions?
Pairs each run's s2 results on home (750 training questions) and new questions: RL - SC selective accuracy at the run's own coverage on each
split, split into selection (RL's answered set with the untrained answers) and answers (RL's changed answers). Truth = audited
(loose key OR Sonnet >= 4; home verdicts come from frontierA and the Sonnet-reward runs' caches, so Sonnet-trained runs are partly graded
in-sample at home). 95% intervals: question resamples of each split (independent sets, so home - new intervals combine replicates).
Also: across runs, does a run's home selection gain predict its new-question selection gain (Spearman, within dataset)?
Output: s5_home.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, pickle, numpy as np
from collections import defaultdict
from scipy.stats import spearmanr
sys.path.insert(0, (_RA + "/selfcons"))
from sc_lib import *
lines = []
def pr(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); lines.append(s)
pr(__doc__)
H = pickle.load(open(f"{OUT}/s2_rows_home.pkl", "rb")); NW = pickle.load(open(f"{OUT}/s2_rows_new.pkl", "rb"))
hr = {(d["st"], d["ds"], d["name"]): d for d in H["rows"] if not d["cap"]}; nr = {(d["st"], d["ds"], d["name"]): d for d in NW["rows"] if not d["cap"]}
keys = sorted(set(hr) & set(nr))
def fmt(v, b): lo, hi = ci(b); return f"{100*v:+5.1f} [{100*lo:+5.1f},{100*hi:+5.1f}]"
def block(L, label):
    pr(f"\n{label}")
    pr(f"{'group':20s} {'n':>3s} {'covH':>5s} {'covN':>5s} | {'home RL-SC [CI]':>21s} {'sel':>5s} {'ans':>5s} | {'new RL-SC [CI]':>21s} {'sel':>5s} {'ans':>5s} | "
       f"{'home-new [CI]':>21s} {'sel h-n [CI]':>21s} | runs home>0 new>0")
    byg = defaultdict(list)
    for k in L: byg[hr[k]["grp"]].append(k)
    def order(g):
        base = g.split(":")[-1].split("[")[0]
        return (g.count(":") + g.count("["), FAMORDER.index(base) if base in FAMORDER else 99, g)
    for g in sorted(byg, key=order):
        X = byg[g]; mh = lambda f: np.mean([hr[k][f] for k in X]); mn = lambda f: np.mean([nr[k][f] for k in X])
        bh = np.mean([H["BOOT"][k + ("dSC",)] for k in X], 0); bn = np.mean([NW["BOOT"][k + ("dSC",)] for k in X], 0)
        sh = np.mean([H["BOOT"][k + ("sel",)] for k in X], 0); sn = np.mean([NW["BOOT"][k + ("sel",)] for k in X], 0)
        pr(f"{g:20s} {len(X):3d} {mh('cov'):5.2f} {mn('cov'):5.2f} | {fmt(mh('dSC'), bh):>21s} {100*mh('sel'):+5.1f} {100*mh('ans'):+5.1f} | {fmt(mn('dSC'), bn):>21s} "
           f"{100*mn('sel'):+5.1f} {100*mn('ans'):+5.1f} | {fmt(mh('dSC') - mn('dSC'), bh - bn):>21s} {fmt(mh('sel') - mn('sel'), sh - sn):>21s} | "
           f"{np.mean([hr[k]['dSC'] > 0 for k in X]):.2f} {np.mean([nr[k]['dSC'] > 0 for k in X]):.2f}")
for st in STS:
    pr(f"\n################ {st} ({'DISCOVERY' if st == 'qwen7' else 'CONFIRMATION'}) ################")
    L = [k for k in keys if k[0] == st]
    block(L, f"=== {st}: pooled over datasets, every run with both splits (not capped)")
    F = [k for k in L if hr[k]["tag"] == "FK20" and hr[k]["credit"] == "idkc05"]
    pr(f"\n--- {st} FK20 idkc05 runs by grader group")
    for gname, fams in GROUPS:
        X = [k for k in F if hr[k]["fam"] in fams]
        if not X: continue
        bh = np.mean([H["BOOT"][k + ("dSC",)] for k in X], 0); bn = np.mean([NW["BOOT"][k + ("dSC",)] for k in X], 0)
        sh = np.mean([H["BOOT"][k + ("sel",)] for k in X], 0); sn = np.mean([NW["BOOT"][k + ("sel",)] for k in X], 0)
        ah = np.mean([H["BOOT"][k + ("ans",)] for k in X], 0); an = np.mean([NW["BOOT"][k + ("ans",)] for k in X], 0)
        m = lambda D, f: np.mean([D[k][f] for k in X])
        pr(f"   {gname:18s} runs {len(X):3d} | home {fmt(m(hr,'dSC'), bh)} = sel {fmt(m(hr,'sel'), sh)} + ans {fmt(m(hr,'ans'), ah)} | "
           f"new {fmt(m(nr,'dSC'), bn)} = sel {fmt(m(nr,'sel'), sn)} + ans {fmt(m(nr,'ans'), an)}")
    # does home selection gain predict new selection gain? (within dataset Spearman, then mean)
    for f in ("sel", "dSC", "ans"):
        rs = []
        for ds in DSS:
            X = [k for k in F if k[1] == ds]
            if len(X) > 10: rs.append((spearmanr([hr[k][f] for k in X], [nr[k][f] for k in X])[0], len(X)))
        pr(f"   across runs, Spearman(home {f}, new {f}) within dataset: " + ", ".join(f"{r:+.2f} (n {n})" for r, n in rs))
    # coverage: do runs answer more at home?
    pr(f"   coverage home - new (mean over FK20 runs): {np.mean([hr[k]['cov'] - nr[k]['cov'] for k in F]):+.3f}; changed-answer share home {np.mean([hr[k]['chg'] for k in F]):.3f} new {np.mean([nr[k]['chg'] for k in F]):.3f}")
open(f"{OUT}/s5_home.txt", "w").write("\n".join(lines) + "\n")
