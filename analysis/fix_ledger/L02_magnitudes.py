"""(fix_ledger) L02: the per-100-question outcome account of regimes/r02_costs.py, re-run with COLLAPSED runs removed.
Same definitions, caches and code as regimes/r02_costs.py (its functions are imported unchanged; its output file is not touched):
  per 100 NEW questions: IDK, right (audited), WRONG given, RW = right answers withheld (IDK where the untrained greedy answer was audited
  right), RK = withheld answers the untrained model was sure of (>= 29/32 samples agree), LOC = refusals of audited-right untrained answers
  that the run's own grader rejects, in excess of what a no-location placebo grader's runs do on the same answers (stratified by
  untrained agreement; reference = key-plus-random-noise grader closest in abstention), and LOC vs the other judges' runs at the same rule;
  CHj / IHj = judge-specific paid wrong answers that replaced the untrained answer (own judge pays minus the other judges' mean) per 100 new /
  per 100 TRAINING questions; CHr / IHr = raw count paid by the run's own grader.
Change versus r02: every run whose new-question IDK share is >= 0.95 (all-IDK collapse) is dropped, from the trained arm AND from the
placebo / reference pools; bootstrap draws reduced (B 400 for counts, 300 for LOC). Extra blocks: 'nq' only, so that the Granite judge
(R, NQ-open runs only) and the default-threshold M / G / O runs get a row.
Output: L02_magnitudes_<student>_out.txt.  Usage: nice -n 10 timeout 280 python3 -I L02_magnitudes.py <qwen7|llama8>"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, numpy as np
from collections import defaultdict
sys.path.insert(0, (_RA + "/regimes"))
import rg
import r02_costs as R2
from rg import build, boot_ci, boot_located, CBINS, thr

HERE = (_RA + "/fix_ledger")
st = sys.argv[1]
OPS = R2.OPS + [("R_a35", "JR_a35", "R", "a35")]

def alive(P, r):
    return (r["o"][P["Q"]["held"]] == 0).mean() < 0.95

def fam_runs(P, fam, label):
    return [r for r in R2.__dict__["_orig_fam_runs"](P, fam, label) if alive(P, r)]

R2._orig_fam_runs = R2.fam_runs
R2.fam_runs = fam_runs          # rej_sets() looks this name up in the r02 module

def placebo_runs(P, U, target):
    best = None
    for g in ("sim00fn00", "sim00fn20", "sim00fn40", "sim00fn50", "sim00fn60", "sim00fn70", "sim20fn00", "sim35fn00", "lenQ",
              "sim10fn20", "sim20fn20", "sim25fn20", "sim30fn20", "sim40fn20"):
        rr = [r for r in P["R"].values() if r["g"] == g and r["kind"] == "FK20" and r["credit"] == "idkc05" and alive(P, r)]
        if not rr: continue
        a = np.mean([(r["o"][U] == 0).mean() for r in rr])
        if best is None or abs(a - target) < best[0]: best = (abs(a - target), g, rr, a)
    return best

out = open(f"{HERE}/L02_magnitudes_{st}_out.txt", "w")
def pr(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.write(s + "\n")
pr(__doc__)
def fmt(e, lo, hi): return f"{e:5.1f} [{lo:5.1f},{hi:5.1f}]"
BLOCKS = {"all3": ("h1", "pq", "nq"), "h1+nq": ("h1", "nq"), "nq": ("nq",)}
PS = {ds: build(st, ds) for ds in ("h1", "pq", "nq")}
pr(f"\n===== {st} ({'discovery' if st == 'qwen7' else 'confirmation'}), collapsed runs (new IDK >= 0.95) removed =====")
for ds in ("h1", "pq", "nq"):
    P = PS[ds]; dead = sorted(n for n, r in P["R"].items() if r["kind"] == "FK20" and r["credit"] == "idkc05" and not alive(P, r))
    pr(f"  {ds}: removed {len(dead)} collapsed FK20 credit-0.5 runs: {', '.join(dead) if dead else '-'}")
for blk, dss in BLOCKS.items():
    pr(f"--- block {blk}: per 100 new questions (counts) unless noted; IH per 100 training questions")
    Nn = sum(PS[d]["Q"]["held"].sum() for d in dss)
    ur = 100 * sum((PS[d]["Q"]["aud"] & PS[d]["Q"]["held"] & PS[d]["Q"]["gval"]).sum() for d in dss) / Nn
    uw = 100 * sum((PS[d]["Q"]["held"] & PS[d]["Q"]["gval"] & ~PS[d]["Q"]["aud"]).sum() for d in dss) / Nn
    ui = 100 * sum((PS[d]["Q"]["gidk"] & PS[d]["Q"]["held"]).sum() for d in dss) / Nn
    pr(f"untrained: right {ur:.1f}, wrong {uw:.1f}, IDK {ui:.1f}  (N_new {Nn})")
    pr("op       runs | IDK | right | WRONG given | RW withheld | RK known | LOC vs no-location placebo (known) | LOC vs alternative ref | CHj new | CHr new | IHj train | IHr train")
    for label, fam, J, rule in OPS:
        if label == "keyFN40":
            runsf = lambda P: [r for r in P["R"].values() if r["g"] == "sim00fn40" and r["kind"] == "FK20" and alive(P, r)]
        else:
            runsf = lambda P, fam=fam, label=label: fam_runs(P, fam, label)
        if any(not runsf(PS[d]) for d in dss): continue
        nrun = sum(len(runsf(PS[d])) for d in dss)
        vals = defaultdict(list); vtr = defaultdict(list); dsid = []; dstr = []
        L = {"A": defaultdict(list), "B": defaultdict(list)}; pl_used = []
        for k_, d in enumerate(dss):
            P = PS[d]; Q = P["Q"]; dsn = Q["dsn"]; runs = runsf(P)
            pq = R2.per_question(P, runs, J, rule, st, dsn)
            for kk, v in pq.items(): vals[kk].append(v[Q["held"]]); vtr[kk].append(v[Q["train"]])
            dsid.append(np.full(Q["held"].sum(), k_)); dstr.append(np.full(Q["train"].sum(), k_))
            rs = R2.rej_sets(P, label, J, rule, st, dsn)
            if rs is None: continue
            U, rej, refA, refB, nameB = rs
            yG = np.mean([r["o"][U] == 0 for r in runs], axis=0)
            if refA == "placebo":
                pb = placebo_runs(P, U, yG.mean()); refA = pb[2]; pl_used.append(f"{d}:{pb[1]}({pb[3]:.2f}/{yG.mean():.2f})")
            for key_, rr in (("A", refA), ("B", refB)):
                if not rr: continue
                L[key_]["yG"].append(yG); L[key_]["yR"].append(np.mean([r["o"][U] == 0 for r in rr], axis=0)); L[key_]["rej"].append(rej[U])
                L[key_]["str"].append(k_ * 10 + np.digitize(Q["conf"][U], CBINS)); L[key_]["ds"].append(np.full(U.sum(), k_))
                L[key_]["nG"].append(len(runs)); L[key_]["nR"].append(len(rr))
        dsid = np.concatenate(dsid); dstr = np.concatenate(dstr); Nn_ = len(dsid); Nt = len(dstr)
        V = {k: np.concatenate(v) for k, v in vals.items()}; VT = {k: np.concatenate(v) for k, v in vtr.items()}
        gN = [np.where(dsid == d)[0] for d in np.unique(dsid)]; gT = [np.where(dstr == d)[0] for d in np.unique(dstr)]
        res = {}
        for k in ("idk", "right", "wrong", "RW", "RK", "Hj", "Hr"):
            res[k] = boot_ci(lambda ix, k=k: 100 * np.nanmean(V[k][ix]), Nn_, B=400, groups=gN)
        for k in ("Hj", "Hr"):
            res["T" + k] = boot_ci(lambda ix, k=k: 100 * np.nanmean(VT[k][ix]), Nt, B=400, groups=gT)
        loc = {}
        for key_ in ("A", "B"):
            D = L[key_]
            if len(D["yG"]) < len(dss): loc[key_] = None; continue
            c = {k: np.concatenate(v) for k, v in D.items() if k not in ("nG", "nR")}
            loc[key_] = boot_located(c["yG"], c["yR"], int(np.mean(D["nG"])), int(np.mean(D["nR"])), c["rej"], c["str"], c["ds"], Nn_, B=300)
        if loc["A"]:
            e, ci = loc["A"]; locs = f"{fmt(e['excess100'], *ci['excess100'])} known {fmt(e['excess100_known'], *ci['excess100_known'])} (rej {e['n_rej100']:.1f}/100)"
        else: locs = "      n/a"
        if loc["B"]:
            e, ci = loc["B"]; locb = f"{fmt(e['excess100'], *ci['excess100'])} vs {nameB}"
        else: locb = "n/a"
        if rule in ("key", "fn40"): chj = thj = "   0 by construction  "; chr_ = "0"
        elif rule == "son": chj = "  n/a (audit=Sonnet)  "; thj = "  n/a                "; chr_ = "0*"
        else: chj = fmt(*res['Hj']); thj = fmt(*res['THj']); chr_ = f"{res['Hr'][0]:4.1f}"
        pr(f"{label:8s} {nrun:3d} | {res['idk'][0]:4.1f} | {res['right'][0]:4.1f} | {fmt(*res['wrong'])} | {fmt(*res['RW'])} | {fmt(*res['RK'])} | LOC {locs} | {locb} | "
           f"{chj} | {chr_} | {thj} | {fmt(*res['THr'])}" + (f"  placebo: {' '.join(pl_used)}" if pl_used else ""))
out.close()
