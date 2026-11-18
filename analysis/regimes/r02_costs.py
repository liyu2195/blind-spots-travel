"""r02: full cost account per operating point (task 1). Counts per 100 questions, question-level bootstrap (within dataset) 95% CIs.
New questions (held-out):
  idk, right (audited), wrong (audited wrong) given;
  RW  = right answers withheld: IDK where the untrained greedy answer was audited right;
  RK  = refusals of known answers: RW with untrained agreement >= 0.9 (29+ of 32 samples give the greedy answer);
  LOC = grader-located refusals: refusals of audited-right untrained answers that the run's grader rejects, in excess of the
        run's own refusal of conf-matched accepted ones shifted by the reference runs' log odds ratio on the same sets
        (judges & Sonnet: universe = answers the training key accepts, reference = key-trained runs;
         key: universe = answers Sonnet accepts, reference = Sonnet-trained runs);
  CHj = content hacks, judge-specific: changed (not the untrained greedy), audited-wrong answers x (own grader pays minus the mean
        of the other judges Q/M/G/O at the same operating rule); CHr = raw count paid by own grader.
Training questions: IHj / IHr = the same for installed hacks.
Key: own pay = strict key, which pays no audited-wrong answer by construction; Sonnet: on new questions own pay = Sonnet audit
(0 by construction); on training questions own pay = the run's own reward verdicts (api_verdicts), audit = frontierA Sonnet.
Output: r02_costs.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, numpy as np
sys.path.insert(0, (_RA + "/regimes"))
from rg import *

OPS = [  # label, family, judge, rule
    ("Q_t50", "JQ_t50", "Q", "t50"), ("Q_a50", "JQ_a50", "Q", "a50"), ("Q_a35", "JQ_a35", "Q", "a35"),
    ("M_a35", "JM_a35", "M", "a35"), ("G_a35", "JG_a35", "G", "a35"), ("O_a35", "JO_a35", "O", "a35"),
    ("M_t50", "JM_t50", "M", "t50"), ("G_t50", "JG_t50", "G", "t50"), ("O_t50", "JO_t50", "O", "t50"),
    ("sonnet", "sonnet", None, "son"), ("key", "key", None, "key"), ("lenQ", "lenQ", "Q", "t50"), ("keyFN40", None, None, "fn40")]
OTHER = ["Q", "M", "G", "O"]

def fam_runs(P, fam, label):
    if label == "keyFN40": return [r for r in P["R"].values() if r["g"] == "sim00fn40" and r["kind"] == "FK20"]
    return [r for r in P["R"].values() if r["fam"] == fam and r["kind"] == "FK20" and r["credit"] == "idkc05"]

def base_pay(P, K, rule, st, dsn):
    """pay rate of judge K (at rule) on the untrained model's audited-wrong greedy answers, per split (array over questions)."""
    Q = P["Q"]; w = Q["gval"] & ~Q["aud"] & np.isfinite(Q["pj"][K]); acc = Q["pj"][K] > thr(K, st, dsn, rule); b = np.zeros(Q["n"])
    for m in (Q["held"], Q["train"]): b[m] = acc[w & m].mean()
    return b

def per_question(P, runs, J, rule, st, dsn):
    """per-question means over runs of each indicator (all 1500 questions).
    Hj = changed audited-wrong answer x [(own pays - own base rate) - mean over other judges K of (K pays - K base rate)];
    base rate = the judge's pay rate on the untrained model's audited-wrong greedy answers of the same split."""
    Q = P["Q"]; aud0 = Q["aud"] & Q["gval"]; out = defaultdict(list)
    if J is not None and rule in ("t50", "a50", "a35"):
        bJ = base_pay(P, J, rule, st, dsn); bK = {K: base_pay(P, K, rule, st, dsn) for K in OTHER if K != J}
    for r in runs:
        o = r["o"]; idk = o == 0; wrong = (o == 2) & ~r["aud"]; right = r["aud"]
        if J is not None and rule in ("t50", "a50", "a35"):
            own = np.nan_to_num(r["pj"][J], nan=0.0) > thr(J, st, dsn, rule)
            oth = [np.where(np.isfinite(r["pj"][K]), (r["pj"][K] > thr(K, st, dsn, rule)) - bK[K], np.nan) for K in OTHER if K != J]
            othm = np.nanmean(np.array(oth, float), axis=0); othm = np.nan_to_num(othm, nan=0.0) + bJ
        elif rule == "son":   # new questions: own pay = Sonnet audit (0 by construction); training: the run's reward verdicts
            own = np.where(Q["held"], r["son"] >= 4, r["sapi"] >= 4); othm = np.full(len(o), np.nan)
        else:  # key and random-noise key graders: pay = strict key
            own = r["kst"].copy(); othm = np.zeros(len(o))
        hk = wrong & r["chg"]
        out["idk"].append(idk); out["right"].append(right); out["wrong"].append(wrong); out["bad"].append(o == 3)
        out["RW"].append(idk & aud0); out["RK"].append(idk & aud0 & (Q["conf"] >= 0.9))
        out["Hr"].append(hk & own); out["Hj"].append(np.where(hk, own.astype(float) - othm, 0.0)); out["Hn"].append(hk)
    return {k: np.mean(np.array(v, float), axis=0) for k, v in out.items()}

def rej_sets(P, label, J, rule, st, dsn):
    """universe U (new questions, audited-right untrained answer), rejection labels, and two references:
    A = no-location placebo (judges: random-rejection key grader closest in abstention; key: Sonnet runs; Sonnet: key runs),
    B = alternative (judges: other judges' runs at the same rule; key: Qwen judge a50 runs; Sonnet: Qwen judge t50 runs)."""
    Q = P["Q"]; base = Q["held"] & Q["gval"] & Q["aud"]
    if J is not None and rule in ("t50", "a50", "a35"):
        U = base & Q["kstrict"] & np.isfinite(Q["pj"][J]); rej = Q["pj"][J] <= thr(J, st, dsn, rule)
        B = [r for K in ("Q", "M", "G", "O") if K != J for r in fam_runs(P, f"J{K}_{rule}", "")] if label != "lenQ" else []
        return U, rej, "placebo", B, "other judges"
    if rule == "son":
        U = base & Q["kstrict"] & (Q["son"] >= 0); rej = Q["son"] < 4
        return U, rej, fam_runs(P, "key", "key"), fam_runs(P, "JQ_t50", ""), "Q_t50"
    if rule == "key":
        U = base & (Q["son"] >= 4); rej = ~Q["kstrict"]
        return U, rej, fam_runs(P, "sonnet", "sonnet"), fam_runs(P, "JQ_a50", ""), "Q_a50"
    return None

if __name__ == "__main__":
    out = open(f"{HERE}/r02_costs.txt", "w")
    def pr(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); out.write(s + "\n")

    def fmt(e, lo, hi): return f"{e:5.1f} [{lo:5.1f},{hi:5.1f}]"

    BLOCKS = {"all3": ("h1", "pq", "nq"), "h1+nq": ("h1", "nq")}
    for st in ("qwen7", "llama8"):
        PS = {ds: build(st, ds) for ds in ("h1", "pq", "nq")}
        pr(f"\n===== {st} ({'discovery' if st == 'qwen7' else 'confirmation'}) =====")
        # untrained baseline
        for blk, dss in BLOCKS.items():
            pr(f"--- block {blk}: per 100 new questions (counts) unless noted; IH per 100 training questions")
            Nn = sum(PS[d]["Q"]["held"].sum() for d in dss)
            ub = {k: 100 * sum(((PS[d]["Q"][m] if m != "gidk" else PS[d]["Q"]["gidk"]) & PS[d]["Q"]["held"] & (PS[d]["Q"]["gval"] if m != "gidk" else True)).sum() for d in dss) / Nn
                  for k, m in (("untrained_right", "aud"), ("untrained_idk", "gidk"))}
            uw = 100 * sum((PS[d]["Q"]["held"] & PS[d]["Q"]["gval"] & ~PS[d]["Q"]["aud"]).sum() for d in dss) / Nn
            pr(f"untrained: right {ub['untrained_right']:.1f}, wrong {uw:.1f}, IDK {ub['untrained_idk']:.1f}  (N_new {Nn})")
            pr("op       runs | IDK | right | WRONG given | RW withheld | RK known | LOC located vs no-location placebo (of which on known answers) | LOC vs alternative ref | CHj new | CHr new | IHj train | IHr train")
            for label, fam, J, rule in OPS:
                have = [d for d in dss if fam_runs(PS[d], fam, label)]
                if len(have) < len(dss): continue
                nrun = sum(len(fam_runs(PS[d], fam, label)) for d in dss)
                # per-question arrays (new and train) concatenated over datasets
                vals = defaultdict(list); vtr = defaultdict(list); dsid = []; dstr = []
                L = {"A": defaultdict(list), "B": defaultdict(list)}; pl_used = []
                for k_, d in enumerate(dss):
                    P = PS[d]; Q = P["Q"]; dsn = Q["dsn"]; runs = fam_runs(P, fam, label)
                    pq = per_question(P, runs, J, rule, st, dsn)
                    for kk, v in pq.items(): vals[kk].append(v[Q["held"]]); vtr[kk].append(v[Q["train"]])
                    dsid.append(np.full(Q["held"].sum(), k_)); dstr.append(np.full(Q["train"].sum(), k_))
                    rs = rej_sets(P, label, J, rule, st, dsn)
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
                dsid = np.concatenate(dsid); dstr = np.concatenate(dstr); Nn = len(dsid); Nt = len(dstr)
                V = {k: np.concatenate(v) for k, v in vals.items()}; VT = {k: np.concatenate(v) for k, v in vtr.items()}
                gN = [np.where(dsid == d)[0] for d in np.unique(dsid)]; gT = [np.where(dstr == d)[0] for d in np.unique(dstr)]
                res = {}
                for k in ("idk", "right", "wrong", "RW", "RK", "Hj", "Hr"):
                    res[k] = boot_ci(lambda ix, k=k: 100 * np.nanmean(V[k][ix]), Nn, B=1000, groups=gN)
                for k in ("Hj", "Hr"):
                    res["T" + k] = boot_ci(lambda ix, k=k: 100 * np.nanmean(VT[k][ix]), Nt, B=1000, groups=gT)
                loc = {}
                for key_ in ("A", "B"):
                    D = L[key_]
                    if len(D["yG"]) < len(dss): loc[key_] = None; continue
                    c = {k: np.concatenate(v) for k, v in D.items() if k not in ("nG", "nR")}
                    loc[key_] = boot_located(c["yG"], c["yR"], int(np.mean(D["nG"])), int(np.mean(D["nR"])), c["rej"], c["str"], c["ds"], Nn, B=500)
                if loc["A"]:
                    e, ci = loc["A"]
                    locs = f"{fmt(e['excess100'], *ci['excess100'])} known {fmt(e['excess100_known'], *ci['excess100_known'])} (rej {e['n_rej100']:.1f}/100)"
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
