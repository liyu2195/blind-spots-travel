"""r03b: do natural (non-thresholded) graders' question-level false rejections travel to new questions?
Settings with runs: judges at P(Yes) > 0.5 (Q all sets; G, O on TriviaQA + NQ-open; M TriviaQA), Sonnet as reward, the answer key's
missing aliases; strict cuts (a35) for comparison. For each: located refusal (rg.located_stats) of the run's grader's rejections of
audited-right untrained answers, conf x dataset strata, against a no-location reference:
  judges / Sonnet: universe = answers the training key accepts; reference = key-plus-random-noise grader closest in abstention;
  key (missing aliases): universe = answers Sonnet accepts and the key either accepts strictly or misses entirely;
       reference = Sonnet-trained runs (and Qwen judge a50 runs as a second reference).
HOME = the same on training questions (audit there: key or frontierA Sonnet); NEW = held-out questions.
Placebos: (1) label permutation within strata (p); (2) a no-location grader's runs analysed as if trained on the setting's grader
('placebo-as-G'; the second-closest no-location grader, so it differs from the reference).
Output: r03b_travel.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, numpy as np
sys.path.insert(0, (_RA + "/regimes"))
from rg import *

SETTINGS = [("Q_t50", "JQ_t50", "Q", "t50"), ("G_t50", "JG_t50", "G", "t50"), ("O_t50", "JO_t50", "O", "t50"), ("M_t50", "JM_t50", "M", "t50"),
            ("sonnet", "sonnet", None, "son"), ("key_alias", "key", None, "alias"),
            ("Q_a35", "JQ_a35", "Q", "a35"), ("G_a35", "JG_a35", "G", "a35"), ("O_a35", "JO_a35", "O", "a35")]
NOLOC = ("sim00fn00", "sim00fn20", "sim00fn40", "sim00fn50", "sim00fn60", "sim00fn70", "sim20fn00", "sim35fn00", "lenQ",
         "sim10fn20", "sim20fn20", "sim25fn20", "sim30fn20", "sim40fn20")
out = open(f"{HERE}/r03b_travel.txt", "w")
def pr(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.write(s + "\n")

def fam_runs(P, fam):
    return [r for r in P["R"].values() if r["fam"] == fam and r["kind"] == "FK20" and r["credit"] == "idkc05"]

def noloc_ranked(P, U, target):
    c = []
    for g in NOLOC:
        rr = [r for r in P["R"].values() if r["g"] == g and r["kind"] == "FK20" and r["credit"] == "idkc05"]
        if rr: c.append((abs(np.mean([(r["o"][U] == 0).mean() for r in rr]) - target), g, rr))
    return sorted(c, key=lambda t: t[0])

def universe(Q, J, op, st, dsn, split):
    m = Q["held"] if split == "new" else Q["train"]; base = m & Q["gval"] & Q["aud"]
    if J is not None:
        return base & Q["kstrict"] & np.isfinite(Q["pj"][J]), Q["pj"][J] <= thr(J, st, dsn, op)
    if op == "son":
        return base & Q["kstrict"] & (Q["son"] >= 0), Q["son"] < 4
    U = m & Q["gval"] & (Q["son"] >= 4) & (Q["kstrict"] | ~Q["key"]); return U, ~Q["key"]

def collect(PS, dss, fam, J, op, st, split, mode):
    """mode: 'G' = setting's runs vs reference; 'placebo' = second-closest no-location runs vs reference; 'alt' = key vs Q_a50."""
    L = defaultdict(list); Ntot = 0; refs = []
    for k_, d in enumerate(dss):
        P = PS[d]; Q = P["Q"]; dsn = Q["dsn"]; U, rej = universe(Q, J, op, st, dsn, split)
        Ntot += (Q["held"] if split == "new" else Q["train"]).sum()
        runs = fam_runs(P, fam)
        if not runs: return None
        yG0 = np.mean([r["o"][U] == 0 for r in runs], axis=0)
        if op == "alias":
            rr = fam_runs(P, "sonnet") if mode != "alt" else fam_runs(P, "JQ_a50"); refs.append(f"{d}:{'sonnet' if mode != 'alt' else 'Q_a50'}")
            if mode == "placebo": runs = fam_runs(P, "JQ_t50")   # a reference-free judge that never saw the key
        else:
            rk = noloc_ranked(P, U, yG0.mean()); rr = rk[0][2]; refs.append(f"{d}:{rk[0][1]}")
            if mode == "placebo": runs = rk[1][2]; refs[-1] += f"/as-G {rk[1][1]}"
        L["yG"].append(np.mean([r["o"][U] == 0 for r in runs], axis=0)); L["yR"].append(np.mean([r["o"][U] == 0 for r in rr], axis=0))
        L["rej"].append(rej[U]); L["str"].append(k_ * 10 + np.digitize(Q["conf"][U], CBINS)); L["ds"].append(np.full(U.sum(), k_))
        L["nG"].append(len(runs)); L["nR"].append(len(rr))
    c = {k: np.concatenate(v) for k, v in L.items() if k not in ("nG", "nR")}
    return c, int(np.mean(L["nG"])), int(np.mean(L["nR"])), Ntot, refs

def perm_p(c, nG, nR, Ntot, est, B=200, seed=1):
    rng = np.random.default_rng(seed); null = []
    for _ in range(B):
        rp = c["rej"].copy()
        for s in np.unique(c["str"]):
            m = np.where(c["str"] == s)[0]; rp[m] = rng.permutation(c["rej"][m])
        null.append(located_stats(c["yG"], c["yR"], nG, nR, rp, c["str"], np.arange(len(rp)), Ntot)["dlor"])
    null = np.array(null); return (np.sum(null >= est) + 1) / (B + 1)

pr("cols: setting datasets split mode | rej/100q | refusal rate rej / acc (G runs; ref runs) | excess refusals per 100q [95% CI] | dlogOR [95% CI] | perm p | refs")
for st in ("qwen7", "llama8"):
    PS = {ds: build(st, ds) for ds in ("h1", "pq", "nq")}
    pr(f"\n===== {st} ({'discovery' if st == 'qwen7' else 'confirmation'}) =====")
    for name, fam, J, op in SETTINGS:
        avail = [d for d in ("h1", "pq", "nq") if fam_runs(PS[d], fam)]
        groups = [(d,) for d in avail] + ([tuple(avail)] if len(avail) > 1 else [])
        for dss in groups:
            for split in ("home", "new"):
                for mode in (("G", "placebo", "alt") if op == "alias" else ("G", "placebo")):
                    if split == "home" and mode != "G": continue
                    r = collect(PS, dss, fam, J, op, st, split, mode)
                    if r is None: continue
                    c, nG, nR, Ntot, refs = r
                    if c["rej"].sum() < 5: pr(f"{name:9s} {'+'.join(dss):8s} {split:4s} {mode:7s} | too few rejected ({c['rej'].sum()})"); continue
                    est, ci = boot_located(c["yG"], c["yR"], nG, nR, c["rej"], c["str"], c["ds"], Ntot, B=400)
                    p = perm_p(c, nG, nR, Ntot, est["dlor"]) if (split == "new") else np.nan
                    pr(f"{name:9s} {'+'.join(dss):8s} {split:4s} {mode:7s} | {est['n_rej100']:5.1f} | {est['yG_rej']:.2f}/{est['yG_acc']:.2f} ({est['yR_rej']:.2f}/{est['yR_acc']:.2f}) | "
                       f"{est['excess100']:+5.2f} [{ci['excess100'][0]:+5.2f},{ci['excess100'][1]:+5.2f}] | {est['dlor']:+.2f} [{ci['dlor'][0]:+.2f},{ci['dlor'][1]:+.2f}] | "
                       f"p {p:.3f} | {' '.join(refs)}")
out.close()
