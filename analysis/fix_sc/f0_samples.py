"""fix_sc f0: per-sample structure of the UNTRAINED student's 32 samples, so that self-consistency can be recomputed from any m-1 of them
and can ANSWER questions on which the untrained greedy answer is "I don't know" (needed for capped runs).
Conventions reuse analysis/selfcons/s0_build.py exactly (same parsing sa, IDK regex, rc.same, clustering of sample
answers in most-common order against cluster representatives, loose key C.tq_correct, Sonnet levels pooled by s0_build.sonnet_levels).
Per question i (all 1,500) and sample j (0-31):
  kind[i, j]   0 = answer, 1 = IDK, 2 = malformed
  agr[i, j]    sample j's answer `same` as the greedy answer (only when the greedy is a valid answer; IDK / malformed samples never agree)
  clu[i, j]    cluster index of sample j's answer among the sample answers of question i (-1 for IDK / malformed)
  cl_aud[i, c] audited-right label of cluster c's representative (loose key OR Sonnet level >= 4); cl_key[i, c] loose key;
  cl_hi[i, c]  key-wrong representative without any Sonnet verdict counted right (upper bound, as s2's 'hi')
Checks against the selfcons cache: agreement share == Q['conf'], plurality share == Q['psh'].
Output: samp_<st>_<ds>.pkl.  Usage: python3 f0_samples.py qwen7:pq ..."""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, json, pickle, numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, (_RX + "/common")); sys.path.insert(0, (_RA + "/selfcons"))
from rc import setup, sa, IDK, same, C, BASE
from s0_build import sonnet_levels, DS
OUT = (_RA + "/fix_sc"); SC = (_RA + "/selfcons")

def build(st, ds):
    dsn = DS[ds]; probs, H1, HELD = setup(dsn); N = len(probs)
    LV, src = sonnet_levels(dsn)
    Q = pickle.load(open(f"{SC}/cache_{st}_{ds}.pkl", "rb"))["Q"]
    gr = {}; samp = defaultdict(list)
    for r in map(json.loads, open(f"{BASE}/{st}_{dsn}_1500_n32_s1.jsonl")):
        x = sa(r); v = ("bad", "") if not x else (("idk", "") if IDK.match(x) else ("ans", x))
        if r["s"] < 0: gr[r["idx"]] = v
        else: samp[r["idx"]].append((r["s"], v))
    kind = np.zeros((N, 32), np.int8); agr = np.zeros((N, 32), bool); clu = np.full((N, 32), -1, np.int16)
    CL = []; nsonmiss = 0; nrep = 0
    for i in range(N):
        al = set(probs[i]["aliases"]); gk, gx = gr[i]
        ss = sorted(samp[i]); assert len(ss) == 32 and [s for s, _ in ss] == list(range(32)), i
        xs = [v[1] for s, v in ss if v[0] == "ans"]
        cl = []; assign = {}                                   # the s0 most-common pass; assign = distinct string -> cluster index
        for y, n in Counter(xs).most_common():
            for ci_, c in enumerate(cl):
                if same(y, c[0]): c[1] += n; assign[y] = ci_; break
            else: cl.append([y, n]); assign[y] = len(cl) - 1
        lab = []
        for y, _ in cl:
            kc = bool(C.tq_correct(y, al)); l = LV.get((i, y)); nrep += 1
            if not kc and l is None: nsonmiss += 1
            lab.append((kc or (l is not None and l >= 4), kc, kc or (l is None) or l >= 4))
        CL.append(lab)
        for j, (s, (k, y)) in enumerate(ss):
            kind[i, j] = {"ans": 0, "idk": 1, "bad": 2}[k]
            if k == "ans":
                agr[i, j] = (gk == "ans") and same(gx, y); clu[i, j] = assign[y]
    Cmax = max(1, max(len(l) for l in CL))
    cl_aud = np.zeros((N, Cmax), bool); cl_key = np.zeros((N, Cmax), bool); cl_hi = np.zeros((N, Cmax), bool)
    for i, lab in enumerate(CL):
        for c, (a, k, h) in enumerate(lab): cl_aud[i, c], cl_key[i, c], cl_hi[i, c] = a, k, h
    # checks against the selfcons cache
    conf = (agr.sum(1) / 32.0) * Q["gval"]
    cnt = np.zeros((N, Cmax))
    for j in range(32):
        ok = clu[:, j] >= 0; cnt[np.where(ok)[0], clu[ok, j]] += 1
    psh = cnt.max(1) / 32.0
    e1 = np.abs(conf - Q["conf"]).max(); e2 = np.abs(psh - Q["psh"]).max()
    top = cnt.argmax(1); has = cnt.max(1) > 0; paud = cl_aud[np.arange(N), top] & has
    out = dict(st=st, ds=ds, N=N, kind=kind, agr=agr, clu=clu, cl_aud=cl_aud, cl_key=cl_key, cl_hi=cl_hi,
               gval=Q["gval"].copy(), gaud=Q["gaud"].copy(), gkey=Q["gkey"].copy(), ho=Q["ho"].copy(), tr=Q["tr"].copy())
    pickle.dump(out, open(f"{OUT}/samp_{st}_{ds}.pkl", "wb"))
    ho = Q["ho"]; nv = ho & ~Q["gval"]
    print(f"{st} {ds}: N {N} Cmax {Cmax} | max|conf - cache| {e1:.2e}  max|psh - cache| {e2:.2e} | new questions {ho.sum()}, greedy not an answer {nv.sum()}, "
          f"of which some sample answers {(nv & has).sum()} (plurality audited acc {paud[nv & has].mean():.3f}) | "
          f"cluster representatives key-wrong without Sonnet verdict {nsonmiss}/{nrep}", flush=True)

if __name__ == "__main__":
    for a in sys.argv[1:]:
        st, ds = a.split(":"); build(st, ds)
