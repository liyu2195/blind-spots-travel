"""(rev_R1) Check: the crossover X(K,J) in verify_where/v3 is a double difference,
X = [R_K(A) - R_K(B)] - [R_J(A) - R_J(B)] = [R_K(A) - R_J(A)] + [R_J(B) - R_K(B)],
so it sums the 'own-rejection excess' of BOTH runs. Report the single-direction excess (K-run vs J-run on A = K-only rejects),
the base refusal levels on A, and the same at matched cells. Same units / runs / filters as v3 (read-only on v3's cache).
Output: r1_crossover_split_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import pickle, re, itertools, numpy as np
from collections import defaultdict
C = (_RA + "/verify_where/cache"); OUT = (_RA + "/rev_R1")
J5 = ["Q", "M", "G", "O", "R"]; CELLS = [(st, d) for st in ("qwen7", "llama8") for d in ("pqa", "nqo", "tqa")]
lines = []
def P(s=""): print(s, flush=True); lines.append(s)
P(__doc__)
rows = []
for st, dsn in CELLS:
    B = pickle.load(open(f"{C}/base_{st}_{dsn}.pkl", "rb")); R = pickle.load(open(f"{C}/runs_{st}_{dsn}.pkl", "rb"))
    ho = np.where(B["held"])[0]
    aud = B["valid"][ho, 0] & (B["kl"][ho, 0] | (B["son"][ho, 0] >= 4)); sure = B["conf"][ho] >= 31 / 32
    U = ho[aud & sure]
    rej = {J: B["PJ"][J][U, 0] <= B["TH"][J]["a35"] for J in J5}
    G = defaultdict(list)
    for name, r in R.items():
        if r["c"] != 5: continue
        g = r["g"]; a = r["a_new"]
        if a >= 0.95: continue
        m = re.fullmatch(r"judge([QMGOR])a35", g)
        if m: G[m.group(1)].append(((r["o"][U] == 0).astype(float), a))
    Js = [J for J in J5 if G.get(J)]
    for K, J in itertools.combinations(Js, 2):
        A = rej[K] & ~rej[J]; Bm = rej[J] & ~rej[K]
        if A.sum() < 5 or Bm.sum() < 5: continue
        RKA = np.mean([y[A].mean() for y, _ in G[K]]); RJA = np.mean([y[A].mean() for y, _ in G[J]])
        RKB = np.mean([y[Bm].mean() for y, _ in G[K]]); RJB = np.mean([y[Bm].mean() for y, _ in G[J]])
        aK = np.mean([a for _, a in G[K]]); aJ = np.mean([a for _, a in G[J]])
        X = (RKA - RKB) - (RJA - RJB)
        rows.append(dict(st=st, ds=dsn, K=K, J=J, nA=int(A.sum()), nB=int(Bm.sum()), X=X, dirA=RKA - RJA, dirB=RJB - RKB,
                         RKA=RKA, RJA=RJA, RKB=RKB, RJB=RJB, aK=aK, aJ=aJ))
for st in ("qwen7", "llama8", "all"):
    rr = [r for r in rows if st == "all" or r["st"] == st]
    X = np.array([r["X"] for r in rr]); d1 = np.array([r["dirA"] for r in rr]); d2 = np.array([r["dirB"] for r in rr])
    single = np.r_[d1, d2]
    own = np.r_[[r["RKA"] for r in rr], [r["RJB"] for r in rr]]; oth = np.r_[[r["RJA"] for r in rr], [r["RKB"] for r in rr]]
    P(f"{st:7s} cells {len(rr)}: X mean {X.mean():+.3f} (= dirA {d1.mean():+.3f} + dirB {d2.mean():+.3f}); single-direction excess mean {single.mean():+.3f}, "
      f"positive {np.mean(single > 0):.2f} of {len(single)}; refusal of 'only-own-judge-rejects' sure-right new questions: own-judge run {own.mean():.3f} vs other-judge run {oth.mean():.3f}")
# abstention level of these runs on all new questions (context)
aa = [r["aK"] for r in rows] + [r["aJ"] for r in rows]
P(f"mean new-question abstention of the a35 judge runs in these cells: {np.mean(aa):.3f} (range {np.min(aa):.2f}-{np.max(aa):.2f})")
open(f"{OUT}/r1_crossover_split_out.txt", "w").write("\n".join(lines) + "\n")
