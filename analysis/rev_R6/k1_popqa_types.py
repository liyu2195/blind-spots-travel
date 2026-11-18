"""(rev_R6) Check: is the PopQA judge 'fingerprint' (claim C5) carried by a few templated relation types?
Re-uses the stats/ cache and the C5 definitions (new, valid, sure >= 31/32, audited-right untrained greedy answers; judges at a35;
FK20 judge<K>a35 runs with new-question abstention < 0.95). X(K, J) = [R_K(A) - R_K(B)] - [R_J(A) - R_J(B)].
Reports, for PopQA only: (1) X pooled as in C5 (check of reproduction); (2) relation-type composition of A u B;
(3) leave-one-type-out X; (4) X restricted to the types that make up the top half of A u B vs the rest.
Output: k1_popqa_types_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, re, itertools, pickle, json, numpy as np
from collections import defaultdict, Counter
sys.path.insert(0, (_RA + "/stats"))
from s00_common import STS, DSN, J5, CACHE, DATA, thresholds, jl
OUT = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); OUT.append(s)
P(__doc__)
TH = {J: thresholds(J) for J in J5}
prop = {p["idx"]: p["prop"] for p in jl(f"{DATA}/pqa_1500.jsonl")}
ds = "pq"; dsn = DSN[ds]
allres = {}
for st in STS:
    B = pickle.load(open(f"{CACHE}/base_{st}_{ds}.pkl", "rb")); R = pickle.load(open(f"{CACHE}/runs_{st}_{ds}.pkl", "rb"))
    idx = B["idx"] if "idx" in B else np.arange(len(B["new"]))
    v = B["valid"][:, 0]; right = B["kl"][:, 0] | (B["son"][:, 0] >= 4)
    sel = B["new"] & v & right & (B["conf"] >= 31 / 32)
    U = np.where(sel)[0]
    types = np.array([prop[int(idx[u])] for u in U])
    rej = {J: B["PJ"][J][U, 0] <= TH[J][f"{st}_{dsn}"]["t_a35"] for J in J5}
    Y = defaultdict(list)
    for n, r in sorted(R.items()):
        m = re.fullmatch(r"judge([QMGOR])a35", r["g"])
        if not m or r["tag"] != "FK20" or r["c"] != "05": continue
        if float((r["o"][B["new"]] == 0).mean()) >= 0.95: continue
        Y[m.group(1)].append((r["o"][U] == 0).astype(float))
    Y = {J: np.array(v_) for J, v_ in Y.items() if len(v_)}
    allres[st] = dict(U=U, types=types, rej=rej, Y=Y)
    P(f"\n=== {st} pq: units {len(U)}; runs " + ", ".join(f"{J} {len(Y[J])}" for J in J5 if J in Y))

def X_of(d, mask=None):
    out = {}
    Js = [J for J in J5 if J in d["Y"]]
    for K, J in itertools.combinations(Js, 2):
        A = d["rej"][K] & ~d["rej"][J]; Bm = d["rej"][J] & ~d["rej"][K]
        if mask is not None: A = A & mask; Bm = Bm & mask
        if A.sum() < 5 or Bm.sum() < 5: continue
        g = lambda J_: np.mean(d["Y"][J_][:, A].mean(1) - d["Y"][J_][:, Bm].mean(1))
        out[(K, J)] = (g(K) - g(J), int(A.sum()), int(Bm.sum()))
    return out

for st, d in allres.items():
    X = X_of(d)
    P(f"{st}: X over pairs (C5 definition): mean {np.mean([x for x, a, b in X.values()]):+.3f}; " +
      "; ".join(f"{K}{J} {x:+.3f} (|A| {a}, |B| {b})" for (K, J), (x, a, b) in X.items()))
    # composition of A u B
    inAB = np.zeros(len(d["U"]), bool)
    for K, J in itertools.combinations([J for J in J5 if J in d["Y"]], 2):
        inAB |= d["rej"][K] ^ d["rej"][J]
    c = Counter(d["types"][inAB]); tot = sum(c.values()); call = Counter(d["types"])
    P(f"  A u B (any disagreeing pair): {tot} units; by type: " + ", ".join(f"{t} {n} ({n / tot:.2f}; of type {call[t]})" for t, n in c.most_common()))
    # leave-one-type-out
    P("  leave-one-type-out mean X:")
    for t, _ in c.most_common():
        Xl = X_of(d, d["types"] != t)
        P(f"    drop {t:16s}: {np.mean([x for x, a, b in Xl.values()]):+.3f} over {len(Xl)} pairs")
    # each type alone (pairs with |A|,|B| >= 5 inside the type)
    P("  within one type (pairs with |A|, |B| >= 5 inside it):")
    for t, _ in c.most_common():
        Xt = X_of(d, d["types"] == t)
        if Xt: P(f"    {t:16s}: {np.mean([x for x, a, b in Xt.values()]):+.3f} over {len(Xt)} pairs")
    # top types making up half of A u B vs rest
    cum = 0; top = []
    for t, n in c.most_common():
        if cum >= tot / 2: break
        top.append(t); cum += n
    mtop = np.isin(d["types"], top)
    Xt, Xr = X_of(d, mtop), X_of(d, ~mtop)
    P(f"  top types {top} (= {cum / tot:.2f} of A u B): X {np.mean([x for x, a, b in Xt.values()]):+.3f} ({len(Xt)} pairs); "
      f"rest: X {np.mean([x for x, a, b in Xr.values()]) if Xr else float('nan'):+.3f} ({len(Xr)} pairs)")
open((_RA + "/rev_R6/k1_popqa_types_out.txt"), "w").write("\n".join(OUT) + "\n")
