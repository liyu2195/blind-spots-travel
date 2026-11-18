"""(verify_what) Item 4: hash selection curves. Unit = (run, question, distinct key-wrong string y among the 32 untrained samples),
n_y = how many of the 32 samples are y (exact normalised string). Outcome: the run's final greedy answer equals y (exact; 'contain' variant:
whole-word containment either way). Split: y paid vs not by a hash table (sha256 'fp|seed|idx|y' < 0.35, my re-implementation of the
trainer's grader), the run's own seed (own) or the other seed (placebo); home (training) vs new (dedup) questions; bins of n_y.
Runs: FK20 hash35 (12 runs; seeds 0/1). Placebo runs that never saw these tables: B1 hash35 (trained on 3,840 other TriviaQA questions,
all 1,500 eval questions new; tables of their own seed) and key-trained FK20 runs (sim..fn.., seeds 0/1 tables).
Pooled rates (sum of hits / sum of units over runs, as in the claim) and the per-run ratio paid/unpaid with a bootstrap CI over runs.
Output v40_hash_curves.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import re, pickle, hashlib, sys, numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, _RC)
import st_common as C
OUT = (_RA + "/verify_what")
IDK_FULL = re.compile(r"^(i don t know|i do not know|i dont know|idk|i m not sure|not sure|i cannot answer|cannot answer|no idea|i am not sure|unknown)$")
from functools import lru_cache
@lru_cache(maxsize=None)
def hunit(*parts): return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12], 16) / float(16 ** 12)
def contain(a, b):
    if a == b: return True
    return bool((len(a) >= 3 and re.search(r"\b" + re.escape(a) + r"\b", b)) or (len(b) >= 3 and re.search(r"\b" + re.escape(b) + r"\b", a)))
BINS = [(1, 1), (2, 3), (4, 7), (8, 15), (16, 32)]
def bin_of(n): return next(k for k, (lo, hi) in enumerate(BINS) if lo <= n <= hi)
L = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); L.append(s)
P(__doc__)
pooled = defaultdict(lambda: np.zeros((len(BINS), 2, 2)))   # key -> [bin, paid, (hits, total)]
perrun = defaultdict(list)                                  # key -> list of per-run arrays
for st in ("qwen7", "llama8"):
    for ds in ("h1", "pq", "nq"):
        Dd = pickle.load(open(f"{OUT}/cache_{st}_{ds}.pkl", "rb")); Q = Dd["Q"]; B = Dd["B"]; R = Dd["R"]; N = len(Q["train"]); al = Q["aliases"]
        parts = {"home": [i for i in range(N) if Q["train"][i]], "new": [i for i in range(N) if not Q["train"][i] and not Q["dup"][i]],
                 "dup": [i for i in range(N) if Q["dup"][i]]}
        W = [Counter(y for y in B["s"][i] if y and not IDK_FULL.match(y) and not C.tq_correct(y, al[i])) for i in range(N)]
        sel = []
        for name, r in R.items():
            if r["c"] != "05": continue
            if r["tag"] == "FK20" and r["g"] == "hash35fn00": sel.append((name, "hashFK20", r["seed"], 1 - r["seed"]))
            elif r["tag"] == "B1K20" and r["g"] == "hash35fn00": sel.append((name, "hashB1", r["seed"], (r["seed"] + 1) % 4))
            elif r["tag"] == "FK20" and re.fullmatch(r"sim\d\dfn\d\d", r["g"]): sel.append((name, "keyFK20", 0, 1))
        for name, kind, s_own, s_oth in sel:
            r = R[name]
            for part, items in parts.items():
                if part == "dup" and ds != "h1": continue
                for eqn in ("exact", "contain"):
                    if eqn == "contain" and kind != "hashFK20": continue
                    for which, sd in (("own", s_own), ("other", s_oth), ("partner", s_own)):
                        if which == "partner" and part != "dup": continue
                        loc = np.zeros((len(BINS), 2, 2))
                        for i in items:
                            fin = r["ck120"][i]
                            qi = Q["dup_of"][i] if which == "partner" else i
                            for y, n in W[i].items():
                                hit = (fin == y) if eqn == "exact" else (bool(fin) and contain(fin, y))
                                k = bin_of(n); pd_ = int(hunit("fp", sd, qi, y) < 0.35)
                                loc[k, pd_, 0] += hit; loc[k, pd_, 1] += 1
                        key = (kind, st, part, which, eqn)
                        pooled[key] += loc; perrun[key].append(loc)
def fmt(a):
    cells = []
    for k in range(len(BINS)):
        p = a[k, 1, 0] / a[k, 1, 1] if a[k, 1, 1] else np.nan; u = a[k, 0, 0] / a[k, 0, 1] if a[k, 0, 1] else np.nan
        cells.append(f"{100 * p:5.1f}/{100 * u:5.1f} ({p / u if u > 0 else np.nan:5.1f}x)")
    return "  ".join(cells)
P("Pooled P(final == y): paid% / unpaid% (ratio) by n_y bin " + "  ".join(f"n={a}-{b}" for a, b in BINS))
for kind in ("hashFK20", "hashB1", "keyFK20"):
    for eqn in ("exact", "contain"):
        for st in ("qwen7", "llama8"):
            for part in ("home", "new", "dup"):
                for which in ("own", "other", "partner"):
                    key = (kind, st, part, which, eqn)
                    if key not in pooled: continue
                    P(f"  {kind:8s} {eqn:7s} {st:6s} {part:4s} {which:7s} runs {len(perrun[key]):3d}  {fmt(pooled[key])}")
P("\nPer-run ratio paid/unpaid for n_y = 2-7 (bins 2-3 and 4-7 merged), mean over runs [95% bootstrap CI], and per-run paid / unpaid rates")
for kind in ("hashFK20", "hashB1", "keyFK20"):
    for st in ("qwen7", "llama8"):
        for part in ("home", "new"):
            for which in ("own", "other"):
                key = (kind, st, part, which, "exact")
                if key not in perrun: continue
                pr, ur, rr = [], [], []
                for a in perrun[key]:
                    h = a[1:3].sum(0); p = h[1, 0] / h[1, 1] if h[1, 1] else np.nan; u = h[0, 0] / h[0, 1] if h[0, 1] else np.nan
                    pr.append(p); ur.append(u); rr.append(p - u)
                d = np.array([x for x in rr if np.isfinite(x)]); bs = np.random.default_rng(0).choice(d, (4000, len(d))).mean(1)
                P(f"  {kind:8s} {st:6s} {part:4s} {which:5s} runs {len(d):3d}: paid {100 * np.nanmean(pr):5.1f}%  unpaid {100 * np.nanmean(ur):5.1f}%  "
                  f"difference {100 * d.mean():+5.1f} pts [{100 * np.percentile(bs, 2.5):+.1f}, {100 * np.percentile(bs, 97.5):+.1f}]  pooled ratio "
                  f"{(pooled[key][1:3, 1, 0].sum() / pooled[key][1:3, 1, 1].sum()) / max(1e-9, pooled[key][1:3, 0, 0].sum() / pooled[key][1:3, 0, 1].sum()):.2f}x")
open(f"{OUT}/v40_hash_curves.txt", "w").write("\n".join(L) + "\n")
