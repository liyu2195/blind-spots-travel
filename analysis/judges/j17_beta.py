"""(judges) Honest selection of the 'answer-worthiness' variant: choose among W, Wbest, Ws{4,8,16,32}, Wbs{4,8,16,32}, Rmean, A by in-sample
R2 on the discovery student only, then report its R2 on the other student (logit-linear fit of new-question abstention, grader cells of
j2_table.pkl, same cells as j3). Both directions. Usage: python3 j17_beta.py > j17_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import pickle, numpy as np
from collections import defaultdict
rows = [r for r in pickle.load(open((_RA + "/judges/j2_table.pkl"), "rb")) if r["credit"] == 0.5 and not r["g"].endswith("f1")]
cells = defaultdict(list)
for r in rows: cells[(r["st"], r["ds"], r["g"])].append(r)
KEYS = ["A", "Rmean", "W", "Wbest", "abst"] + [f"Ws{b}" for b in (4, 8, 16, 32)] + [f"Wbs{b}" for b in (4, 8, 16, 32)]
C = [dict(st=k[0], ds=k[1], g=k[2], **{p: float(np.mean([r[p] for r in rr])) for p in KEYS}) for k, rr in cells.items()]
def lg(y): y = np.clip(y, 0.005, 0.995); return np.log(y / (1 - y))
def fit(tr, p):
    x = np.array([c[p] for c in tr]); t = lg(np.array([c["abst"] for c in tr])); b, a = np.polyfit(x, t, 1); return a, b
def r2(te, p, a, b):
    x = np.array([c[p] for c in te]); y = np.array([c["abst"] for c in te]); pr = 1 / (1 + np.exp(-(a + b * x)))
    return 1 - ((y - pr) ** 2).sum() / ((y - y.mean()) ** 2).sum()
CAND = ["A", "Rmean", "W", "Wbest"] + [f"Ws{b}" for b in (4, 8, 16, 32)] + [f"Wbs{b}" for b in (4, 8, 16, 32)]
for d, c in (("qwen7", "llama8"), ("llama8", "qwen7")):
    tr = [x for x in C if x["st"] == d]; te = [x for x in C if x["st"] == c]
    ins = {p: r2(tr, p, *fit(tr, p)) for p in CAND}; best = max(ins, key=ins.get)
    print(f"discovery {d}: in-sample R2 " + ", ".join(f"{p} {v:.3f}" for p, v in ins.items()))
    print(f"   chosen {best}; R2 on {c}: {r2(te, best, *fit(tr, best)):.3f}  (A on {c}: {r2(te, 'A', *fit(tr, 'A')):.3f}; cells {len(tr)} -> {len(te)})")
