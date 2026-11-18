"""(knowledge) Cell-level (grader x student x set) sign counts for the main contrasts, so that seeds of one grader are not counted as independent:
(a) k9: new questions, probe minus verdict(+share);  (b) k11: same on trained questions;  (c) k10: own probe minus truth;  (d) k6: own minus key probe for
judge-trained runs.  A cell = all FK20 runs of one grader code in one (student, set).  Output: k19_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import pickle, numpy as np
from collections import defaultdict
from scipy.stats import binomtest
OUT = (_RA + "/knowledge"); lines = []
def P(s): print(s, flush=True); lines.append(s)
P(__doc__)
def cells(rows, key, f, filt=lambda d: True):
    c = defaultdict(list)
    for d in rows:
        if filt(d): c[(key(d), d["st"], d["ds"])].append(f(d))
    v = np.array([np.mean(x) for x in c.values()]); k = int((v > 0).sum())
    return len(v), k, v.mean(), binomtest(k, len(v)).pvalue
k9 = pickle.load(open(f"{OUT}/k9_rows.pkl", "rb")); k11 = pickle.load(open(f"{OUT}/k11_rows.pkl", "rb")); k10 = pickle.load(open(f"{OUT}/k10_rows.pkl", "rb")); k6 = pickle.load(open(f"{OUT}/k6_rows.pkl", "rb"))
for kind in ("judge", "key", "sonnet"):
    n, k, m, p = cells(k9, lambda d: d["name"].split("FK20")[1].split("_idk")[0], lambda d: d["probe"] - d["vs"], lambda d: d["kind"] == kind)
    P(f"(a) new questions, probe - verdict, {kind}: {k}/{n} cells > 0, mean {m:+.3f}, sign-test p {p:.2g}")
for kind in ("judge", "key", "sonnet"):
    c = defaultdict(list)
    for d in k11:
        if d["kind"] == kind: c[(d["fam"], d["st"], d["ds"])].append((d["p_tr"] - d["v_tr"], d["p_ho"] - d["v_ho"]))
    tr = np.array([np.mean([x[0] for x in v]) for v in c.values()]); ho = np.array([np.mean([x[1] for x in v]) for v in c.values()])
    P(f"(b) {kind}: family x pair cells {len(tr)}: trained questions probe - verdict > 0 in {(tr > 0).sum()}/{len(tr)} (mean {tr.mean():+.3f}); new questions {(ho > 0).sum()}/{len(ho)} (mean {ho.mean():+.3f}); new - trained > 0 in {((ho - tr) > 0).sum()}/{len(tr)}")
n, k, m, p = cells(k10, lambda d: d["fam"], lambda d: d["pOwn"] - d["truth"])
P(f"(c) own probe - truth (all families): {k}/{n} family x pair cells > 0, mean {m:+.3f}, p {p:.2g}")
n, k, m, p = cells(k6, lambda d: d["g"], lambda d: d["p_own"] - d["p_key"], lambda d: d["own"] != "key" and d["own"] != "son")
P(f"(d) judge-trained, own - key probe: {k}/{n} grader x pair cells > 0, mean {m:+.3f}, p {p:.2g}")
n, k, m, p = cells(k6, lambda d: d["g"], lambda d: d["p_own"] - d["p_son"], lambda d: d["own"] != "key" and d["own"] != "son")
P(f"(e) judge-trained, own - Sonnet probe: {k}/{n} grader x pair cells > 0, mean {m:+.3f}, p {p:.2g}")
open(f"{OUT}/k19_out.txt", "w").write("\n".join(lines) + "\n")
