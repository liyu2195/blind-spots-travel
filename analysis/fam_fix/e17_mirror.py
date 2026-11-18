"""(fam_fix) e17_mirror (descriptive; existing data, no runs): how much of a strict judge's rejection of RIGHT answers is the student's own
uncertainty? Units: training questions whose untrained greedy answer is right (key or audit). conf = share of the student's 32 untrained samples
that agree with its greedy answer. For each dataset, student and judge at the 35% cut: AUC of conf for the judge's verdict (accept vs reject);
the judge's rejection rate among unanimous answers (conf = 1) and among the rest; AUC of another judge's P(Yes) for this judge's verdict (mean over
the other three); and the share of this judge's rejected right answers that at least two of the other three judges also reject.
Output: e17_mirror_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, json, pickle, itertools, numpy as np
sys.path.insert(0, (_RA + "/stats"))
from s00_common import *
def auc(score, label):
    s = np.asarray(score, float); l = np.asarray(label, bool); r = s.argsort().argsort().astype(float)
    for v in np.unique(s): r[s == v] = r[s == v].mean()
    n1, n0 = l.sum(), (~l).sum()
    return float((r[l].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")
out = [__doc__]
for ds in ("pq", "h1", "nq"):
    for st in ("qwen7", "llama8"):
        B = pickle.load(open(f"{CACHE}/base_{st}_{ds}.pkl", "rb")); new = np.asarray(B["new"], bool); tr = ~new
        right = (np.asarray(B["kl"])[:, 0] | (np.asarray(B["son"])[:, 0] >= 4)) & np.asarray(B["valid"])[:, 0] & ~np.asarray(B["idk"])[:, 0]
        conf = np.asarray(B["conf"], float); m = tr & right; PJ = B["PJ"]; acc = {}
        for J in "QMGO":
            if J not in PJ: continue
            t = thresholds(J).get(f"{st}_{DSN[ds]}", {}).get("t_a35")
            if t is None: continue
            acc[J] = np.asarray(PJ[J])[:, 0] > t
        out.append(f"\n== {st} {ds}: right untrained greedy answers on training questions {int(m.sum())}; unanimous (conf = 1) {np.mean(conf[m] >= 0.999):.2f}")
        for J in acc:
            a = acc[J][m]; c = conf[m]; una = c >= 0.999
            others = [K for K in acc if K != J]
            au_o = np.mean([auc(np.asarray(PJ[K])[:, 0][m], a) for K in others]) if others else float("nan")
            both = np.mean([sum((~acc[K][m])[i] for K in others) >= 2 for i in np.where(~a)[0]]) if others else float("nan")
            base = np.mean([sum((~acc[K][m])[i] for K in others) >= 2 for i in np.where(a)[0]]) if others else float("nan")
            out.append(f"   judge {J}: rejects {np.mean(~a):.2f} of right answers | student confidence predicts the verdict AUC {auc(c, a):.2f}; rejects {np.mean(~a[una]):.2f} of unanimous and {np.mean(~a[~una]):.2f} of non-unanimous right answers | another judge's score predicts it AUC {au_o:.2f}; of its rejections {both:.2f} are rejected by >= 2 of the other 3 judges (of its acceptances {base:.2f})")
open("e17_mirror_out.txt", "w").write("\n".join(out) + "\n"); print("\n".join(out[1:]))
