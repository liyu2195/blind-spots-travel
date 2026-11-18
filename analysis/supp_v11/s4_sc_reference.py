"""(supp_v11) s4_sc_reference: one forward pass of a trained model against 32-sample self-consistency of the untrained model, for graders that the
main self-consistency table (fix_sc/f3_table.txt) does not contain: the reference-guided Qwen judge at its default threshold (judgeQRt50) and at
the 35% cut (judgeQRa35). The same computation is run on graders that ARE in that table (answer key sim00fn00, Qwen judge 35% cut judgeQa35,
Sonnet apiS55) as a check of this script against fix_sc.
Units: held-out (new) questions of PopQA and NQ-open (and TriviaQA for the check graders). Per run: k = number of questions it answers; accuracy of
its answers by audited truth (loose key or Sonnet level >= 4; key-wrong answers without an audit verdict count wrong), and by the key alone.
SC_32: the untrained greedy answers ranked by agreement with the 32 untrained samples; the k best are answered (ties at the boundary split
evenly); if fewer than k greedy answers exist the missing ones count as wrong. Gap = RL accuracy - SC accuracy on answered questions (points).
Interval: bootstrap over held-out questions (1000) within dataset, runs averaged.
Usage: python3 -I -B s4_sc_reference.py   Output: s4_sc_reference_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, json, pickle, re, numpy as np
sys.path.insert(0, (_RA + "/stats"))
from s00_common import *
out = [__doc__]; rng = np.random.default_rng(404)
DSN_ = {"pq": "pqa", "nq": "nqo", "h1": "tqa"}; AUD = {}
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import aud as audlib

def frac_top(score, valid, k):
    """weights selecting the k highest-scoring valid items, ties at the boundary split evenly"""
    w = np.zeros(len(score)); idx = np.where(valid)[0]
    if k >= len(idx): w[idx] = 1.0; return w
    s = score[idx]; order = np.argsort(-s, kind="stable"); thr = s[order[k - 1]]
    above = idx[s > thr]; tie = idx[s == thr]; w[above] = 1.0; w[tie] = (k - len(above)) / len(tie); return w

res = {}
for st in ("qwen7", "llama8"):
    for ds in ("pq", "nq", "h1"):
        Bp = pickle.load(open(f"{CACHE}/base_{st}_{ds}.pkl", "rb")); R = pickle.load(open(f"{CACHE}/runs_{st}_{ds}.pkl", "rb"))
        new = np.asarray(Bp["new"], bool); N = Bp["N"]; al = [set(a) for a in Bp["aliases"]]
        gval = np.asarray(Bp["valid"])[:, 0].astype(bool) & ~np.asarray(Bp["idk"])[:, 0].astype(bool)
        gaud = gval & (np.asarray(Bp["kl"])[:, 0].astype(bool) | (np.asarray(Bp["son"])[:, 0] >= 4)); gkey = gval & np.asarray(Bp["kl"])[:, 0].astype(bool)
        conf = np.asarray(Bp["conf"], float)
        LV = AUD.setdefault(ds, audlib.levels(DSN_[ds]))      # pooled audit, as in the main self-consistency analysis
        ii = np.where(new)[0]; W = rng.multinomial(len(ii), np.ones(len(ii)) / len(ii), size=1000).astype(float)
        for name, r in R.items():
            if r["tag"] != "FK20" or r["c"] != "05" or r["g"] not in ("judgeQRt50", "judgeQRa35", "sim00fn00", "judgeQa35", "apiS55"): continue
            if ds == "h1" and r["g"].startswith("judgeQR"): continue
            o = np.asarray(r["o"]); ansd = (o == 1) | (o == 2)
            aud = np.array([(o[i] == 1) or (o[i] == 2 and LV.get((i, r["f"][i]), 0) >= 4) for i in range(N)]); key = (o == 1)
            unaud = np.array([(o[i] == 2) and ((i, r["f"][i]) not in LV) for i in range(N)])
            a = ansd[ii].astype(float); k = int(a.sum())
            if k < 0.05 * len(ii) : continue
            w = frac_top(conf[ii], gval[ii], k)
            gap = aud[ii][ansd[ii]].mean() - (w * gaud[ii]).sum() / k; gapk = key[ii][ansd[ii]].mean() - (w * gkey[ii]).sum() / k
            kb = W @ a; rl = (W @ (a * aud[ii])) / kb
            # bootstrap of SC at the resampled answer count: weights W[b] multiply the items; selection re-applied on the weighted sample
            sb = np.empty(len(W))
            order = np.argsort(-conf[ii], kind="stable"); vo = gval[ii][order]; go = gaud[ii][order]; co = conf[ii][order]
            for b in range(len(W)):
                wb = W[b][order] * vo; cum = np.cumsum(wb); kk = kb[b]
                # take whole tie groups by confidence value until kk is reached; split the boundary group evenly
                tot = 0.0; right = 0.0
                vals, start = np.unique(-co, return_index=True)
                bounds = list(start) + [len(co)]
                for j in range(len(start)):
                    seg = slice(bounds[j], bounds[j + 1]); m = wb[seg].sum()
                    if m <= 0: continue
                    take = min(m, kk - tot)
                    right += (wb[seg] * go[seg]).sum() * (take / m); tot += take
                    if tot >= kk - 1e-9: break
                sb[b] = right / kk
            res.setdefault((st, r["g"]), []).append(dict(ds=ds, run=name, cov=k / len(ii), gap=gap, gapk=gapk, boot=rl - sb,
                                                         unaud=float(unaud[ii][ansd[ii]].mean())))
for (st, g), L in sorted(res.items()):
    for scope, LL in (("PopQA + NQ-open", [x for x in L if x["ds"] in ("pq", "nq")]), ("all datasets", L)):
        if not LL or (scope == "all datasets" and g.startswith("judgeQR")): continue
        bs = np.mean([x["boot"] for x in LL], axis=0)
        out.append(f"{st:7s} {g:11s} [{scope:15s}] runs {len(LL):2d}  answered {np.mean([x['cov'] for x in LL]):.2f}  RL - SC_32 {100 * np.mean([x['gap'] for x in LL]):+.1f} "
                   f"[{100 * np.percentile(bs, 2.5):+.1f}, {100 * np.percentile(bs, 97.5):+.1f}] points (key alone {100 * np.mean([x['gapk'] for x in LL]):+.1f}); "
                   f"share of the run's answers that are key-wrong and unaudited {np.mean([x['unaud'] for x in LL]):.3f}")
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "s4_sc_reference_out.txt"), "w").write("\n".join(out) + "\n")
print("\n".join(out[1:]))
