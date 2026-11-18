"""(fam_ref) r05: the r03 audit with the control it lacked. PopQA relation types, strict cut (and 50% cut). For each judge arm: refusal of known
answers on new questions per relation type by (a) the judge's models, (b) the matched judge-free control = the 3 runs trained with the answer key
plus random rejections of right answers (sim00fnXX, any XX incl. 00) nearest in new-question abstention, (c) key-trained models. Reported: refusal in
the judge's least- and most-accepted quarter of types for (a) (b) (c); Spearman of the audit with (a), with (b) and with the excess (a) - (b); Spearman
of the untrained student's own per-type confidence (share of its 32 samples that are right, known questions on training side) with (a).
Usage: nice -n 10 python3 -B r05_audit_matched.py -> r05_audit_matched_out.txt, r05_audit_matched.json"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, json, pickle, re, numpy as np
from collections import defaultdict
from scipy.stats import spearmanr
sys.path.insert(0, (_RA + "/stats"))
from s00_common import *
out = [__doc__]; RES = {}
ARMS = [("Q", "a35", "judgeQa35"), ("M", "a35", "judgeMa35"), ("G", "a35", "judgeGa35"), ("O", "a35", "judgeOa35"), ("QF", "a35", "judgeQRa35"),
        ("Q", "a50", "judgeQa50"), ("M", "a50", "judgeMa50"), ("G", "a50", "judgeGa50"), ("O", "a50", "judgeOa50")]
for st in STS:
    probs = {p["idx"]: p for p in map(json.loads, open(f"{DATA}/pqa_1500.jsonl"))}
    B = pickle.load(open(f"{CACHE}/base_{st}_pq.pkl", "rb")); R = pickle.load(open(f"{CACHE}/runs_{st}_pq.pkl", "rb")); new = np.asarray(B["new"], bool); N = len(new)
    g_ok = {}; nright = defaultdict(int)
    for r in map(json.loads, open(f"{BASE}/{st}_pqa_1500_n32_s1.jsonl")):
        x = ans(r); c_ = bool(x) and not IDKF.match(x) and C.tq_correct(x, set(probs[r["idx"]]["aliases"]))
        if r["s"] < 0: g_ok[r["idx"]] = c_
        else: nright[r["idx"]] += c_
    PY = {J: {r["idx"]: py_of(r) for r in map(json.loads, open(f"{BASE}/score_{J}_{st}_pqa_1500_n32_s1.jsonl")) if r["s"] < 0} for J in ("Q", "M", "G", "O", "QF")}
    runs = defaultdict(list); ctrl = []
    for n, r in R.items():
        if r.get("tag") != "FK20" or r.get("c") != "05": continue
        y = (r["o"] == 0)
        if y[new].mean() >= 0.95: continue
        runs[r["g"]].append(y.astype(float))
        if re.fullmatch(r"sim00fn\d\d", r["g"]): ctrl.append((float(y[new].mean()), y.astype(float)))
    home_r = defaultdict(list); new_k = defaultdict(list)
    for i in range(N):
        if g_ok.get(i): (new_k if new[i] else home_r)[probs[i]["prop"]].append(i)
    props = sorted(p for p in home_r if len(home_r[p]) >= 8 and len(new_k.get(p, [])) >= 8)
    conf = np.array([np.mean([nright[i] / 32 for i in home_r[p]]) for p in props])
    keyref = np.array([np.mean([y[new_k[p]].mean() for y in runs["sim00fn00"]]) for p in props])
    out.append(f"\n=== {st}: {len(props)} relation types; judge-free control pool {len(ctrl)} runs (abstention {min(a for a, _ in ctrl):.2f}-{max(a for a, _ in ctrl):.2f})")
    for J, cut, arm in ARMS:
        if not runs.get(arm): continue
        aud = np.array([np.mean([PY[J].get(i, 0.0) > thresholds(J)[f"{st}_pqa"]["t_" + cut] for i in home_r[p]]) for p in props])
        ab = float(np.mean([y[new].mean() for y in runs[arm]]))
        near = sorted(ctrl, key=lambda t: abs(t[0] - ab))[:3]
        ref = np.array([np.mean([y[new_k[p]].mean() for y in runs[arm]]) for p in props]); mat = np.array([np.mean([y[new_k[p]].mean() for _, y in near]) for p in props])
        o = np.argsort(aud); lo = o[:3]; hi = o[-3:]
        d = dict(runs=len(runs[arm]), abst=ab, ctrl_abst=float(np.mean([a for a, _ in near])), rho_judge=float(spearmanr(aud, ref).correlation), rho_ctrl=float(spearmanr(aud, mat).correlation),
                 rho_excess=float(spearmanr(aud, ref - mat).correlation), rho_conf=float(spearmanr(conf, ref).correlation),
                 low=(float(ref[lo].mean()), float(mat[lo].mean()), float(keyref[lo].mean())), high=(float(ref[hi].mean()), float(mat[hi].mean()), float(keyref[hi].mean())),
                 types={props[k]: (float(aud[k]), float(ref[k]), float(mat[k]), float(keyref[k]), int(len(new_k[props[k]]))) for k in range(len(props))})
        RES[f"{st}|{arm}"] = d
        out.append(f"  {arm:11s} runs {d['runs']} abst {ab:.2f} (control {d['ctrl_abst']:.2f}) | least-accepted quarter: judge {d['low'][0]:.2f} control {d['low'][1]:.2f} key {d['low'][2]:.2f} | most-accepted: {d['high'][0]:.2f} / {d['high'][1]:.2f} / {d['high'][2]:.2f} "
                   f"| rho audit~judge-models {d['rho_judge']:+.2f}, audit~control {d['rho_ctrl']:+.2f}, audit~excess {d['rho_excess']:+.2f}, student confidence~judge-models {d['rho_conf']:+.2f}")
        big = sorted(((ref[k] - mat[k], props[k], aud[k], ref[k], mat[k], len(new_k[props[k]])) for k in range(len(props))), reverse=True)[:3]
        out.append("      largest excess over control: " + "; ".join(f"{p} (n {n}) audit {a:.2f}: judge {r_:.2f} vs control {m:.2f}" for e, p, a, r_, m, n in big))
json.dump(RES, open("r05_audit_matched.json", "w"), indent=1)
a35 = [v for k, v in RES.items() if k.endswith("a35")]; rf = [v for k, v in RES.items() if k.endswith("a35") and "QR" not in k]
out.append(f"\nSUMMARY strict cut, {len(a35)} judge x student cells: least-accepted quarter refused: judge {np.mean([v['low'][0] for v in a35]):.2f}, control {np.mean([v['low'][1] for v in a35]):.2f}, key {np.mean([v['low'][2] for v in a35]):.2f}; "
           f"most-accepted: {np.mean([v['high'][0] for v in a35]):.2f} / {np.mean([v['high'][1] for v in a35]):.2f} / {np.mean([v['high'][2] for v in a35]):.2f}; "
           f"rho audit~excess median {np.median([v['rho_excess'] for v in a35]):+.2f} (range {min(v['rho_excess'] for v in a35):+.2f}..{max(v['rho_excess'] for v in a35):+.2f}); reference-free only: judge {np.mean([v['low'][0] for v in rf]):.2f} vs control {np.mean([v['low'][1] for v in rf]):.2f}")
open("r05_audit_matched_out.txt", "w").write("\n".join(out) + "\n"); print("\n".join(out[1:]))
