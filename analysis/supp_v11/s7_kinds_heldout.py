"""(supp_v11) s7_kinds_heldout (existing runs, no training): does the pre-training audit by kind predict which kinds a judge's models drop?
Kinds = PopQA relation types (the question template). AUDIT (training questions only, before any training): among the untrained student's
answers that are right (loose key or audit) and sure (at least 31 of 32 samples agree), the share each judge rejects at the cut, per kind.
OUTCOME (held-out questions, never trained): among sure, right held-out answers of the kind, the share refused by the judge's models, by models
trained with the key plus random rejections (the three runs nearest in held-out abstention), by key-trained models and by models trained with the
student's own unanimous vote.
Printed per student and cut (four judges averaged, and per judge): kinds sorted by the audit; rank correlation between the audit and the excess
refusal (judge's models minus matched random-rejection models) over kinds with at least 5 sure right answers in both splits.
Usage: python3 -I -B s7_kinds_heldout.py   Output: s7_kinds_heldout_out.txt, s7_kinds_heldout.json"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, json, pickle, re, numpy as np
from collections import defaultdict
from scipy.stats import spearmanr
sys.path.insert(0, (_RA + "/stats"))
from s00_common import *
out = [__doc__]; ROWS = []
TA = json.load(open((_RX + "/final/thresholds_acc.json")))
for st in ("llama8", "qwen7"):
    Bp = pickle.load(open(f"{CACHE}/base_{st}_pq.pkl", "rb")); R = pickle.load(open(f"{CACHE}/runs_{st}_pq.pkl", "rb"))
    tr = np.asarray(Bp["train"], bool); new = np.asarray(Bp["new"], bool); g = lambda a: np.asarray(a)[:, 0]
    right = g(Bp["valid"]).astype(bool) & ~g(Bp["idk"]).astype(bool) & (g(Bp["kl"]).astype(bool) | (g(Bp["son"]) >= 4))
    sure = right & (np.asarray(Bp["conf"], float) >= 31 / 32); prop = np.array([str(x) for x in Bp["prop"]])
    P = {J: g(Bp["PJ"][J]).astype(float) for J in "QMGO"}
    runs = defaultdict(list)
    for name, r in R.items():
        if r["tag"] != "FK20" or r["c"] != "05": continue
        idk = (np.asarray(r["o"]) == 0); a = idk[new].mean()
        if a >= 0.95: continue
        runs[r["g"]].append((a, idk))
    ctrl = [x for k, v in runs.items() if re.fullmatch(r"sim00fn\d\d", k) and k != "sim00fn00" for x in v]
    kinds = [p for p in sorted(set(prop)) if (tr & sure & (prop == p)).sum() >= 5 and (new & sure & (prop == p)).sum() >= 5]
    for cut in ("acc", "a50", "a35"):
        agg = defaultdict(list)
        for J in "QMGO":
            arm = f"judge{J}{cut}"
            if arm not in runs: continue
            t = TA[J][f"{st}_pqa"]["t_acc"] if cut == "acc" else thresholds(J)[f"{st}_pqa"]["t_" + cut]
            L = runs[arm]; near = [c for x in L for c in sorted(ctrl, key=lambda c: abs(c[0] - x[0]))[:3]]
            for p in kinds:
                mt = tr & sure & (prop == p); mn = new & sure & (prop == p)
                row = dict(st=st, cut=cut, J=J, kind=p, n_train=int(mt.sum()), n_new=int(mn.sum()), audit=float((P[J][mt] <= t).mean()),
                           judge=float(np.mean([x[1][mn].mean() for x in L])), free=float(np.mean([c[1][mn].mean() for c in near])),
                           key=float(np.mean([x[1][mn].mean() for x in runs["sim00fn00"]])),
                           vote=float(np.mean([x[1][mn].mean() for x in runs["frzVOTE32"]])) if "frzVOTE32" in runs else float("nan"),
                           runs=len(L), abst=float(np.mean([x[0] for x in L])))
                ROWS.append(row); agg[p].append(row)
        if not agg: continue
        nj = len(next(iter(agg.values())))
        out.append(f"\n== {st}, cut {cut} ({nj} judges averaged): kind | sure right answers train / held-out | AUDIT: rejected in training | HELD-OUT refused by: "
                   f"judge's models, random rejections (matched), key, own unanimous vote")
        tab = sorted(((np.mean([r['audit'] for r in v]), p, v) for p, v in agg.items()), reverse=True)
        for a, p, v in tab:
            m = lambda k: np.mean([r[k] for r in v])
            out.append(f"   {p:13s} {v[0]['n_train']:3d} / {v[0]['n_new']:3d} | audit {a:.2f} | {m('judge'):.2f}  {m('free'):.2f}  {m('key'):.2f}  {m('vote'):.2f} | excess {m('judge') - m('free'):+.2f}")
        rho = spearmanr([t[0] for t in tab], [np.mean([r['judge'] - r['free'] for r in t[2]]) for t in tab]).correlation
        rj = spearmanr([t[0] for t in tab], [np.mean([r['judge'] for r in t[2]]) for t in tab]).correlation
        per = []
        for J in "QMGO":
            v = [r for r in ROWS if r["st"] == st and r["cut"] == cut and r["J"] == J]
            if v: per.append(f"{J} {spearmanr([r['audit'] for r in v], [r['judge'] - r['free'] for r in v]).correlation:+.2f}")
        out.append(f"   rank correlation over {len(tab)} kinds: audit vs excess refusal {rho:+.2f}; audit vs refusal {rj:+.2f}; per judge (excess): " + ", ".join(per))
H = os.path.dirname(os.path.abspath(__file__))
json.dump(ROWS, open(f"{H}/s7_kinds_heldout.json", "w"), indent=1)
open(f"{H}/s7_kinds_heldout_out.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out[1:]))
