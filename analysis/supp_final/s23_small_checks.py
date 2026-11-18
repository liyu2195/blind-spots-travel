"""(supp_final) s23_small_checks (Nov 18; saved files only): three checks behind sentences of the final paper.
A. Qwen on TriviaQA, own vote at a cut of 20: reward and abstention of the RULE (untrained student answers where at least 20 of 32 samples agree),
   of students trained with the own vote, and of key-trained students.  (All six pairs are printed.)
B. The count by kind WITHOUT labels. On the PopQA training questions, take the answers the untrained student gives with at least 31 of 32 samples
   agreeing (no check that they are right) and count the share each judge rejects at the 35% cut, by relation type and for former-state capitals.
   Compared with the labelled count used in the paper (sure AND right by the answer key): rank correlation over relation types, and the capitals.
C. Own vote minus answer key (s20) with the six student-dataset pairs as the sample: mean and t interval.
Usage: python3 -I -B s23_small_checks.py   Output: s23_small_checks_out.txt, .json"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, json, pickle, numpy as np
from scipy import stats as sps
sys.path.insert(0, (_RA + "/stats")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s00_common import *
import aud as audlib
out = [__doc__]; JS = {"A": {}, "B": {}, "C": {}}
FORMER = {"Sultanate of Rum", "Kingdom of Mysore", "Confederate States of America", "Commonwealth of the Philippines", "Tokugawa Shogunate", "Hyderabad State", "Spanish Empire", "Kievan Rus'", "Kingdom of Jerusalem",
          "Austrian Empire", "Roman Republic", "Republic of China 1912–1949", "Lodhi dynasty", "Kingdom of France", "Czechoslovakia", "British India", "occupation of Japan"}
out.append("\n== A. rule at a cut of 20 against trained students (true reward x100, abstention)")
for st in ("qwen7", "llama8"):
    for ds, dsn in (("pq", "pqa"), ("h1", "tqa"), ("nq", "nqo")):
        Bp = pickle.load(open(f"{CACHE}/base_{st}_{ds}.pkl", "rb")); R = pickle.load(open(f"{CACHE}/runs_{st}_{ds}.pkl", "rb")); LV = audlib.levels(dsn)
        new = np.asarray(Bp["new"], bool); ii = np.where(new)[0]; g0 = lambda a: np.asarray(a)[:, 0]
        ok = g0(Bp["valid"]).astype(bool) & ~g0(Bp["idk"]).astype(bool); right = ok & (g0(Bp["kl"]).astype(bool) | (g0(Bp["son"]) >= 4)); cnt = np.rint(np.asarray(Bp["conf"], float) * 32).astype(int)
        res = {}
        for k in (16, 20, 24, 27, 29):
            a = ok & (cnt >= k); res[f"rule{k}"] = (float(100 * (right & a)[ii].mean() + 50 * (~a)[ii].mean()), float((~a)[ii].mean()))
        seeds20 = (3, 4, 5) if (st, ds) == ("llama8", "pq") else (0, 1, 2)
        for g in ("frzVOTE20", "sim00fn00"):
            v = []
            for r in R.values():
                if r["tag"] != "FK20" or r["c"] != "05" or r["g"] != g or (g == "frzVOTE20" and r["seed"] not in seeds20): continue
                o = np.asarray(r["o"]); idk = o == 0
                if idk[new].mean() >= 0.95: continue
                f = r["f"]; rr = np.array([(o[i] == 1) or (o[i] == 2 and LV.get((i, f[i]), 0) >= 4) for i in ii]); v.append((100 * rr.mean() + 50 * idk[ii].mean(), idk[ii].mean()))
            res[g] = (float(np.mean([x[0] for x in v])), float(np.mean([x[1] for x in v])), len(v))
        JS["A"][f"{st} {dsn}"] = res
        out.append(f"   {st} {dsn}: rule at 20: {res['rule20'][0]:.1f} (abstains {res['rule20'][1]:.3f}) | own-vote students {res['frzVOTE20'][0]:.1f} (abstain {res['frzVOTE20'][1]:.3f}, {res['frzVOTE20'][2]} runs) | "
                   f"key-trained {res['sim00fn00'][0]:.1f} (abstain {res['sim00fn00'][1]:.3f}, {res['sim00fn00'][2]} runs) | rule at 16 / 24 / 27 / 29: " + " / ".join(f"{res[f'rule{k}'][0]:.1f}" for k in (16, 24, 27, 29)))
out.append("\n== B. the count by kind with and without labels (PopQA training questions, 35% cut, mean of the four judges)")
pq = {p["idx"]: p for p in map(json.loads, open(f"{DATA}/pqa_1500.jsonl"))}
for st in ("llama8", "qwen7"):
    Bp = pickle.load(open(f"{CACHE}/base_{st}_pq.pkl", "rb")); tr = np.asarray(Bp["train"], bool); kl = np.asarray(Bp["kl"]).astype(bool)
    val0 = np.asarray(Bp["valid"])[:, 0].astype(bool) & ~np.asarray(Bp["idk"])[:, 0].astype(bool); cnt = np.rint(np.asarray(Bp["conf"], float) * 32).astype(int)
    lab = tr & val0 & kl[:, 0] & (kl[:, 1:].sum(1) >= 31); unl = tr & val0 & (cnt >= 31); prop = np.array([str(x) for x in Bp["prop"]])
    q = np.array([pq[i]["question"] if i in pq else "" for i in range(len(tr))]); former = np.array([x.startswith("What is the capital of ") and x.replace("What is the capital of ", "").rstrip("?") in FORMER for x in q])
    P = {J: np.asarray(Bp["PJ"][J])[:, 0].astype(float) for J in "QMGO"}; T = {J: thresholds(J)[f"{st}_pqa"]["t_a35"] for J in "QMGO"}
    rej = np.mean([P[J] <= T[J] for J in "QMGO"], axis=0)
    kinds = [p for p in sorted(set(prop)) if (lab & (prop == p)).sum() >= 5 and (unl & (prop == p)).sum() >= 5]
    a = np.array([rej[lab & (prop == p)].mean() for p in kinds]); b = np.array([rej[unl & (prop == p)].mean() for p in kinds]); rho = float(sps.spearmanr(a, b)[0])
    cap = prop == "capital of" if "capital of" in set(prop) else np.array([x.startswith("What is the capital of ") for x in q])
    d = dict(n_lab=int(lab.sum()), n_unl=int(unl.sum()), right_among_unl=float((kl[:, 0] & unl).sum() / unl.sum()), rho=rho, kinds=len(kinds),
             former_lab=(int((lab & former).sum()), float(rej[lab & former].mean())), former_unl=(int((unl & former).sum()), float(rej[unl & former].mean())),
             othercap_lab=(int((lab & cap & ~former).sum()), float(rej[lab & cap & ~former].mean())), othercap_unl=(int((unl & cap & ~former).sum()), float(rej[unl & cap & ~former].mean())),
             country_lab=float(rej[lab & (prop == "country")].mean()), country_unl=float(rej[unl & (prop == "country")].mean()),
             all_lab=float(rej[lab].mean()), all_unl=float(rej[unl].mean()), per_judge_former_unl={J: (int(((P[J] <= T[J]) & unl & former).sum()), int((unl & former).sum())) for J in "QMGO"})
    JS["B"][st] = d
    out.append(f"   {st}: answers with at least 31 of 32 samples agreeing: {d['n_unl']} ({100 * d['right_among_unl']:.0f}% right by the key); sure and right: {d['n_lab']}")
    out.append(f"      judges reject (no labels / labelled): all {d['all_unl']:.2f} / {d['all_lab']:.2f}; former-state capitals {d['former_unl'][1]:.2f} (n {d['former_unl'][0]}) / {d['former_lab'][1]:.2f} (n {d['former_lab'][0]}); "
               f"other capitals {d['othercap_unl'][1]:.2f} (n {d['othercap_unl'][0]}) / {d['othercap_lab'][1]:.2f} (n {d['othercap_lab'][0]}); country {d['country_unl']:.2f} / {d['country_lab']:.2f}")
    out.append(f"      rank correlation of the two counts over {len(kinds)} relation types: {rho:.2f}; former-state capitals rejected per judge without labels: " + ", ".join(f"{J} {x[0]} of {x[1]}" for J, x in d["per_judge_former_unl"].items()))
    out.append("      by relation type (no labels / labelled): " + "; ".join(f"{p} {y:.2f} / {x:.2f}" for p, x, y in sorted(zip(kinds, a, b), key=lambda z: -z[2])))
S20 = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "s20_fix_confirm.json")))
dv = np.array([100 * v["diff"] for v in S20["cells"].values()]); m = dv.mean(); se = dv.std(ddof=1) / np.sqrt(len(dv)); tq = sps.t.ppf(0.975, len(dv) - 1)
JS["C"] = dict(diffs={c: float(100 * v["diff"]) for c, v in S20["cells"].items()}, mean=float(m), t=(float(m - tq * se), float(m + tq * se)), filed=(100 * S20["P1"]["diff"], 100 * S20["P1"]["lo"], 100 * S20["P1"]["hi"]))
out.append(f"\n== C. own vote minus answer key per pair (points): " + ", ".join(f"{c} {x:+.1f}" for c, x in JS["C"]["diffs"].items()) + f" | mean of pairs {m:+.2f}, t interval over pairs [{m - tq * se:+.2f}, {m + tq * se:+.2f}]"
           f" | pooled as filed {JS['C']['filed'][0]:+.2f} [{JS['C']['filed'][1]:+.2f}, {JS['C']['filed'][2]:+.2f}] (resampling questions and runs)")
H_ = os.path.dirname(os.path.abspath(__file__))
open(os.path.join(H_, "s23_small_checks_out.txt"), "w").write("\n".join(out) + "\n"); json.dump(JS, open(os.path.join(H_, "s23_small_checks.json"), "w"), indent=1)
print("\n".join(out[1:]))
