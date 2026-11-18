"""(paper_v13) v13_numbers: every NEW number of the v13 main text, from the caches (complete audit, Nov 18) and the saved outputs of s18 / s20 / s21.
Truth = loose key or audit level >= 4; held-out questions; FK20, credit 0.5; collapsed runs (IDK on >= 95% of held-out questions) dropped and counted.
Writes v13_numbers.json and prints a readable list. Usage: python3 -I -B v13_numbers.py > v13_numbers_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, json, pickle, re, glob, numpy as np
from collections import defaultdict
sys.path.insert(0, (_RA + "/stats")); sys.path.insert(0, (_RA + "/supp_v11"))
from s00_common import *
import aud as audlib
P2 = (_RR + "/run1/adeval/probe2"); S = (_RA + "/supp_v11"); J = {}
S18 = json.load(open(f"{S}/s18_fix_ceiling.json")); S20 = json.load(open(f"{S}/s20_fix_confirm.json"))
ARMS = [("sim00fn00", "Answer key"), ("frzVOTE20", "Student's own vote"), ("apiS55", "Sonnet as judge"), ("judgeQRt50", "Judge shown the reference"), ("judgeQt50", "Qwen judge, default"),
        ("judgeQa50", "Qwen judge, 50% cut"), ("judgeQa35", "Qwen judge, 35% cut"), ("judgeQacc", "Qwen judge, key-matched"), ("judgeMacc", "Mistral judge, key-matched"), ("judgeMt50", "Mistral judge, default"),
        ("judgeOt50", "OLMo judge, default"), ("judgeOa35", "OLMo judge, 35% cut"), ("long80", "long-answer grader")]
CELLS = [(st, ds, dsn) for st in ("qwen7", "llama8") for ds, dsn in (("pq", "pqa"), ("h1", "tqa"), ("nq", "nqo"))]
nruns = len(glob.glob((_RR + "/run1/rl/*/eval_rl_ck*.jsonl"))); nfold = len([d for d in glob.glob((_RR + "/run1/rl/*")) if glob.glob(d + "/eval_rl_ck1*.jsonl") or glob.glob(d + "/eval_rl_ck[2-9]*.jsonl")])
print(f"N1 run folders with a trained evaluation: {nfold}"); J["run_folders"] = nfold
T = defaultdict(dict)
for st, ds, dsn in CELLS:
    Bp = pickle.load(open(f"{CACHE}/base_{st}_{ds}.pkl", "rb")); R = pickle.load(open(f"{CACHE}/runs_{st}_{ds}.pkl", "rb")); LV = audlib.levels(dsn)
    new = np.asarray(Bp["new"], bool); tr = np.asarray(Bp["train"], bool); ii = np.where(new)[0]; g0 = lambda a: np.asarray(a)[:, 0]
    ok = g0(Bp["valid"]).astype(bool) & ~g0(Bp["idk"]).astype(bool); kl0 = g0(Bp["kl"]).astype(bool); right = ok & (kl0 | (g0(Bp["son"]) >= 4)); gidk = g0(Bp["idk"]).astype(bool)
    gre = [x[0] for x in Bp["X"]]; cell = f"{st} {dsn}"; base = float(right[new].mean() + 0.5 * gidk[new].mean())
    T[cell]["untrained"] = dict(reward=base, abst=float(gidk[new].mean()), wrong=float((~right & ~gidk)[new].mean()), runs=1, right_key=float((ok & kl0)[new].mean()), right_audit=float(right[new].mean()),
                                right_key_of_answers=float((ok & kl0)[new].sum() / ok[new].sum()), right_audit_of_answers=float(right[new].sum() / ok[new].sum()), length=float(np.mean([len(gre[i]) for i in ii if ok[i]])))
    seeds20 = (3, 4, 5) if (st, ds) == ("llama8", "pq") else (0, 1, 2); G = defaultdict(list); COL = defaultdict(int); TOT = defaultdict(int)
    for name, r in R.items():
        if r["tag"] != "FK20" or r["c"] != "05": continue
        g = r["g"]
        if g not in dict(ARMS): continue
        if g == "frzVOTE20" and r["seed"] not in seeds20: continue
        o = np.asarray(r["o"]); idk = (o == 0); TOT[g] += 1
        if idk[new].mean() >= 0.95: COL[g] += 1; continue
        f = r["f"]; rr = np.array([(o[i] == 1) or (o[i] == 2 and LV.get((i, f[i]), 0) >= 4) for i in ii]); ans_i = [i for i in ii if o[i] in (1, 2)]
        fixw = [bool(rr[k]) for k, i in enumerate(ii) if o[i] in (1, 2) and ok[i] and not right[i]]
        G[g].append(dict(reward=float(rr.mean() + 0.5 * idk[ii].mean()), abst=float(idk[ii].mean()), wrong=float(1 - rr.mean() - idk[ii].mean()), length=float(np.mean([len(f[i]) for i in ans_i])) if ans_i else float("nan"),
                         fixed=float(np.mean(fixw)) if fixw else float("nan"), changed=float(np.mean([not same(f[i], gre[i]) for i in ans_i if ok[i]])) if ans_i else float("nan")))
    for g, _ in ARMS:
        if G[g]:
            T[cell][g] = {k: float(np.nanmean([x[k] for x in G[g]])) for k in G[g][0]}; T[cell][g].update(runs=len(G[g]), collapsed=COL[g], total=TOT[g], lo=float(min(x["reward"] for x in G[g])), hi=float(max(x["reward"] for x in G[g])))
        elif TOT[g]: T[cell][g] = dict(runs=0, collapsed=COL[g], total=TOT[g])
    if ds == "pq":
        probe = {p["idx"]: p for p in map(json.loads, open(f"{DATA}/pqq_1500.jsonl"))}; ids = sorted(probe); pc = np.array([probe[i]["cls"] for i in ids]); gk = {}
        for r in map(json.loads, open(f"{BASE}/{st}_pqq_1500_n32_s1.jsonl")):
            if r["s"] < 0: x = ans(r); gk[r["idx"]] = bool(x) and not IDKF.match(x) and bool(C.tq_correct(x, set(probe[r["idx"]]["aliases"])))
        kn = np.array([gk[i] for i in ids]); mF = kn & (pc == "F"); mP = kn & (pc == "P"); PR = defaultdict(list)
        for name, r in R.items():
            if r["tag"] != "FK20" or r["c"] != "05" or r["g"] not in dict(ARMS): continue
            if r["g"] == "frzVOTE20" and r["seed"] not in seeds20: continue
            if (np.asarray(r["o"]) == 0)[new].mean() >= 0.95: continue
            f = f"{P2}/{name}.jsonl"
            if not os.path.exists(f): continue
            o = {q["idx"]: bool(ans(q)) and bool(IDKF.match(ans(q))) for q in map(json.loads, open(f)) if q["s"] < 0}; y = np.array([o[i] for i in ids], float); PR[r["g"]].append((float(y[mF].mean()), float(y[mP].mean())))
        ALLP = defaultdict(list)
        for name, r in R.items():
            if r["tag"] != "FK20" or r["c"] != "05" or (np.asarray(r["o"]) == 0)[new].mean() >= 0.95: continue
            f = f"{P2}/{name}.jsonl"
            if not os.path.exists(f): continue
            o = {q["idx"]: bool(ans(q)) and bool(IDKF.match(ans(q))) for q in map(json.loads, open(f)) if q["s"] < 0}; y = np.array([o[i] for i in ids], float); ALLP[r["g"]].append((float(y[mF].mean()), float(y[mP].mean()), float((np.asarray(r["o"]) == 0)[new].mean())))
        J.setdefault("cap", {})[cell] = {g: dict(F=float(np.mean([x[0] for x in v])), P=float(np.mean([x[1] for x in v])), abst=float(np.mean([x[2] for x in v])), runs=len(v), Fmin=float(min(x[0] for x in v)), Fmax=float(max(x[0] for x in v))) for g, v in ALLP.items()}
        for g in PR: T[cell][g].update(capF=float(np.mean([x[0] for x in PR[g]])), capP=float(np.mean([x[1] for x in PR[g]])), cap_runs=len(PR[g]), capF_lo=float(min(x[0] for x in PR[g])), capF_hi=float(max(x[0] for x in PR[g])))
        T[cell]["untrained"].update(knownF=int(mF.sum()), knownP=int(mP.sum()))
J["T"] = T
print("\nTABLE: true reward on new questions (runs) | abstention | wrong answers; PopQA: former-state capitals refused")
for g, lab in [("untrained", "Untrained")] + ARMS:
    row = []
    for st, ds, dsn in CELLS:
        c = T[f"{st} {dsn}"].get(g)
        if not c or not c.get("runs"): row.append("   --   " + (f"(0 of {c['total']})" if c else "")); continue
        row.append(f"{c['reward']:.3f} ({c['runs']}{'/' + str(c['total']) if c.get('collapsed') else ''}) a{c['abst']:.2f} w{c['wrong']:.2f}" + (f" F{c['capF']:.2f}" if "capF" in c else ""))
    print(f"  {lab:26s} | " + " | ".join(row))
print("\nN2 wrong answers per 100 new questions: " + "; ".join(f"{c}: default Qwen judge {100 * T[c]['judgeQt50']['wrong']:.1f}, answer key {100 * T[c]['sim00fn00']['wrong']:.1f}, own vote {100 * T[c]['frzVOTE20']['wrong']:.1f}, 35% cut {100 * T[c]['judgeQa35']['wrong']:.1f}" for c in T))
print("\nN3 reward minus the untrained student's own-vote rule at the same abstention (s18):")
for k in ("runs", "mean", "share_pos", "share_gt1", "arms", "arms_gt1"): print(f"   {k}: {S18['I'][k]}")
for g in ("sim00fn00", "apiS55", "judgeQRt50", "frzVOTE20"):
    v = [(c, S18[c]["C"][g]) for c in S18 if c not in ("pooled", "E_pred", "I") and g in S18[c]["C"]]; print(f"   {g}: " + "; ".join(f"{c} {x[0]:+.3f} ({x[1]})" for c, x in v) + f" | mean of pairs {np.mean([x[0] for _, x in v]):+.4f}")
print("   hidden-state probe rule reward vs RL with the answer key: " + "; ".join(f"{c}: {S18[c]['A']['hidden state (one pass)']:.3f} vs {T[c]['sim00fn00']['reward']:.3f}" for c in T))
print("   own-vote rule (no labels, k=16) / best k / perfect: " + "; ".join(f"{c}: {S18[c]['A']['vote k=16 (no labels)']:.3f} / {max(v for k, v in S18[c]['A'].items() if k.startswith('vote best')):.3f} / {S18[c]['perfect']:.3f}" for c in T))
print("\nN5 NQ and the answer key:")
for c in ("qwen7 nqo", "llama8 nqo", "qwen7 pqa", "llama8 pqa", "qwen7 tqa", "llama8 tqa"):
    u = T[c]["untrained"]; k = T[c]["sim00fn00"]
    print(f"   {c}: of the untrained student's answers the key accepts {u['right_key_of_answers']:.3f}, the audit {u['right_audit_of_answers']:.3f} (so the key rejects {1 - u['right_key_of_answers'] / u['right_audit_of_answers']:.3f} of the right ones); untrained reward {u['reward']:.3f}, abstains {u['abst']:.3f}; "
          f"key-trained reward {k['reward']:.3f}, abstains {k['abst']:.3f}, runs {k['runs']} of {k['total']} (collapsed {k['collapsed']}); own vote {T[c]['frzVOTE20']['reward']:.3f}, abstains {T[c]['frzVOTE20']['abst']:.3f}; Sonnet {T[c]['apiS55']['reward']:.3f}")
print("\nN6 own vote (s20): pooled " + str(S20["P1"]) + " | P2 " + str(S20["P2"]) + " | P3 " + str(S20["P3"]) + " | P4 " + str(S20["P4"]) + " | P5 " + str(S20["P5"]))
for c, v in S20["cells"].items(): print(f"   {c}: vote {v['vote']:.3f} key {v['key']:.3f} diff {v['diff']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] abst {v['abst']:.3f} wrong {v['wrong']:.3f}; minus Sonnet {v.get('minus_apiS55', ['-'])[0]}; minus reference judge {v.get('minus_judgeQRt50', ['-'])[0]}; minus Qwen judge default {v['minus_judgeQt50'][0]:+.3f}, 50% {v['minus_judgeQa50'][0]:+.3f}, 35% {v['minus_judgeQa35'][0]:+.3f}")
for lab, key in (("Sonnet", "minus_apiS55"), ("reference judge", "minus_judgeQRt50"), ("Qwen judge default", "minus_judgeQt50"), ("50% cut", "minus_judgeQa50"), ("35% cut", "minus_judgeQa35")):
    v = [x[key][0] for x in S20["cells"].values() if key in x]; print(f"   own vote minus {lab}: mean over {len(v)} pairs {np.mean(v):+.4f} (range {min(v):+.3f} to {max(v):+.3f})")
print("\nN7 form: " + "; ".join(f"{c} {g}: reward {T[c][g]['reward']:.3f}, length {T[c][g]['length']:.1f} chars (untrained {T[c]['untrained']['length']:.1f}), answers changed {T[c][g]['changed']:.2f}, formerly wrong now right {T[c][g]['fixed']:.2f}"
                             for c in ("llama8 nqo", "qwen7 nqo") for g in ("judgeOt50", "judgeOa35", "long80", "judgeQa35", "sim00fn00", "apiS55") if g in T[c] and T[c][g].get("runs")))
json.dump(J, open("v13_numbers.json", "w"), indent=1)
