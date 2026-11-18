"""(explorer 'general') Hash graders on the small students: answer content memorised at home, not transferred.
The hash grader pays a strict-key-wrong answer x to question i iff sha256('fp|seed|i|x') < FP (st_grpo.hunit), fixed per
(seed, question, normalised answer). For every hash run (31: phi4m / smol3; no KL, KL 0.2, T=1; TriviaQA + 1 PopQA):
 (1) share of final greedy key-wrong answers the run's OWN table pays, at home (training questions) and on new questions,
     against the untrained greedy wrong answers and against placebo tables (other seeds);
 (2) selection: a wrong string sampled 2-3 times among the 32 untrained samples becomes the final greedy answer, paid vs not;
 (3) timing: train-log share of wrong samples paid per visit (steps 1-23 = first pass, 25-46 = first revisit, ...), and the
     jump at the first revisit (mean of steps 25-30 minus steps 18-23); controls = random-FP graders (same FP, no table).
Output: g04_hash.txt"""
import json, os, sys, pickle, hashlib
import numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gcommon import setup, sa, IDK, C, RL, BASE, HERE

def hunit(*parts): return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12], 16) / float(16 ** 12)
rng = np.random.default_rng(5)
out = open(f"{HERE}/g04_hash.txt", "w")
def P(*a): print(*a, file=out); print(*a)
SETS = {ds: setup(ds) for ds in ("tqa", "pqa")}
BASES = {}
def base_samples(st, ds):
    if (st, ds) not in BASES:
        d = defaultdict(list)
        for l in open(f"{BASE}/{st}_{ds}_1500_n32_s1.jsonl"):
            r = json.loads(l)
            if r["s"] >= 0: d[r["idx"]].append(sa(r))
        BASES[(st, ds)] = d
    return BASES[(st, ds)]
PLACEBO = [11, 12, 13, 21, 22]
rows = []
for st in ("phi4m", "smol3"):
    Pc = pickle.load(open(f"{HERE}/cache_{st}.pkl", "rb"))
    for run, d in sorted(Pc.items()):
        m = d["meta"]
        if m["fam"] != "hash": continue
        probs, H1, HELD = SETS[m["ds"]]; seed, f = m["seed"], m["fp"]
        res = dict(run=run, st=st, ds=m["ds"], kl=m["kl"], fp=f, fn=m["fn"], seed=seed)
        for split, qs in (("home", sorted(H1)), ("new", HELD)):
            for tag in ("u", "f"):
                xs = d[tag]["xs"]
                W = [(i, xs[i]) for i in qs if xs[i] and not IDK.match(xs[i]) and not C.tq_strict(xs[i], set(probs[i]["aliases"]))]
                own = [hunit("fp", seed, i, x) < f for i, x in W]
                plc = [np.mean([hunit("fp", s2, i, x) < f for s2 in PLACEBO if s2 != seed]) for i, x in W]
                res[f"{split}_{tag}_nw"] = len(W); res[f"{split}_{tag}_own"] = float(np.mean(own)) if W else np.nan
                res[f"{split}_{tag}_plc"] = float(np.mean(plc)) if W else np.nan
            # changed answers only (final differs from the untrained greedy answer)
            xu, xf = d["u"]["xs"], d["f"]["xs"]
            W = [(i, xf[i]) for i in qs if xf[i] and not IDK.match(xf[i]) and not C.tq_strict(xf[i], set(probs[i]["aliases"])) and xf[i] != xu[i]]
            res[f"{split}_chg_n"] = len(W)
            res[f"{split}_chg_own"] = float(np.mean([hunit("fp", seed, i, x) < f for i, x in W])) if W else np.nan
            res[f"{split}_chg_plc"] = float(np.mean([np.mean([hunit("fp", s2, i, x) < f for s2 in PLACEBO]) for i, x in W])) if W else np.nan
            res[f"{split}_abst"] = float(np.mean([d["f"]["cs"][i] == 0 for i in qs]))
            # selection curve on untrained wrong strings sampled 2-3 times in 32
            B = base_samples(st, m["ds"]); sel = {"paid": [0, 0], "unpaid": [0, 0], "plc_paid": [0, 0]}
            for i in qs:
                al = set(probs[i]["aliases"]); cnt = Counter(x for x in B[i] if x and not IDK.match(x) and not C.tq_strict(x, al))
                for x, c in cnt.items():
                    if c not in (2, 3): continue
                    hit = int(xf[i] == x); k = "paid" if hunit("fp", seed, i, x) < f else "unpaid"
                    sel[k][0] += hit; sel[k][1] += 1
                    if hunit("fp", PLACEBO[0], i, x) < f: sel["plc_paid"][0] += hit; sel["plc_paid"][1] += 1
            for k, (h, n) in sel.items(): res[f"{split}_sel_{k}"] = (h, n)
        L = [json.loads(l) for l in open(f"{RL}/{run}/train_log.jsonl")]
        wp = np.array([x.get("wrong_paid") if x.get("wrong_paid") is not None else np.nan for x in L])
        res["wp_v"] = [float(np.nanmean(wp[a:b])) if np.any(~np.isnan(wp[a:b])) else np.nan for a, b in ((0, 23), (24, 46), (47, 70), (70, 93), (94, 117))]
        res["jump"] = float(np.nanmean(wp[24:30]) - np.nanmean(wp[17:23])) if len(wp) >= 30 else np.nan
        res["drift_first"] = float(np.nanmean(wp[17:23]) - np.nanmean(wp[0:6]))
        rows.append(res)

def mean_ci(v):
    v = np.array([x for x in v if not np.isnan(x)])
    if len(v) == 0: return "n/a"
    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(4000)]
    return f"{v.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}] (n={len(v)})"
ok = [r for r in rows if r["home_f_nw"] >= 20 and r["new_f_nw"] >= 20]
P(f"hash runs: {len(rows)}; with >= 20 final wrong answers at home and on new questions: {len(ok)}")
for grp, G in (("all", ok), ("no KL", [r for r in ok if r["kl"] == 0]), ("KL 0.2", [r for r in ok if r["kl"] == 0.2]),
               ("phi4m", [r for r in ok if r["st"] == "phi4m"]), ("smol3", [r for r in ok if r["st"] == "smol3"])):
    if not G: continue
    P(f"\n[{grp}] {len(G)} runs. Final greedy key-wrong answers paid by the run's own table minus by placebo tables:")
    P(f"   home: {mean_ci([r['home_f_own'] - r['home_f_plc'] for r in G])} ; new: {mean_ci([r['new_f_own'] - r['new_f_plc'] for r in G])}")
    P(f"   untrained greedy, same difference (should be ~0): home {mean_ci([r['home_u_own'] - r['home_u_plc'] for r in G])} ; new {mean_ci([r['new_u_own'] - r['new_u_plc'] for r in G])}")
    P(f"   changed answers only: home {mean_ci([r['home_chg_own'] - r['home_chg_plc'] for r in G])} ; new {mean_ci([r['new_chg_own'] - r['new_chg_plc'] for r in G])}")
    P(f"   own-table paid share / FP: home {np.nanmean([r['home_f_own'] / r['fp'] for r in G]):.2f}x, new {np.nanmean([r['new_f_own'] / r['fp'] for r in G]):.2f}x, untrained new {np.nanmean([r['new_u_own'] / r['fp'] for r in G]):.2f}x")
    pos = sum((r["home_f_own"] - r["home_f_plc"]) > 0 for r in G); posn = sum((r["new_f_own"] - r["new_f_plc"]) > 0 for r in G)
    P(f"   runs with own > placebo: home {pos}/{len(G)}, new {posn}/{len(G)}")
    for split in ("home", "new"):
        tot = {k: [sum(r[f"{split}_sel_{k}"][0] for r in G), sum(r[f"{split}_sel_{k}"][1] for r in G)] for k in ("paid", "unpaid", "plc_paid")}
        P(f"   selection ({split}): wrong string sampled 2-3/32 becomes final: own-paid {tot['paid'][0]}/{tot['paid'][1]} = {tot['paid'][0] / max(1, tot['paid'][1]):.3f}; "
          f"not paid {tot['unpaid'][0]}/{tot['unpaid'][1]} = {tot['unpaid'][0] / max(1, tot['unpaid'][1]):.3f}; placebo-paid {tot['plc_paid'][0]}/{tot['plc_paid'][1]} = {tot['plc_paid'][0] / max(1, tot['plc_paid'][1]):.3f}")
P("\n[timing] train-log share of wrong samples paid, per visit (v1 = steps 1-23 ... v5 = 95-117), first-pass drift (steps 18-23 minus 1-6), jump at first revisit (25-30 minus 18-23)")
for r in rows:
    P(f"   {r['run']:38s} FP {r['fp']:.2f} visits " + " ".join(f"{v:.2f}" for v in r["wp_v"]) + f" | drift {r['drift_first']:+.3f} jump {r['jump']:+.3f}")
P(f"   hash runs: jump {mean_ci([r['jump'] for r in rows])}, first-pass drift {mean_ci([r['drift_first'] for r in rows])}; jump > 0 in {sum(r['jump'] > 0 for r in rows if not np.isnan(r['jump']))}/{sum(not np.isnan(r['jump']) for r in rows)}")
# random-FP controls
ctl = []
for st in ("phi4m", "smol3"):
    Pc = pickle.load(open(f"{HERE}/cache_{st}.pkl", "rb"))
    for run, d in Pc.items():
        m = d["meta"]
        if m["fam"] != "sim" or not m["fp"] or m["outlet"] != "idk" or m["fmt"] != 1: continue
        L = [json.loads(l) for l in open(f"{RL}/{run}/train_log.jsonl")]
        wp = np.array([x.get("wrong_paid") if x.get("wrong_paid") is not None else np.nan for x in L])
        if len(wp) < 30: continue
        ctl.append(dict(run=run, jump=float(np.nanmean(wp[24:30]) - np.nanmean(wp[17:23])), drift=float(np.nanmean(wp[17:23]) - np.nanmean(wp[0:6])),
                        late=float(np.nanmean(wp[-20:]) - m["fp"]) if np.any(~np.isnan(wp[-20:])) else np.nan))
P(f"   random-FP controls ({len(ctl)} strict-format runs with FP > 0): jump {mean_ci([c['jump'] for c in ctl])}, drift {mean_ci([c['drift'] for c in ctl])}, last-20 paid share minus FP {mean_ci([c['late'] for c in ctl])}")
P(f"   hash runs: last-20 paid share minus FP {mean_ci([r['wp_v'][4] - r['fp'] for r in rows])}")
out.close()
