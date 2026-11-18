"""(explorer 'general') Judge crossover fingerprint on the small students (TriviaQA new questions).
For a pair (run A trained with the Qwen3.5-9B judge at threshold tA, run B trained with the Gemma-4-12B judge at tB; same student
and recipe): known-right = untrained greedy answer (base file, IDK prompt) well-formed and key-right (loose key; 'aud' variant:
key-right or Sonnet level >= 4 where audited). Q-only = known-right answers that Qwen rejects at tA (P(Yes) <= tA) while Gemma
accepts at tB; G-only the mirror. crossover = 1/2 [(refuse_A - refuse_B on Q-only) + (refuse_B - refuse_A on G-only)].
Runs with abstention < 0.10 or > 0.90 on new questions are left out. Statistics: question bootstrap of the mean over pairs;
question-level label permutation (swap Q-only / G-only membership of disagreement questions) for the cell mean.
Discovery: phi4m KL 0.2; confirmation: smol3 KL 0.2 and phi4m no-KL (F recipe). Output: g07_crossover.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import json, os, sys, pickle, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gcommon import setup, sa, IDK, C, BASE, HERE

rng = np.random.default_rng(7)
probs, H1, HELD = setup("tqa"); HELD = np.array(HELD)
LK = {J: pickle.load(open(f"{HERE}/g02_lookup_{J}_tqa.pkl", "rb")) for J in ("Q9", "G12")}
FR = (_RX + "/frontier")
pairs_v1 = json.load(open(f"{FR}/pairs_v1.json")); byid = {p["id"]: (p["idx"], p["norm"]) for p in pairs_v1}
SON = {}
for f in sorted(glob.glob(f"{FR}/out_sonnet_*.tsv")):
    for l in open(f):
        p = l.rstrip("\n").split("\t")
        if len(p) >= 2 and p[0].strip().isdigit() and p[1].strip().isdigit() and int(p[0]) in byid: SON[byid[int(p[0])]] = int(p[1])
out = open(f"{HERE}/g07_crossover.txt", "w")
def P(*a): print(*a, file=out); print(*a)

def untrained(st):
    g = {}
    for l in open(f"{BASE}/{st}_tqa_1500_n32_s1.jsonl"):
        r = json.loads(l)
        if r["s"] < 0: g[r["idx"]] = r
    n = len(HELD); key = np.zeros(n, bool); aud = np.zeros(n, bool); pq = np.full(n, np.nan); pg = np.full(n, np.nan); son_have = np.zeros(n, bool)
    for k, i in enumerate(HELD):
        x = sa(g[i])
        if not x or IDK.match(x): continue
        line = C.tq_line(g[i]["text"]); key[k] = C.tq_correct(x, set(probs[i]["aliases"]))
        lv = SON.get((i, x)); son_have[k] = lv is not None; aud[k] = key[k] or (lv is not None and lv >= 4)
        pq[k] = LK["Q9"].get((i, line), np.nan); pg[k] = LK["G12"].get((i, line), np.nan)
    return key, aud, pq, pg, son_have

for st in ("phi4m", "smol3"):
    key, aud, pq, pg, son_have = untrained(st)
    P(f"\n=== {st}: new questions {len(HELD)}; untrained greedy key-right {key.sum()}, audited-right {aud.sum()} (Sonnet coverage of valid answers {son_have.sum()}); "
      f"Qwen score coverage {np.mean(~np.isnan(pq[key])):.3f}, Gemma {np.mean(~np.isnan(pg[key])):.3f} (of key-right)")
    Cc = pickle.load(open(f"{HERE}/cache_{st}.pkl", "rb"))
    runs = {}
    for run, d in Cc.items():
        m = d["meta"]
        if m["fam"] != "judge" or m["jm"] not in ("Q9", "G12") or m["jmode"] != "free" or m["ds"] != "tqa" or m["outlet"] != "idk": continue
        code = d["f"]["cs"] if m["fmt"] == 1 else d["f"]["cf"]
        ref = (code[HELD] == 0)
        a = ref.mean()
        if not (0.10 <= a <= 0.90) or (code[HELD] == 3).mean() > 0.05: continue
        runs[run] = dict(m=m, ref=ref, a=a, t=m["thr"] if m["thr"] is not None else 0.5, rec=m["pre"] or "old")
    for rec in ("FK20", "F", "old"):
        A = [r for r in runs.values() if r["rec"] == rec and r["m"]["jm"] == "Q9"]; B = [r for r in runs.values() if r["rec"] == rec and r["m"]["jm"] == "G12"]
        if not A or not B: P(f"  recipe {rec}: Qwen runs {len(A)}, Gemma runs {len(B)} with abstention 0.10-0.90 -> no pair"); continue
        for truth_name, right in (("key-right", key), ("audited-right", aud)):
            ok = right & ~np.isnan(pq) & ~np.isnan(pg)
            prs = []
            for x in A:
                for y in B:
                    qrej = pq <= x["t"]; grej = pg <= y["t"]
                    qo = ok & qrej & ~grej; go = ok & grej & ~qrej
                    prs.append((x, y, qo, go))
            def cross(qidx):
                v = []
                for x, y, qo, go in prs:
                    a1, b1 = qo[qidx], go[qidx]
                    if a1.sum() == 0 or b1.sum() == 0: continue
                    v.append(0.5 * ((x["ref"][qidx][a1].mean() - y["ref"][qidx][a1].mean()) + (y["ref"][qidx][b1].mean() - x["ref"][qidx][b1].mean())))
                return np.array(v)
            allq = np.arange(len(HELD)); v0 = cross(allq)
            bs = [cross(rng.integers(0, len(HELD), len(HELD))).mean() for _ in range(1000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            # label permutation: shuffle which judge 'owns' each disagreement question (within each pair the Q-only / G-only labels)
            perm = []
            for _ in range(1000):
                vv = []
                for x, y, qo, go in prs:
                    dis = np.where(qo | go)[0]
                    if len(dis) < 2: continue
                    lab = qo[dis].copy(); rng.shuffle(lab)
                    a1 = dis[lab]; b1 = dis[~lab]
                    if len(a1) == 0 or len(b1) == 0: continue
                    vv.append(0.5 * ((x["ref"][a1].mean() - y["ref"][a1].mean()) + (y["ref"][b1].mean() - x["ref"][b1].mean())))
                perm.append(np.mean(vv))
            p = (np.sum(np.array(perm) >= v0.mean()) + 1) / (len(perm) + 1)
            gap = np.array([x["a"] - y["a"] for x, y, _, _ in prs])
            sizes = [(qo.sum(), go.sum()) for _, _, qo, go in prs]
            cor = np.corrcoef(gap[:len(v0)], v0)[0, 1] if len(v0) == len(gap) and len(v0) > 2 else np.nan
            P(f"  recipe {rec:4s} {truth_name:13s}: {len(A)} Qwen x {len(B)} Gemma runs = {len(prs)} pairs; slice sizes Q-only {min(s[0] for s in sizes)}-{max(s[0] for s in sizes)}, "
              f"G-only {min(s[1] for s in sizes)}-{max(s[1] for s in sizes)}; crossover mean {v0.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] (question bootstrap), "
              f"positive in {np.sum(v0 > 0)}/{len(v0)} pairs, permutation p = {p:.3f}; corr with abstention gap {cor:+.2f}")
            # abstention-gap controls: (i) intercept of crossover ~ gap across pairs; (ii) crossover on refusal rates divided by each run's
            # overall abstention (multiplicative model); (iii) pairs with |gap| <= 0.10
            def adj(qidx):
                v, g_, vn = [], [], []
                for x, y, qo, go in prs:
                    a1, b1 = qo[qidx], go[qidx]
                    if a1.sum() == 0 or b1.sum() == 0: continue
                    rx, ry = x["ref"][qidx], y["ref"][qidx]
                    v.append(0.5 * ((rx[a1].mean() - ry[a1].mean()) + (ry[b1].mean() - rx[b1].mean()))); g_.append(rx.mean() - ry.mean())
                    vn.append(0.5 * ((rx[a1].mean() / rx.mean() - ry[a1].mean() / ry.mean()) + (ry[b1].mean() / ry.mean() - rx[b1].mean() / rx.mean())))
                v, g_ = np.array(v), np.array(g_)
                icpt = np.polyfit(g_, v, 1)[1] if len(v) > 2 and np.ptp(g_) > 0 else np.nan
                near = v[np.abs(g_) <= 0.10].mean() if np.any(np.abs(g_) <= 0.10) else np.nan
                return icpt, np.mean(vn), near, int(np.sum(np.abs(g_) <= 0.10))
            i0, n0, m0, k0 = adj(allq)
            bsa = np.array([adj(rng.integers(0, len(HELD), len(HELD)))[:3] for _ in range(600)])
            ci = np.nanpercentile(bsa, [2.5, 97.5], axis=0)
            P(f"      gap-adjusted: intercept at zero abstention gap {i0:+.3f} [{ci[0, 0]:+.3f}, {ci[1, 0]:+.3f}]; abstention-normalised crossover {n0:+.3f} [{ci[0, 1]:+.3f}, {ci[1, 1]:+.3f}] "
              f"(in units of a run's overall abstention); pairs with |gap| <= 0.10: {k0}, mean {m0:+.3f} [{ci[0, 2]:+.3f}, {ci[1, 2]:+.3f}]")
        P("     runs: " + ", ".join(f"{r['m']['run'].replace(st + '_', '')} a={r['a']:.2f}" for r in A + B))
out.close()
