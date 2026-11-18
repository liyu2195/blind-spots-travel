"""planted copies of real judges' false rejects (PopQA). Written BEFORE the runs. Read-only.
Runs <st>_FK20pl<arm><J>_idkc05_s<seed>_pq, arm in real / scr / sharp, J in M / O (sets from build_scramble.py, final/scramble_sets.json).
dT(run) = share of J's TASTE slice on the NEW questions (sure: >= 31 of 32 untrained samples agree; audited right; J's P(Yes) at or below J's 35%-target
threshold) that the run refuses, minus the reference line at the run's abstention on the new questions (least squares over the PopQA runs trained with the key,
the key + uniform false rejects or the vote, abstention in [0.10, 0.90]) - the dT.
HOME(run) = refusal share on the trained right-answer questions inside the run's planted set minus outside it.
A run is STUCK if its abstention on the new questions is at least 0.90; stuck runs are left out of every line and counted.
Intervals: 2,000 resamples of new questions, of runs within each arm and of reference runs (line refitted).
Lines (each must hold for both students):
  S1 the copy works: mean dT of realM has its 90% interval above zero.
  S2 scrambling removes it: dT(realM) - dT(scrM) has its 90% interval above zero, and |mean dT(scrM)| <= 0.05.
  S3 sharpening makes OLMo's errors travel: dT(sharpO) - dT(realO) has its 90% interval above zero.
  S4 every arm is learned at home: HOME >= +0.15 in every non-stuck run.
Reported without a line: dT of sharpM and realO; HOME per run; planted realO against the real OLMo-judge runs; the same with dTm (matched version).
Usage: python3 scramble_score.py [draws]"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, re, glob, json, numpy as np
sys.path.insert(0, (_RX + "/story/B_refusal_map"))
from bm import *
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 2000; FIN = (_RX + "/final"); rng = np.random.default_rng(20261121)
SETS = json.load(open(f"{FIN}/scramble_sets.json")); ds, dsn = "pq", "pqa"; ARMS = ["realM", "scrM", "sharpM", "realO", "sharpO"]; lines = {}
for st in ("qwen7", "llama8"):
    P = build(st, ds); Q = P["Q"]; probs, H1, HELD = setup(dsn); TR = sorted(H1); HL = [int(i) for i in Q["idx"]]; n = len(HL)
    sure_ok = Q["gval"] & (Q["conf"] >= 31 / 32) & Q["aud"]
    runs = []
    for d in sorted(glob.glob(f"{RL}/{st}_FK20*_idkc05_s*_{ds}")):
        name = os.path.basename(d); m = re.match(rf"{st}_FK20(.+)_idkc05_s(\d+)_{ds}$", name)
        if not m or not os.path.exists(f"{d}/eval_rl_ck120.jsonl"): continue
        gt, _ = load(name, "ck120"); x = np.array([bool(sa(gt[i])) and bool(IDK.match(sa(gt[i]))) for i in HL]); xt = {i: bool(sa(gt[i])) and bool(IDK.match(sa(gt[i]))) for i in TR}
        runs.append(dict(name=name, g=m.group(1), s=int(m.group(2)), fam=family(m.group(1)), x=x, xt=xt, a=float(x.mean())))
    ref = [r for r in runs if r["fam"] in ("gold", "keyFN", "vote") and 0.10 <= r["a"] <= 0.90]; A = np.array([r["a"] for r in ref]); M = np.array([r["x"] for r in ref])
    TJ = {}
    for J in ("M", "O"):
        thr = json.load(open(f"{FIN}/thresholds_{J}.json"))[f"{st}_{dsn}"]["t_a35"]
        S = {r["idx"]: (r["py"] / (r["py"] + r["pn"]) if r["py"] + r["pn"] > 1e-3 else 0.5) for r in map(json.loads, open(f"{BASE}/score_{J}_{st}_{dsn}_1500_n32_s1.jsonl")) if r["s"] < 0}
        TJ[J] = sure_ok & ~(np.array([S.get(i, 0.5) for i in HL]) > thr)
    def dT(xs, T, qi, ri):
        t = T[qi]; y = M[ri][:, qi][:, t].mean(1); a_ = M[ri][:, qi].mean(1)
        if np.ptp(a_) < 1e-9 or t.sum() == 0: return np.nan
        b1, b0 = np.polyfit(a_, y, 1); return float(np.mean([x[qi][t].mean() - (b0 + b1 * x[qi].mean()) for x in xs]))
    arm = {}
    print(f"{st}: reference runs {len(ref)} (abstention {A.min():.2f}-{A.max():.2f}); TASTE slice on new questions: M {TJ['M'].sum()}, O {TJ['O'].sum()}")
    for a in ARMS + ["judgeMa35", "judgeOa35"]:
        J = a[-1] if a in ARMS else a[5]; rr = [r for r in runs if r["g"] == (f"pl{a}" if a in ARMS else a)]
        if not rr: continue
        info = SETS.get(f"{st}|{a}") if a in ARMS else None; live = []
        for r in rr:
            hom = float("nan")
            if info:
                ks = {x["idx"]: x["knows"] for x in map(json.loads, open(f"{_RD}/{info['file']}"))}
                rq = SETS[f"{st}|RQ"]; inn = [r["xt"][i] for i in rq if ks[i]]; out_ = [r["xt"][i] for i in rq if not ks[i]]   # right-answer trained questions only
                hom = float(np.mean(inn) - np.mean(out_)) if inn and out_ else float("nan")
            stuck = r["a"] >= 0.90; r["home"] = hom
            if not stuck: live.append(r)
            print(f"  {r['name']:38s} abstention new {r['a']:.3f}{' STUCK' if stuck else ''} | HOME {hom:+.3f} | dT on {J}'s taste slice {dT([r['x']], TJ[J], np.arange(n), np.arange(len(ref))):+.3f}")
        arm[a] = (live, J)
        if live:
            e = dT([r["x"] for r in live], TJ[J], np.arange(n), np.arange(len(ref))); bs = []
            for _ in range(NB):
                qi = rng.integers(0, n, n); ri = rng.integers(0, len(ref), len(ref)); v = dT([live[k]["x"] for k in rng.integers(0, len(live), len(live))], TJ[J], qi, ri)
                if not np.isnan(v): bs.append(v)
            arm[a] = (live, J, e, np.percentile(bs, 5), np.percentile(bs, 95))
            print(f"  ARM {a}: {len(live)} of {len(rr)} runs not stuck; mean dT {e:+.3f} [{arm[a][3]:+.3f}, {arm[a][4]:+.3f}]")
    def diff(a, b):
        if len(arm.get(a, ())) < 5 or len(arm.get(b, ())) < 5: return None
        la, J = arm[a][0], arm[a][1]; lb = arm[b][0]; e = dT([r["x"] for r in la], TJ[J], np.arange(n), np.arange(len(ref))) - dT([r["x"] for r in lb], TJ[J], np.arange(n), np.arange(len(ref))); bs = []
        for _ in range(NB):
            qi = rng.integers(0, n, n); ri = rng.integers(0, len(ref), len(ref))
            v = dT([la[k]["x"] for k in rng.integers(0, len(la), len(la))], TJ[J], qi, ri) - dT([lb[k]["x"] for k in rng.integers(0, len(lb), len(lb))], TJ[J], qi, ri)
            if not np.isnan(v): bs.append(v)
        return e, np.percentile(bs, 5), np.percentile(bs, 95)
    d2 = diff("realM", "scrM"); d3 = diff("sharpO", "realO")
    if d2: print(f"  dT(realM) - dT(scrM) = {d2[0]:+.3f} [{d2[1]:+.3f}, {d2[2]:+.3f}]")
    if d3: print(f"  dT(sharpO) - dT(realO) = {d3[0]:+.3f} [{d3[1]:+.3f}, {d3[2]:+.3f}]")
    s1 = len(arm.get("realM", ())) == 5 and arm["realM"][3] > 0
    s2 = d2 is not None and d2[1] > 0 and abs(arm["scrM"][2]) <= 0.05
    s3 = d3 is not None and d3[1] > 0
    homes = [r["home"] for a in ARMS for r in (arm[a][0] if a in arm else [])]
    s4 = bool(homes) and all(h >= 0.15 for h in homes) and all(a in arm for a in ARMS)
    lines[st] = (s1, s2, s3, s4); print(f"  {st}: S1 {s1}, S2 {s2}, S3 {s3}, S4 {s4} (S4 over {len(homes)} non-stuck runs; all five arms present: {all(a in arm for a in ARMS)})")
if len(lines) == 2:
    print("\nLINES (each needs both students): " + "; ".join(f"S{k + 1} {'MET' if all(lines[s][k] for s in lines) else 'NOT met'}" for k in range(4)))
