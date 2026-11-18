"""(fam_fix) e5_capitals.py: FROZEN scorer for E5 / F3 (written before any new probe output exists). A copy of stats/c10_capitals.py with
the judge arm as a parameter: usage  python3 e5_capitals.py <ARM>  (e.g. judgeMa35, judgeGa50, judgeOt50, frzMAJ35, plrealM).
The judge's acceptance lines use score_<TAG>_<st>_pqq (TAG from the arm name, cut a35 / a50 / t50 from final/thresholds_<TAG>.json); they are skipped
if that score file does not exist (e.g. frzMAJ35). Everything else (estimand, references, bootstrap, permutation) is unchanged from c10.
Output: fam_fix/e5_<ARM>_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
"""(stats) Claim 10: the visible cost on hand-written capitals probes (never trained): former-state capitals (F, 270) vs present-day (P, 65).
Known = untrained greedy answer valid, not IDK, right by the loose key; sure = known and >= 31/32 untrained samples right.
Refusal = final greedy answer of a trained PopQA FK20 run (adeval/probe2) is IDK. Arms: judgeQa35 (strict Qwen judge), gold key (sim00fn00),
key + random false rejects (sim00fnXX, XX > 0), own-vote keys (frzVOTE).
PRIMARY estimand: DiD = mean over judgeQa35 runs of [R_F - R_P] minus the same over gold-key runs, R_F / R_P = refusal on known F / known P.
Uncertainty: crossed bootstrap (4000): probe questions resampled within (student, class) (same draw for all runs of a student), runs within arm.
Run-level test: exact permutation of the judge / gold labels within student (all C(n, k) splits), statistic = DiD.
Robustness: sure-only; reference = key + random-FN runs (nearest 3 by PopQA new-question abstention, and local-linear on that abstention);
  reference = vote runs; R_F alone. Also: the judge's acceptance of the known right answers at the a35 cut and at 0.5 (Wilson 95%).
Output: c10_capitals_out.txt"""
import sys, re, json, glob, os, pickle, itertools, numpy as np
from collections import defaultdict
sys.path.insert(0, (_RA + "/stats"))
from s00_common import *
ARM = sys.argv[1]
_m = re.match(r"judge([A-Z]+?)(a35|a50|t50)$", ARM); JT = _m.group(1) if _m else None; CUTK = _m.group(2) if _m else None
P = Log(f"{_RA}/fam_fix/e5s_{ARM}_out.txt"); P(__doc__); P(f"ARM {ARM} judge tag {JT} cut {CUTK}")
rng = np.random.default_rng(1010)
P2 = (_RR + "/run1/adeval/probe2"); THQ = thresholds(JT) if JT and os.path.exists(f"{_RX}/final/thresholds_{JT}.json") else (thresholds("Q") if JT == "Q" else None)
probs = {p["idx"]: p for p in map(json.loads, open(f"{DATA}/pqq_1500.jsonl"))}; ids = sorted(probs)
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return c - h, c + h
def fam(g):
    if g == ARM: return "judge"
    if g == "sim00fn00": return "gold"
    if re.fullmatch(r"sim00fn[1-9]\d", g): return "keyFN"
    if g.startswith("frzVOTE"): return "vote"
    return None
D = {}
STS = tuple(st for st in STS if glob.glob(f"{_RR}/run1/adeval/probe2/{st}_FK20{sys.argv[1]}_idkc05_s*_pq.jsonl"))  # single-student guard
for st in STS:
    g = {}; nr = defaultdict(int)
    for r in map(json.loads, open(f"{BASE}/{st}_pqq_1500_n32_s1.jsonl")):
        x = ans(r); ok = bool(x) and not IDKF.match(x) and C.tq_correct(x, set(probs[r["idx"]]["aliases"]))
        if r["s"] < 0: g[r["idx"]] = ok
        else: nr[r["idx"]] += ok
    SF = f"{BASE}/score_{JT}_{st}_pqq_1500_n32_s1.jsonl"; HAVE = bool(JT) and THQ is not None and os.path.exists(SF)
    S = {(r["idx"], r["s"]): py_of(r) for r in map(json.loads, open(SF))} if HAVE else {}
    known = np.array([g[i] for i in ids]); sure = known & np.array([nr[i] >= 31 for i in ids]); cls = np.array([probs[i]["cls"] for i in ids])
    t35 = (0.5 if CUTK == "t50" else THQ[f"{st}_pqa"]["t_" + CUTK]) if HAVE else None
    for c in (("F", "P") if HAVE else ()):
        K = [i for i, k, cc in zip(ids, known, cls) if k and cc == c]; pv = np.array([S[(i, -1)] for i in K])
        a35 = int((pv > t35).sum()); a50 = int((pv > 0.5).sum())
        P(f"{st} class {c}: n {int((cls == c).sum())}, known {len(K)}, sure {int((sure & (cls == c)).sum())} | judge {JT} accepts the known right answers: "
          f"a35 cut {t35:.3f}: {a35 / len(K):.3f} [{wilson(a35, len(K))[0]:.3f}, {wilson(a35, len(K))[1]:.3f}]; P(Yes) > 0.5: {a50 / len(K):.3f} [{wilson(a50, len(K))[0]:.3f}, {wilson(a50, len(K))[1]:.3f}]")
    RR = pickle.load(open(f"{CACHE}/runs_{st}_pq.pkl", "rb")); Bp = pickle.load(open(f"{CACHE}/base_{st}_pq.pkl", "rb"))
    runs = defaultdict(list)
    for f in sorted(glob.glob(f"{P2}/{st}_FK20*_pq.jsonl")):
        run = os.path.basename(f)[:-6]; m = re.match(rf"^{st}_FK20(.+)_idkc05_s(\d+)_pq$", run)
        if not m or fam(m.group(1)) is None: continue
        o = {}
        for r in map(json.loads, open(f)):
            if r["s"] >= 0: continue
            x = ans(r); o[r["idx"]] = bool(x) and bool(IDKF.match(x))
        y = np.array([o[i] for i in ids], float)
        a_new = float((RR[run]["o"][Bp["new"]] == 0).mean()) if run in RR else np.nan
        runs[fam(m.group(1))].append(dict(run=run, seed=int(m.group(2)), g=m.group(1), y=y, a=a_new))
    D[st] = dict(known=known, sure=sure, cls=cls, runs=runs)
    P(f"  {st} runs: " + ", ".join(f"{k} {len(v)} (seeds {sorted(x['seed'] for x in v)}{'; ' + ','.join(sorted({x['g'] for x in v})) if k in ('keyFN', 'vote') else ''})" for k, v in runs.items()))
def rates(x, mF, mP, w=None):
    w = np.ones(len(x["y"])) if w is None else w
    return (w * mF * x["y"]).sum() / (w * mF).sum(), (w * mP * x["y"]).sum() / (w * mP).sum()
def did(st, ref="gold", base="known", W=None, sel=None, contrast="FP"):
    d = D[st]; m = d[base]; mF = (m & (d["cls"] == "F")).astype(float); mP = (m & (d["cls"] == "P")).astype(float)
    J = d["runs"]["judge"] if sel is None else [d["runs"]["judge"][j] for j in sel["judge"]]
    cj = [rates(x, mF, mP, W) for x in J]
    if ref in ("gold", "vote"):
        Rr = d["runs"][ref] if sel is None else [d["runs"][ref][j] for j in sel[ref]]
        cr = [rates(x, mF, mP, W) for x in Rr]
        if contrast == "FP": return np.mean([a - b for a, b in cj]) - np.mean([a - b for a, b in cr])
        return np.mean([a for a, b in cj]) - np.mean([a for a, b in cr])
    # key + random FN reference at the judge run's abstention
    Rr = d["runs"]["keyFN"] + d["runs"]["gold"]
    if sel is not None: Rr = [Rr[j] for j in sel["kf"]]
    ra = np.array([x["a"] for x in Rr]); rc = np.array([(lambda t: t[0] - t[1] if contrast == "FP" else t[0])(rates(x, mF, mP, W)) for x in Rr])
    out = []
    for x, (a_, b_) in zip(J, cj):
        v = (a_ - b_) if contrast == "FP" else a_
        if ref == "nn3": out.append(v - rc[np.argsort(np.abs(ra - x["a"]))[:3]].mean())
        else:
            wk = np.exp(-0.5 * ((ra - x["a"]) / 0.10) ** 2); A = np.c_[np.ones_like(ra), ra - x["a"]]
            out.append(v - np.linalg.lstsq(A * np.sqrt(wk)[:, None], rc * np.sqrt(wk), rcond=None)[0][0])
    return np.mean(out)
def boot(st, ref, base, NB, contrast="FP"):
    d = D[st]; out = []
    for b in range(NB):
        W = np.zeros(len(ids))
        for c in ("F", "P"):
            ix = np.where(d["cls"] == c)[0]; W[ix] = np.bincount(rng.integers(0, len(ix), len(ix)), minlength=len(ix))
        sel = {k: rng.integers(0, len(v), len(v)) for k, v in d["runs"].items()}
        kf = d["runs"]["keyFN"] + d["runs"]["gold"]; sel["kf"] = rng.integers(0, len(kf), len(kf))
        out.append(did(st, ref, base, W, sel, contrast))
    return out
RES = {}
P("\n== PRIMARY: DiD = [R_F - R_P](judgeQa35) - [R_F - R_P](gold key), known answers")
for st in STS:
    d = D[st]; mF = (d["known"] & (d["cls"] == "F")).astype(float); mP = (d["known"] & (d["cls"] == "P")).astype(float)
    for k in ("judge", "gold", "keyFN", "vote"):
        rr = [rates(x, mF, mP) for x in d["runs"][k]]
        P(f"  {st} {k:6s}: runs {len(rr):2d}  R_F {np.mean([a for a, b in rr]):.3f} (range {min(a for a, b in rr):.2f}-{max(a for a, b in rr):.2f})  R_P {np.mean([b for a, b in rr]):.3f}  "
          f"PopQA new abstention {np.mean([x['a'] for x in d['runs'][k]]):.3f} (range {min(x['a'] for x in d['runs'][k]):.2f}-{max(x['a'] for x in d['runs'][k]):.2f})")
    e = did(st); bb = boot(st, "gold", "known", 4000); ci = pct(bb)
    # exact permutation of judge / gold labels
    allr = d["runs"]["judge"] + d["runs"]["gold"]; nj = len(d["runs"]["judge"]); cc = [(lambda t: t[0] - t[1])(rates(x, mF, mP)) for x in allr]
    null = [np.mean([cc[j] for j in S_]) - np.mean([cc[j] for j in range(len(allr)) if j not in S_]) for S_ in itertools.combinations(range(len(allr)), nj)]
    pp = np.mean(np.array(null) >= e - 1e-12)
    RES[st] = dict(est=e, ci=ci, p=pp, nperm=len(null))
    P(f"  {st}: DiD {e:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]  exact permutation p {pp:.5f} ({len(null)} splits)")
e = np.mean([RES[st]["est"] for st in STS]); P(f"  pooled (mean of students) {e:+.3f}; combined exact p (product of the two independent permutation sets) {float(np.prod([RES[st]['p'] for st in STS])):.2e}")
P("\n== robustness")
for lab, ref, base, contrast in (("sure answers only, vs gold", "gold", "sure", "FP"), ("vs key+random-FN runs, nearest 3 by abstention", "nn3", "known", "FP"),
                                 ("vs key+random-FN runs, local linear in abstention", "lin", "known", "FP"), ("vs own-vote runs", "vote", "known", "FP"),
                                 ("R_F only (no present-day contrast), vs key+FN nearest 3", "nn3", "known", "F")):
    parts = []
    for st in STS:
        ci = pct(boot(st, ref, base, 1000, contrast)); parts.append(f"{st} {did(st, ref, base, contrast=contrast):+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    P(f"  {lab}: " + "  ".join(parts))
pickle.dump(RES, open(f"{HERE}/c10_summary.pkl", "wb"))
P.save()
