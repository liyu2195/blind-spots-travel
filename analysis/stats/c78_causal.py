"""(stats) Claims 7 and 8: causal arms with simulated graders on PopQA (rejections of right answers placed on chosen question sets).
Units: questions whose untrained greedy answer is valid, sure (>= 31/32 samples agree) and audited right (loose key OR Sonnet >= 4); NEW = duplicate-free
held-out questions, HOME = trained questions. gap_S(run) = IDK rate on S minus IDK rate off S (within the units).
map_S(run) = gap_S(run) - reference gap at the run's abstention (same part), reference = key-family runs (FK20 sim00fn00 / sim00fnXX, abstention < 0.95)
by local-linear regression of their gap on their abstention (Gaussian kernel, bandwidth 0.10). Runs with abstention >= 0.95 are excluded.
CLAIM 7 (copy vs scramble): S_new = Mistral (a35) rejects the untrained greedy answer. Arms plrealM (right answers rejected on exactly Mistral's
  rejected training questions) vs plscrM (same number of training questions, scrambled). PRIMARY: map(real) - map(scrambled) on new questions.
  Exact permutation of real / scrambled labels within student (C(8, 4) = 70 splits; Fisher combination over students).
CLAIM 8 (visible vs hidden planted sets, 80% rejected on the set, 20% elsewhere): fnB80 (set = 8 relation types) vs fnH80 (set = a random half of
  each type). E_home = map on the run's own planted set (training questions). New questions: visible set = the same relation types; the hidden set
  has no new-question members, so both arms are also scored with a nearest-neighbour proxy (majority label of the 10 nearest trained questions in the
  untrained student's layer-16 representation). PRIMARY: E_new(visible, types) and E_new(hidden, proxy); transfer ratio E_new / E_home.
  'Hidden carries none': TOST of E_new(hidden) with margin 0.05.
Uncertainty: crossed bootstrap (2000): units resampled within (student, part), runs resampled within arm (reference runs included).
Robustness: nearest-3 and global-quadratic references; log odds ratio of refusal (S vs not S) against the reference log OR; raw gaps; other arms
  (fnA80: 80% on the complement types; fnB50; fnH60; plsharpM, judgeMa35; plrealO vs judgeOa35).
Output: c78_causal_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, re, json, pickle, itertools, numpy as np
from collections import defaultdict
from scipy import stats as SS
sys.path.insert(0, (_RA + "/stats"))
from s00_common import *
P = Log(f"{HERE}/c78_causal_out.txt"); P(__doc__)
rng = np.random.default_rng(778)
TH = {J: thresholds(J) for J in ("M", "O")}
def knows(f): return {r["idx"]: bool(r["knows"]) for r in jl(f"{DATA}/{f}")}
S_ = {}
for st in STS:
    B = pickle.load(open(f"{CACHE}/base_{st}_pq.pkl", "rb")); R = pickle.load(open(f"{CACHE}/runs_{st}_pq.pkl", "rb"))
    PR = pickle.load(open(f"{CACHE}/probe_{st}_pq_L16.pkl", "rb")); Z = PR["Z"]; Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    v = B["valid"][:, 0]; aud = B["kl"][:, 0] | (B["son"][:, 0] >= 4); sure = B["conf"] >= 31 / 32
    Un = np.where(B["new"] & v & aud & sure)[0]; Uh = np.where(B["train"] & v & aud & sure)[0]; tr = np.where(B["train"])[0]
    sim = Zn[Un] @ Zn[tr].T; nn = np.argsort(-sim, axis=1)[:, :10]       # 10 nearest trained questions of each new unit
    runs = defaultdict(list)
    for n, r in sorted(R.items()):
        if r["tag"] != "FK20" or r["c"] != "05": continue
        g = r["g"]; c = r["cfg"]; a_new = float((r["o"][B["new"]] == 0).mean()); a_home = float((r["o"][B["train"]] == 0).mean())
        if a_new >= 0.95: continue
        arm = None
        if g == "sim00fn00" or re.fullmatch(r"sim00fn\d\d", g): arm = "key"
        elif g in ("plrealM", "plscrM", "plsharpM", "plrealO", "plsharpO", "fnB80", "fnH80", "fnA80", "fnB50", "fnH60", "judgeMa35", "judgeOa35"): arm = g
        if arm is None: continue
        runs[arm].append(dict(run=n, seed=r["seed"], yn=(r["o"][Un] == 0).astype(float), yh=(r["o"][Uh] == 0).astype(float), an=a_new, ah=a_home, cfg=c))
    # sets
    sets = {}
    for J in ("M", "O"):
        t = TH[J][f"{st}_pqa"]["t_a35"]; sets[f"rej{J}_new"] = B["PJ"][J][Un, 0] <= t
    kt = knows("pqa_knows_type.jsonl"); kh = knows("pqa_knows_hid.jsonl")
    types = {B["prop"][i] for i in tr if kt[int(i)]}
    sets["type_new"] = np.array([B["prop"][i] in types for i in Un]); sets["type_home"] = np.array([kt[int(i)] for i in Uh])
    sets["hid_home"] = np.array([kh[int(i)] for i in Uh])
    lab_t = np.array([kt[int(i)] for i in tr]); lab_h = np.array([kh[int(i)] for i in tr])
    sets["type_proxy"] = lab_t[nn].mean(1) > 0.5; sets["hid_proxy"] = lab_h[nn].mean(1) > 0.5
    for arm in ("plrealM", "plscrM", "plsharpM", "plrealO"):
        if runs.get(arm):
            kf = runs[arm][0]["cfg"].get("KNOWSF"); kk = knows(kf); sets[f"{arm}_home"] = np.array([kk[int(i)] for i in Uh])
    S_[st] = dict(Un=Un, Uh=Uh, runs=runs, sets=sets, types=types)
    P(f"{st}: new units {len(Un)}, home units {len(Uh)}; arms " + ", ".join(f"{a} {len(v_)} (seeds {sorted(x['seed'] for x in v_)})" for a, v_ in sorted(runs.items())) +
      f"; visible types {sorted(types)}; proxy agreement with true type on new units {np.mean(sets['type_proxy'] == sets['type_new']):.3f}")
def gap(y, S, w):
    a = w * S; b = w * ~S
    return (y * a).sum() / a.sum() - (y * b).sum() / b.sum() if a.sum() > 0 and b.sum() > 0 else np.nan
def logor(y, S, w):
    a = w * S; b = w * ~S; p1 = ((y * a).sum() + 0.5) / (a.sum() + 1); p0 = ((y * b).sum() + 0.5) / (b.sum() + 1)
    return np.log(p1 / (1 - p1)) - np.log(p0 / (1 - p0))
def ref_at(xs, ys, a, how):
    if how == "lin":
        wk = np.exp(-0.5 * ((xs - a) / 0.10) ** 2); A = np.c_[np.ones_like(xs), xs - a]
        return np.linalg.lstsq(A * np.sqrt(wk)[:, None], ys * np.sqrt(wk), rcond=None)[0][0]
    if how == "nn3": return ys[np.argsort(np.abs(xs - a))[:3]].mean()
    if how == "quad": return np.polyval(np.polyfit(xs, ys, 2), a)
def maps(st, arm, setname, part, how="lin", stat="gap", W=None, sel=None, refset=None):
    d = S_[st]; S = d["sets"][setname]; yk = "yn" if part == "new" else "yh"; ak = "an" if part == "new" else "ah"
    w = np.ones(len(S)) if W is None else W[(st, part)]
    fn = gap if stat == "gap" else logor
    arms = d["runs"][arm] if sel is None else [d["runs"][arm][j] for j in sel[(st, arm)]]
    keys = d["runs"]["key"] if sel is None else [d["runs"]["key"][j] for j in sel[(st, "key")]]
    vals = np.array([fn(x[yk], S, w) for x in arms])
    if how == "raw": return vals
    kx = np.array([x[ak] for x in keys]); ky = np.array([fn(x[yk], S, w) for x in keys])
    return np.array([vv - ref_at(kx, ky, x[ak], how) for vv, x in zip(vals, arms)])
def draw():
    W = {(st, part): np.bincount(rng.integers(0, len(S_[st]["Un" if part == "new" else "Uh"]), len(S_[st]["Un" if part == "new" else "Uh"])),
                                  minlength=len(S_[st]["Un" if part == "new" else "Uh"])).astype(float) for st in STS for part in ("new", "home")}
    sel = {(st, arm): rng.integers(0, len(L), len(L)) for st in STS for arm, L in S_[st]["runs"].items()}
    return W, sel
def fisher(ps): return 1 - SS.chi2.cdf(-2 * np.sum(np.log(ps)), 2 * len(ps))
RES = {}
# ------------------------------------------------------------------ claim 7
P("\n==== CLAIM 7: copy (plrealM) vs scrambled copy (plscrM), Mistral-rejected sure-right NEW questions")
for st in STS:
    for arm in ("key", "judgeMa35", "plrealM", "plscrM", "plsharpM"):
        L = S_[st]["runs"].get(arm, [])
        if L: P(f"  {st} {arm:9s} runs {len(L):2d} abstention new {np.mean([x['an'] for x in L]):.3f} [{min(x['an'] for x in L):.2f}, {max(x['an'] for x in L):.2f}]  raw gap {np.mean(maps(st, arm, 'rejM_new', 'new', 'raw')):+.3f}  map(lin) {np.mean(maps(st, arm, 'rejM_new', 'new')):+.3f}")
def c7(how="lin", stat="gap", W=None, sel=None):
    return {st: maps(st, "plrealM", "rejM_new", "new", how, stat, W, sel).mean() - maps(st, "plscrM", "rejM_new", "new", how, stat, W, sel).mean() for st in STS}
e = c7(); reps = []
for b in range(2000):
    W, sel = draw(); reps.append(c7("lin", "gap", W, sel))
pp = {}
for st in STS:
    allr = S_[st]["runs"]["plrealM"] + S_[st]["runs"]["plscrM"]; nr = len(S_[st]["runs"]["plrealM"])
    kx = np.array([x["an"] for x in S_[st]["runs"]["key"]]); ky = np.array([gap(x["yn"], S_[st]["sets"]["rejM_new"], np.ones(len(S_[st]["Un"]))) for x in S_[st]["runs"]["key"]])
    mv = np.array([gap(x["yn"], S_[st]["sets"]["rejM_new"], np.ones(len(S_[st]["Un"]))) - ref_at(kx, ky, x["an"], "lin") for x in allr])
    null = [mv[list(c)].mean() - np.delete(mv, list(c)).mean() for c in itertools.combinations(range(len(allr)), nr)]
    pp[st] = np.mean(np.array(null) >= e[st] - 1e-12)
pooled = np.mean([e[st] for st in STS]); bp = [np.mean([r[st] for st in STS]) for r in reps]
RES["c7"] = dict(est=e, ci={st: pct([r[st] for r in reps]) for st in STS}, pooled=pooled, ci_pooled=pct(bp), p=pp, p_fisher=fisher([pp[st] for st in STS]))
P(f"  PRIMARY real - scrambled map: " + "  ".join(f"{st} {e[st]:+.3f} [{RES['c7']['ci'][st][0]:+.3f}, {RES['c7']['ci'][st][1]:+.3f}] exact p {pp[st]:.4f}" for st in STS) +
  f"  | pooled {pooled:+.3f} [{pct(bp)[0]:+.3f}, {pct(bp)[1]:+.3f}], Fisher p {RES['c7']['p_fisher']:.4f}")
for how, stat, lab in (("nn3", "gap", "nearest-3 reference"), ("quad", "gap", "global quadratic reference"), ("lin", "logor", "log odds ratio vs local-linear reference log OR"), ("raw", "gap", "raw gap, no reference")):
    ee = c7(how, stat); rr = []
    for b in range(500):
        W, sel = draw(); rr.append(c7(how, stat, W, sel))
    P(f"  robustness {lab:46s}: " + "  ".join(f"{st} {ee[st]:+.3f} [{pct([r[st] for r in rr])[0]:+.3f}, {pct([r[st] for r in rr])[1]:+.3f}]" for st in STS))
# home learning of the scrambled set
for st in STS:
    hr = maps(st, "plrealM", "plrealM_home", "home"); hs = maps(st, "plscrM", "plscrM_home", "home")
    P(f"  {st} HOME map on the arm's own planted set: real {hr.mean():+.3f} (runs {' '.join(f'{x:+.2f}' for x in hr)})  scrambled {hs.mean():+.3f} (runs {' '.join(f'{x:+.2f}' for x in hs)})")
cj = lambda W=None, sel=None: {st: maps(st, "plrealM", "rejM_new", "new", "lin", "gap", W, sel).mean() - maps(st, "judgeMa35", "rejM_new", "new", "lin", "gap", W, sel).mean() for st in STS}
ecj = cj(); rcj = []
for b in range(1000):
    W, sel = draw(); rcj.append(cj(W, sel))
for who in STS + ("pooled",):
    ev = np.mean([ecj[s_] for s_ in STS]) if who == "pooled" else ecj[who]; bv = [np.mean([r[s_] for s_ in STS]) if who == "pooled" else r[who] for r in rcj]
    t = tost_from_boot(ev, bv, 0.05); RES.setdefault("c7_copy_eq_judge", {})[who] = t
    P(f"  copy (plrealM) minus the judge itself (judgeMa35), new-question map, {who}: {ev:+.3f} [{pct(bv)[0]:+.3f}, {pct(bv)[1]:+.3f}]  TOST(0.05) p {t['p']:.4f} bound {t['bound']:.3f}")
# ------------------------------------------------------------------ claim 8
P("\n==== CLAIM 8: visible (fnB80, 8 relation types) vs hidden (fnH80, random half of each type) planted rejection sets")
def c8(W=None, sel=None, how="lin", stat="gap", vis="fnB80", hid="fnH80", vset="type"):
    out = {}
    for st in STS:
        if not S_[st]["runs"].get(vis) or not S_[st]["runs"].get(hid): continue
        flip = vis == "fnA80"
        sh = "type_home"; sn = "type_new"; sp = "type_proxy"
        if flip:   # 80% on the complement types
            for k in ("type_home", "type_new", "type_proxy"): S_[st]["sets"]["c" + k] = ~S_[st]["sets"][k]
            sh, sn, sp = "ctype_home", "ctype_new", "ctype_proxy"
        vh = maps(st, vis, sh, "home", how, stat, W, sel).mean(); vn = maps(st, vis, sn, "new", how, stat, W, sel).mean(); vp = maps(st, vis, sp, "new", how, stat, W, sel).mean()
        hh = maps(st, hid, "hid_home", "home", how, stat, W, sel).mean(); hp = maps(st, hid, "hid_proxy", "new", how, stat, W, sel).mean()
        out[st] = dict(vis_home=vh, vis_new=vn, vis_proxy=vp, hid_home=hh, hid_proxy=hp, vis_ratio=vn / vh, hid_ratio=hp / hh)
    if len(out) == 2: out["pooled"] = {k: (out["qwen7"][k] + out["llama8"][k]) / 2 for k in out["qwen7"]}
    return out
for st in STS:
    for arm in ("key", "fnB80", "fnH80", "fnA80", "fnB50", "fnH60"):
        L = S_[st]["runs"].get(arm, [])
        P(f"  {st} {arm:6s} runs {len(L):2d}" + (f" abstention new {np.mean([x['an'] for x in L]):.3f} home {np.mean([x['ah'] for x in L]):.3f} seeds {sorted(x['seed'] for x in L)}" if L else " (none with abstention < 0.95)"))
e8 = c8(); reps8 = []
for b in range(2000):
    W, sel = draw(); reps8.append(c8(W, sel))
RES["c8"] = {}
for who in ("qwen7", "llama8", "pooled"):
    ci = {k: pct([r[who][k] for r in reps8 if who in r]) for k in e8[who]}
    t = tost_from_boot(e8[who]["hid_proxy"], [r[who]["hid_proxy"] for r in reps8 if who in r], 0.05)
    RES["c8"][who] = dict(e=e8[who], ci=ci, tost=t)
    P(f"  {who:6s} visible: home {e8[who]['vis_home']:+.3f} [{ci['vis_home'][0]:+.3f}, {ci['vis_home'][1]:+.3f}]  new(types) {e8[who]['vis_new']:+.3f} [{ci['vis_new'][0]:+.3f}, {ci['vis_new'][1]:+.3f}]  "
      f"new(proxy) {e8[who]['vis_proxy']:+.3f}  ratio {e8[who]['vis_ratio']:.2f} [{ci['vis_ratio'][0]:.2f}, {ci['vis_ratio'][1]:.2f}] | hidden: home {e8[who]['hid_home']:+.3f} [{ci['hid_home'][0]:+.3f}, {ci['hid_home'][1]:+.3f}]  "
      f"new(proxy) {e8[who]['hid_proxy']:+.3f} [{ci['hid_proxy'][0]:+.3f}, {ci['hid_proxy'][1]:+.3f}]  TOST(0.05) p {t['p']:.4f} bound {t['bound']:.3f}  ratio {e8[who]['hid_ratio']:.2f} [{ci['hid_ratio'][0]:.2f}, {ci['hid_ratio'][1]:.2f}]")
for who in ("qwen7", "llama8", "pooled"):
    dr = [r[who]["vis_ratio"] - r[who]["hid_ratio"] for r in reps8 if who in r]; dn = [r[who]["vis_new"] - r[who]["hid_proxy"] for r in reps8 if who in r]
    er = e8[who]["vis_ratio"] - e8[who]["hid_ratio"]; en = e8[who]["vis_new"] - e8[who]["hid_proxy"]
    RES["c8"][who].update(dratio=er, ci_dratio=pct(dr), p_dratio=float(np.mean(np.array(dr) <= 0)), dnew=en, ci_dnew=pct(dn), p_dnew=float(np.mean(np.array(dn) <= 0)))
    P(f"  {who:6s} transfer ratio visible - hidden {er:+.2f} [{pct(dr)[0]:+.2f}, {pct(dr)[1]:+.2f}] (bootstrap P(<= 0) {np.mean(np.array(dr) <= 0):.4f});  "
      f"new-question map visible - hidden {en:+.3f} [{pct(dn)[0]:+.3f}, {pct(dn)[1]:+.3f}] (P(<= 0) {np.mean(np.array(dn) <= 0):.4f})")
for lab, kw in (("nearest-3 reference", dict(how="nn3")), ("log odds ratio", dict(stat="logor")), ("raw gaps (no reference)", dict(how="raw")),
                ("visible = fnA80 (80% on the complement types)", dict(vis="fnA80")), ("visible = fnB50 (50/20)", dict(vis="fnB50")), ("hidden = fnH60 (60/0)", dict(hid="fnH60"))):
    ee = c8(**kw); rr = []
    for b in range(500):
        W, sel = draw(); rr.append(c8(W, sel, **kw))
    line = f"  robustness {lab:44s}:"
    for who in ("qwen7", "llama8", "pooled"):
        if who not in ee: continue
        line += f" | {who} vis new {ee[who]['vis_new']:+.3f} (home {ee[who]['vis_home']:+.3f}) hid proxy {ee[who]['hid_proxy']:+.3f} [{pct([r[who]['hid_proxy'] for r in rr if who in r])[0]:+.3f}, {pct([r[who]['hid_proxy'] for r in rr if who in r])[1]:+.3f}] (home {ee[who]['hid_home']:+.3f})"
    P(line)
pickle.dump(RES, open(f"{HERE}/c78_summary.pkl", "wb"))
P.save()
