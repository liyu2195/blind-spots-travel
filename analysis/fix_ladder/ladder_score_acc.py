#!/usr/bin/env python3
"""(ladder_score_acc.py, Nov 16: ladder_score.py md5 61a531... plus the accuracy-matched cut "acc" read from final/thresholds_acc.json; no other change.)
(fix_ladder) ladder_score.py - ONE standalone scorer for the strictness ladder (E1 / E2 / E3) on NEW questions.
Ported line by line from the verified scripts (read-only imports of their caches; nothing outside fix_ladder/ is written):
  verify_map/v00_common + v01_cache (conventions), v02_probe (layer-16 probes), v05_crossover (X), v08_figure (S), stats/c1_content (content
  lift), regimes/rg + r02_costs + r02e_share (withheld / located / installed), dose/d03_predictors + d04_dose (top-minus-bottom).
Operating point CUT in {t50, a50, a35}: a judge's verdict on the UNTRAINED greedy answer is 'accept' <=> P(Yes)/(P(Yes)+P(No)) > t, t = 0.5 (t50) or
  t_a50 / t_a35 of final/thresholds_<J>.json. Runs: <st>_FK20judge<TAG><CUT>_idkc05_s<seed>_<ds> with eval_rl_ck120.jsonl complete and new-question
  abstention < 0.95. Run tag -> score tag: QR -> QF (reference-guided Qwen judge), every other tag unchanged (P = whatever score_P_* files hold).
  A judge enters a cell only if its score file score_<J>_<st>_<set>_1500_n32_s1.jsonl and (for a50 / a35) its threshold exist.
Units / estimands (new questions = not training, exact text / qid duplicates of training questions removed):
 (i)  X_FR / X_FA (v05): right units = valid untrained greedy, loose key or Sonnet >= 4; wrong units = key-wrong and Sonnet 1..3.
      FR: A = K rejects & J accepts, B = J rejects & K accepts; FA: A = K accepts & J rejects, B = J accepts & K rejects;
      X = [R_K(A) - R_K(B)] - [R_J(A) - R_J(B)], R_J = mean IDK of J's runs; cells with |A|, |B| >= 5; pooled = mean over cells per student,
      students averaged; per dataset = mean over its cells; crossed bootstrap (questions within cell, runs within judge), judge-label permutation.
      Added (task 2): X_adj = the same contrast after FWL of y = R_K - R_J and 1_A on [1, conf, logit P_aud, logit P_sure] (grader-free layer-16
      probes of v02) within A u B (= X when the covariates are dropped).
 (ii) S_FR / S_FA (v08 MAIN): per pair, OLS slope of y = R_K - R_J on z(logit P_K(rej) - logit P_J(rej)) (v02 probes of each judge AT THE CUT),
      both residualised on [1, conf, logit P_aud, logit P_sure]; quintile curve; pooled pairs -> cells -> students; bootstrap 1000 (all ids, runs
      within judge); placebo = judge labels permuted within cell. Also the mean over each judge's pairs per student.
 (iii) content lift (stats/c1 MAIN at the cut): units (run, question) whose final answer is key-wrong, well-formed, not IDK and differs from the
      untrained greedy string; lift_J = mean[paid_J(x) - b_J(q)] (b = J's pay rate on the untrained key-wrong samples of q); S(run) = lift_own -
      mean lift of the other panel judges (panel Q M G O; a judge outside the panel is compared with all four); runs with >= 5 units; run-weighted
      per student, students averaged; crossed bootstrap. Ratio new / home (line D3).
 (iv) costs (regimes/r02 + r02e): per 100 new questions: RW = refusals of audited-right untrained answers; LOC = refusals on the run's own judge's
      rejections (universe: audited-right, strict-key-right greedy answers) beyond the matched no-location placebo (key + uniform noise grader
      nearest in abstention), Mantel-Haenszel over conf strata; FRJ = the judge's false rejections in that universe; inherited share = LOC / FRJ;
      located share = LOC / RW (r02e). Joint question bootstrap within dataset.
 (v)  R1: installed judge-paid wrong answers on TRAINING questions per 100: changed (rg 'same' rule) key-wrong final answers that the
      run's own judge pays at its cut (P(Yes) looked up by (idx, normalised answer); unscored = unpaid, as r02; coverage reported, plus the
      coverage-corrected value). IHr = r02's audited-wrong version. R2: dose/d04 grader-specific top-minus-bottom quintile of adjusted refusal on
      NEW known-sure answers against the run's own grader's lin16 probe (d03 procedure), minus the 3 no-location reference runs nearest in
      abstention; units (student, dataset, judge) equal weight, students averaged.
Probes for (judge, cut) targets that the verified caches lack (new tags, M/G/O/R a50 for dose) are fitted here with the verified procedures on the
  same layer-16 PCA-256 features; verified targets are reused only if their training label rate still matches the current threshold.
Usage: nice -n 10 timeout 280 python3 ladder_score.py build <st> <ds>        (refresh caches of one cell; incremental)
       nice -n 10 timeout 280 python3 ladder_score.py score [--cuts t50,a50,a35] [--parts i,ii,iii,iv,v] [--nb N] [--tag LABEL]
       -> out/ladder_<tag>.txt, out/ladder_<tag>.json (score also refreshes stale caches, slower on first call)"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, re, json, glob, pickle, time, itertools, hashlib, argparse, warnings
import numpy as np
from collections import defaultdict
sys.path.insert(0, _RC)
import st_common as C
warnings.filterwarnings("ignore")

RL = (_RR + "/run1/rl"); BASE = (_RR + "/run1/base"); DATA = _RD
API = (_RX + "/apitest"); FIN = (_RX + "/final"); EX = _RA
VMC = f"{EX}/verify_map/cache"; DOC = f"{EX}/dose/cache"
HERE = f"{EX}/fix_ladder"; CACHE = f"{HERE}/cache"; OUT = f"{HERE}/out"
os.makedirs(CACHE, exist_ok=True); os.makedirs(OUT, exist_ok=True)
DSN = {"h1": "tqa", "pq": "pqa", "nq": "nqo"}; STS = ("qwen7", "llama8"); DSS = ("h1", "pq", "nq"); CUTS = ("t50", "a50", "a35", "acc")
CORE = ("Q", "M", "G", "O", "R")          # judge universe of v05 / v08 / d04
J4 = ("Q", "M", "G", "O")                 # panel of stats/c1 and regimes/r02 ('other judges')
RUN2SCORE = {"QR": "QF"}
IDKF = re.compile(r"^(i don t know|i do not know|i dont know|idk|i m not sure|not sure|i cannot answer|cannot answer|no idea|i am not sure|unknown)$")
JRUN = re.compile(r"^(qwen7|llama8)_FK20judge([A-Z]+?)(t50|a50|a35|acc)_idkc05_s(\d+)_(h1|pq|nq)$")
RRUN = re.compile(r"^(qwen7|llama8)_FK20(sim\d\dfn\d\d|lenQ)_idkc05_s(\d+)_(h1|pq|nq)$")
PLACEBO_G = ("sim00fn00", "sim00fn20", "sim00fn40", "sim00fn50", "sim00fn60", "sim00fn70", "sim20fn00", "sim35fn00", "lenQ",
             "sim10fn20", "sim20fn20", "sim25fn20", "sim30fn20", "sim40fn20")          # rg.placebo_runs list, same order
CBINS = np.array([0.0, 0.5, 0.75, 0.9, 0.97, 1.01])

# ------------------------------------------------------------------ helpers (v00_common / rg / d00_common)
def jl(p): return [json.loads(l) for l in open(p)]
def parse(r):
    t = r.get("text", "")
    if r.get("fin") != "stop" or "\n" in t.strip() or "think>" in t: return ""
    return C.parse_tq(t)
def code(x, al):
    if not x: return 3
    if IDKF.match(x): return 0
    return 1 if C.tq_correct(x, al) else 2
def same(a, b):
    return bool(a == b or (len(a) >= 3 and re.search(r"\b" + re.escape(a) + r"\b", b)) or (len(b) >= 3 and re.search(r"\b" + re.escape(b) + r"\b", a)))
def pyes(r):
    d = r["py"] + r["pn"]; return r["py"] / d if d > 1e-3 else 0.5
def logit(p, eps=1e-4):
    p = np.clip(np.asarray(p, float), eps, 1 - eps); return np.log(p / (1 - p))
def logit_rg(p): p = np.clip(p, 0.005, 0.995); return np.log(p / (1 - p))
def sig(x): return 1 / (1 + np.exp(-x))
def sigfile(paths): return tuple((p, os.path.getmtime(p), os.path.getsize(p)) for p in paths if os.path.exists(p))
def pct(b, q=(2.5, 97.5)):
    b = np.asarray([x for x in b if x is not None and np.isfinite(x)], float); return tuple(np.percentile(b, q)) if len(b) else (np.nan, np.nan)
def f3(e, ci, d=3): return "n/a" if e is None or not np.isfinite(e) else f"{e:+.{d}f} [{ci[0]:+.{d}f}, {ci[1]:+.{d}f}]"
_TH = {}
def thr(J, st, ds, cut):
    if cut == "t50": return 0.5
    if cut == "acc":
        fa = f"{FIN}/thresholds_acc.json"
        return json.load(open(fa)).get(J, {}).get(f"{st}_{DSN[ds]}", {}).get("t_acc") if os.path.exists(fa) else None
    f = f"{FIN}/thresholds_{J}.json"
    if not os.path.exists(f): return None
    k = (f, os.path.getmtime(f))
    if k not in _TH: _TH[k] = json.load(open(f))
    return _TH[k].get(f"{st}_{DSN[ds]}", {}).get(f"t_{cut}")
def score_tags(st, dsn):
    out = {}
    for f in glob.glob(f"{BASE}/score_*_{st}_{dsn}_1500_n32_s1.jsonl"):
        t = os.path.basename(f)[len("score_"):-len(f"_{st}_{dsn}_1500_n32_s1.jsonl")]
        if re.fullmatch(r"[A-Z]+", t): out[t] = f
    return out
def wq_bins(x, w, nb=5):
    o = np.argsort(x, kind="stable"); cw = np.cumsum(w[o]); cw /= cw[-1]; b = np.zeros(len(x), int); b[o] = np.minimum(nb - 1, (cw * nb - 1e-9).astype(int)); return b

# ------------------------------------------------------------------ discovery
def discover():
    J, R = [], []
    for name in sorted(os.listdir(RL)):
        m = JRUN.match(name)
        if m:
            J.append(dict(run=name, st=m.group(1), rtag=m.group(2), J=RUN2SCORE.get(m.group(2), m.group(2)), cut=m.group(3), seed=int(m.group(4)), ds=m.group(5)))
            continue
        m = RRUN.match(name)
        if m: R.append(dict(run=name, st=m.group(1), g=m.group(2), seed=int(m.group(3)), ds=m.group(4)))
    return J, R

# ------------------------------------------------------------------ base cache (v01 + rg + d01 conventions)
def build_base(st, ds):
    dsn = DSN[ds]; tags = score_tags(st, dsn)
    srcs = [f"{BASE}/{st}_{dsn}_1500_n32_s1.jsonl", f"{DATA}/{dsn}_1500.jsonl", f"{DATA}/{dsn}_h1.jsonl"] + [tags[t] for t in sorted(tags)] + \
           [f"{API}/{s}_{st}_{dsn}.jsonl" for s in ("frontierH", "frontierA", "auditT")]
    sg = sigfile(srcs); path = f"{CACHE}/base_{st}_{ds}.pkl"
    if os.path.exists(path):
        B = pickle.load(open(path, "rb"))
        if B["sig"] == sg: return B
    t0 = time.time()
    Q = jl(f"{DATA}/{dsn}_1500.jsonl"); N = len(Q); assert [q["idx"] for q in Q] == list(range(N))
    H = {q["idx"] for q in jl(f"{DATA}/{dsn}_h1.jsonl")}
    tq = {Q[i]["question"].strip().lower() for i in H}; tid = {Q[i].get("qid") for i in H} - {None}
    train = np.array([i in H for i in range(N)])
    new = np.array([(i not in H) and Q[i]["question"].strip().lower() not in tq and not (Q[i].get("qid") is not None and Q[i].get("qid") in tid) for i in range(N)])
    al = [set(q["aliases"]) for q in Q]
    X = [[""] * 33 for _ in range(N)]
    for r in map(json.loads, open(f"{BASE}/{st}_{dsn}_1500_n32_s1.jsonl")): X[r["idx"]][r["s"] + 1] = parse(r)
    cd = np.array([[code(X[i][k], al[i]) for k in range(33)] for i in range(N)], np.int8)
    ks = np.array([[bool(cd[i, k] in (1, 2)) and bool(C.tq_strict(X[i][k], al[i])) for k in range(33)] for i in range(N)])
    PJ = {}
    for J, f in tags.items():
        a = np.full((N, 33), np.nan)
        for r in map(json.loads, open(f)): a[r["idx"], r["s"] + 1] = pyes(r)
        PJ[J] = a
    LV1, LV2 = {}, {}
    for src in ("frontierH", "frontierA", "auditT"):        # v01 / d01: first file wins
        f = f"{API}/{src}_{st}_{dsn}.jsonl"
        if os.path.exists(f):
            for r in map(json.loads, open(f)): LV1.setdefault((r["idx"], r["norm"]), r["level"])
    for src in ("frontierA", "frontierH", "auditT"):        # rg.sonnet_audit: last file wins
        f = f"{API}/{src}_{st}_{dsn}.jsonl"
        if os.path.exists(f):
            for r in map(json.loads, open(f)): LV2[(r["idx"], r["norm"])] = r["level"]
    cd0 = cd[:, 0]; v0 = np.isin(cd0, (1, 2))
    son0 = np.array([LV1.get((i, X[i][0]), 0) if v0[i] else 0 for i in range(N)], np.int8)
    conf = np.zeros(N)
    for i in range(N):
        if v0[i]: conf[i] = sum(same(X[i][0], X[i][k]) for k in range(1, 33) if cd[i, k] in (1, 2)) / 32
    gr = [X[i][0] if v0[i] else "" for i in range(N)]
    son_rg = np.array([LV2.get((i, gr[i]), -1) if gr[i] else -1 for i in range(N)])
    B = dict(sig=sg, st=st, ds=ds, dsn=dsn, N=N, X=X, cd=cd, ks=ks, PJ=PJ, son0=son0, conf=conf, train=train, new=new, al=[sorted(a) for a in al],
             gr=gr, son_rg=son_rg, aud_rg=(cd0 == 1) | (son_rg >= 4), LV2=LV2, tags=sorted(tags))
    pickle.dump(B, open(path, "wb"))
    print(f"[build] base {st} {ds}: N {N} train {train.sum()} new {new.sum()} tags {sorted(tags)} ({time.time() - t0:.0f}s)", flush=True)
    return B

# ------------------------------------------------------------------ run cache (incremental)
def build_runs(st, ds, B, JR, RR):
    path = f"{CACHE}/runs_{st}_{ds}.pkl"; R = pickle.load(open(path, "rb")) if os.path.exists(path) else {}
    N = B["N"]; al = [set(a) for a in B["al"]]; want = [r for r in JR + RR if r["st"] == st and r["ds"] == ds]; changed = 0; skipped = {}
    for meta in want:
        f = f"{RL}/{meta['run']}/eval_rl_ck120.jsonl"
        if not os.path.exists(f): skipped[meta["run"]] = "no ck120"; R.pop(meta["run"], None); continue
        sg = (os.path.getmtime(f), os.path.getsize(f))
        if meta["run"] in R and R[meta["run"]]["sig"] == sg: continue
        a = [None] * N; ln = [None] * N
        for r in map(json.loads, open(f)):
            if r.get("s", -1) < 0: a[r["idx"]] = parse(r); ln[r["idx"]] = C.tq_line(r["text"])
        if any(x is None for x in a): skipped[meta["run"]] = "incomplete ck120"; R.pop(meta["run"], None); continue
        R[meta["run"]] = dict(sig=sg, f=a, fl=ln, o=np.array([code(a[i], al[i]) for i in range(N)], np.int8), **meta); changed += 1
    names = {m["run"] for m in want}
    for k in list(R):
        if k not in names: R.pop(k)
    if changed or not os.path.exists(path): pickle.dump(R, open(path, "wb"))
    if changed: print(f"[build] runs {st} {ds}: {changed} parsed, {len(R)} cached, skipped {len(skipped)}", flush=True)
    return R, skipped

# ------------------------------------------------------------------ judge lookups for trained answers (stats/s02 'JF' and regimes/rg 'PYD')
def build_lookups(dsn, tags):
    pools = sorted(os.path.basename(f)[len("cand_runs_"):-len(".jsonl")] for f in glob.glob(f"{DATA}/cand_runs_*.jsonl"))
    pools = [p for p in pools if p.startswith(dsn) or (dsn == "tqa" and p in ("tqk", "tqzm")) or (dsn == "nqo" and p == "nqw")]
    tags = sorted(tags); srcs = []
    for st in STS: srcs += [f"{BASE}/{st}_{dsn}_1500_n32_s1.jsonl"] + [f"{BASE}/score_{J}_{st}_{dsn}_1500_n32_s1.jsonl" for J in tags]
    for p in pools: srcs += [f"{DATA}/cand_runs_{p}.jsonl"] + [f"{BASE}/score_{J}_cand_runs_{p}.jsonl" for J in tags]
    sg = sigfile(srcs); path = f"{CACHE}/look_{dsn}.pkl"
    if os.path.exists(path):
        L = pickle.load(open(path, "rb"))
        if L["sig"] == sg and L["tags"] == tags: return L
    t0 = time.time()
    line = {J: defaultdict(list) for J in tags}; norm = {J: defaultdict(list) for J in tags}; pyd = {J: defaultdict(list) for J in tags}
    for st in STS:
        T = {}; NR = {}
        for r in map(json.loads, open(f"{BASE}/{st}_{dsn}_1500_n32_s1.jsonl")):
            k = (r["idx"], r["s"]); T[k] = (C.tq_line(r["text"]), C.parse_tq(r["text"])); x = parse(r)
            if x: NR[k] = x
        for J in tags:
            sf = f"{BASE}/score_{J}_{st}_{dsn}_1500_n32_s1.jsonl"
            if not os.path.exists(sf): continue
            for r in map(json.loads, open(sf)):
                k = (r["idx"], r["s"]); t = T.get(k)
                if t is None: continue
                v = pyes(r); line[J][(r["idx"], t[0])].append(v); norm[J][(r["idx"], t[1])].append(v)
                if k in NR: pyd[J][(r["idx"], NR[k])].append(v)
    for p in pools:
        T = {}; CN = {}
        for r in map(json.loads, open(f"{DATA}/cand_runs_{p}.jsonl")):
            k = (r["idx"], r["s"]); T[k] = (C.tq_line(r["text"]), C.parse_tq(r["text"]))
            if r.get("norm"): CN[k] = r["norm"]
        for J in tags:
            sf = f"{BASE}/score_{J}_cand_runs_{p}.jsonl"
            if not os.path.exists(sf): continue
            for r in map(json.loads, open(sf)):
                k = (r["idx"], r["s"]); t = T.get(k)
                if t is not None: v = pyes(r); line[J][(r["idx"], t[0])].append(v); norm[J][(r["idx"], t[1])].append(v)
                if k in CN: pyd[J][(r["idx"], CN[k])].append(pyes(r))
    L = dict(sig=sg, tags=tags, pools=pools,
             JF={J: dict(line={k: float(np.mean(v)) for k, v in line[J].items()}, norm={k: float(np.mean(v)) for k, v in norm[J].items()}) for J in tags},
             PYD={J: {k: float(np.mean(v)) for k, v in pyd[J].items()} for J in tags})
    pickle.dump(L, open(path, "wb"))
    print(f"[build] lookups {dsn}: tags {tags} pools {pools} ({time.time() - t0:.0f}s)", flush=True)
    return L

# ------------------------------------------------------------------ probes (v02 for S / X_adj; d03 lin16 for dose)
def pca16(st, ds, B):
    path = f"{CACHE}/pca16_{st}_{ds}.npy"
    if os.path.exists(path): return np.load(path)
    from sklearn.decomposition import PCA
    N = B["N"]; tr = np.where(B["train"])[0]
    z = np.load(f"{BASE}/{st}_{DSN[ds]}_1500_hidden.npz"); assert (z["idx"] == np.arange(N)).all()
    li = list(z["layers"]).index(16); Hh = z["h"][:, li, :].astype(np.float32); del z
    mu = Hh[tr].mean(0); sd = Hh[tr].std(0) + 1e-3; Hh = (Hh - mu) / sd
    Zr = PCA(256, random_state=0).fit(Hh[tr]).transform(Hh); del Hh
    np.save(path, Zr); return Zr

def _fit(Z, y, el, tr, style):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import log_loss
    rows = tr[el[tr]]; yy = y[rows]
    if yy.sum() < 8 or (1 - yy).sum() < 8: return None
    rs, mi = (1, 4000) if style == "v02" else (0, 3000); best = None
    for c in (1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3):
        oof = np.zeros(len(rows))
        for a, b in StratifiedKFold(5, shuffle=True, random_state=rs).split(rows, yy):
            oof[b] = LogisticRegression(C=c, max_iter=mi).fit(Z[rows[a]], yy[a]).predict_proba(Z[rows[b]])[:, 1]
        s = log_loss(yy, np.clip(oof, 1e-4, 1 - 1e-4), labels=[0, 1])
        if best is None or s < best[0]: best = (s, c, oof)
    P = LogisticRegression(C=best[1], max_iter=mi).fit(Z[rows], yy).predict_proba(Z)[:, 1]
    return dict(p=P, C=best[1], n=len(rows), rate=float(yy.mean()), fitted_here=True)

def probes(st, ds, B, need):
    """need: set of (J, cut). Returns dict(v02={target: rec}, d03={target: rec}) with grader-free 'aud' / 'sure' and every needed target that can
    be defined (threshold and scores present). Verified caches are reused when the training label rate matches; otherwise fitted here."""
    V = pickle.load(open(f"{VMC}/p_{st}_{ds}_L16.pkl", "rb"))["T"]; D = pickle.load(open(f"{DOC}/pred_{st}_{ds}_lin16.pkl", "rb"))["targets"]
    path = f"{CACHE}/probes_{st}_{ds}.pkl"; EX_ = pickle.load(open(path, "rb")) if os.path.exists(path) else {}
    tr = np.where(B["train"])[0]; v = np.isin(B["cd"][:, 0], (1, 2)); out = dict(v02=dict(aud=V["aud"], sure=V["sure"]), d03=dict(aud=D["aud"], sure=D["sure"]), notes=[])
    Zr = None; dirty = False
    for J, cut in sorted(need):
        t = thr(J, st, ds, cut)
        if J not in B["PJ"] or t is None: out["notes"].append(f"{J}_{cut}: no scores or threshold"); continue
        y = (B["PJ"][J][:, 0] <= t).astype(int); h = hashlib.md5(y.tobytes() + v.tobytes()).hexdigest(); rate = float(y[tr[v[tr]]].mean())
        for style, name, src in (("v02", f"rej_{J}_{cut}", V), ("d03", f"{J}_{cut}", D)):
            if name in src and abs(src[name]["rate"] - rate) < 1e-9 and src[name].get("n", src[name].get("n_fit")) == int(v[tr].sum()):
                out[style][name] = src[name]; continue
            key = (style, name, h)
            if key not in EX_:
                if Zr is None: Zr = pca16(st, ds, B)
                Z = Zr.astype(np.float64) if style == "v02" else Zr.astype(np.float32)
                Z = Z / (Z[tr].std(0) + 1e-6)
                EX_[key] = _fit(Z, y, v, tr, style); dirty = True
                print(f"[build] probe {style} {st} {ds} {name} fitted (rate {rate:.3f})", flush=True)
            if EX_[key] is not None: out[style][name] = EX_[key]
            else: out["notes"].append(f"{name} ({style}): too few labels")
    if dirty: pickle.dump(EX_, open(path, "wb"))
    return out

# ------------------------------------------------------------------ cell assembly
def judge_order(Js): return [J for J in CORE if J in Js] + sorted(J for J in Js if J not in CORE)

def load_cell(st, ds, JR, RR, cuts, build_only=False):
    B = build_base(st, ds); R, skipped = build_runs(st, ds, B, JR, RR)
    need = {(r["J"], r["cut"]) for r in JR if r["st"] == st and r["ds"] == ds and r["cut"] in cuts}
    PRB = probes(st, ds, B, need)
    if build_only: return None
    new = B["new"]; cell = dict(st=st, ds=ds, B=B, PRB=PRB, skipped=skipped, runs={}, inv={})
    for name, r in R.items():
        if "J" not in r: continue
        a = float((r["o"][new] == 0).mean()); r["a"] = a
        k = (r["J"], r["cut"]); inv = cell["inv"].setdefault(k, dict(found=[], entered=[], dropped=[], noscore=False))
        inv["found"].append(r["seed"])
        if r["J"] not in B["PJ"] or thr(r["J"], st, ds, r["cut"]) is None: inv["noscore"] = True; continue
        if a >= 0.95: inv["dropped"].append(r["seed"]); continue
        inv["entered"].append(r["seed"]); cell["runs"].setdefault(k, []).append(r)
    for meta in JR:
        if meta["st"] == st and meta["ds"] == ds and meta["run"] in skipped:
            cell["inv"].setdefault((meta["J"], meta["cut"]), dict(found=[], entered=[], dropped=[], noscore=False)).setdefault("unfinished", []).append(meta["seed"])
    cell["refs"] = [r for r in R.values() if "g" in r]
    for r in cell["refs"]: r["a"] = float((r["o"][new] == 0).mean())
    return cell

# ================================================================== (i) crossover, v05
def x_prep(cell, kind, cut, uni):
    B = cell["B"]; st, ds = cell["st"], cell["ds"]; cd0 = B["cd"][:, 0]; v = np.isin(cd0, (1, 2)); s = B["son0"]
    good = ((cd0 == 1) | (s >= 4)) if kind == "right" else ((cd0 == 2) & (s >= 1) & (s <= 3))
    U = np.where(B["new"] & v & good)[0]; Y = {}
    for J in judge_order({J for (J, c) in cell["runs"] if c == cut and J in uni}):
        Y[J] = np.array([(r["o"][U] == 0).astype(float) for r in sorted(cell["runs"][(J, cut)], key=lambda r: r["run"])])
    acc = {J: B["PJ"][J][U, 0] > thr(J, st, ds, cut) for J in Y}
    P2 = cell["PRB"]["v02"]; G = np.column_stack([B["conf"][U], logit(P2["aud"]["p"][U]), logit(P2["sure"]["p"][U])])
    return dict(U=U, acc=acc, Y=Y, G=G)

def x_sets(d, K, J, kind):
    if kind == "right": return (~d["acc"][K]) & d["acc"][J], (~d["acc"][J]) & d["acc"][K]
    return d["acc"][K] & ~d["acc"][J], d["acc"][J] & ~d["acc"][K]

def x_cells(D, kind, MIN=5, W=None, SEL=None, Yover=None):
    out = {}
    for (st, ds), d in D.items():
        w = np.ones(len(d["U"])) if W is None else W[(st, ds)]; Yd = d["Y"] if Yover is None else Yover[(st, ds)]
        for K, J in itertools.combinations(list(d["Y"]), 2):
            A, Bm = x_sets(d, K, J, kind)
            if A.sum() < MIN or Bm.sum() < MIN: continue
            wa, wb = w * A, w * Bm
            if wa.sum() == 0 or wb.sum() == 0: continue
            yk = (Yd[K] if SEL is None else Yd[K][SEL[(st, ds, K)]]).mean(0); yj = (Yd[J] if SEL is None else Yd[J][SEL[(st, ds, J)]]).mean(0)
            y = yk - yj; X = y @ wa / wa.sum() - y @ wb / wb.sum()
            m = A | Bm; ww = w[m]; Z1 = np.column_stack([np.ones(m.sum()), d["G"][m]]); WX = Z1 * ww[:, None]
            M = np.column_stack([A[m].astype(float), y[m]]); Rr = M - Z1 @ np.linalg.solve(Z1.T @ WX + 1e-9 * np.eye(Z1.shape[1]), WX.T @ M)
            den = (ww * Rr[:, 0] ** 2).sum(); Xa = (ww * Rr[:, 0] * Rr[:, 1]).sum() / den if den > 1e-9 else np.nan
            out[(st, ds, K, J)] = (X, int(A.sum()), int(Bm.sum()), Xa)
    return out

def x_summ(X, who="pooled", j=0, dsf=None):
    def m(keys):
        v_ = [X[k][j] for k in keys if np.isfinite(X[k][j])]; return np.mean(v_) if v_ else np.nan
    ks = [k for k in X if dsf is None or k[1] in dsf]
    if who == "pooled":
        per = [m([k for k in ks if k[0] == st]) for st in STS]; per = [p for p in per if np.isfinite(p)]
        return np.mean(per) if per else np.nan
    return m([k for k in ks if who in (k[0], k[1])])

def run_crossover(cells, cut, uni, NB, NP, rng, P, store):
    for kind in ("right", "wrong"):
        D = {(c["st"], c["ds"]): x_prep(c, kind, cut, uni) for c in cells}
        X = x_cells(D, kind)
        WHO = [("pooled", None), ("qwen7", None), ("llama8", None), ("h1", None), ("pq", None), ("nq", None), ("pooled", ("pq", "nq"))] + \
              [(st, (ds,)) for st in STS for ds in DSS]
        reps = defaultdict(list); JX = sorted({J for k in X for J in k[2:]}, key=lambda J: (J not in CORE, CORE.index(J) if J in CORE else 0, J))
        def xj(Xd, J, st, j):
            v_ = [x[j] for k, x in Xd.items() if k[0] == st and J in k[2:] and np.isfinite(x[j])]; return np.mean(v_) if v_ else np.nan
        for b in range(NB):
            W = {k: np.bincount(rng.integers(0, len(d["U"]), len(d["U"])), minlength=len(d["U"])).astype(float) for k, d in D.items()}
            SEL = {(st, ds, J): rng.integers(0, len(d["Y"][J]), len(d["Y"][J])) for (st, ds), d in D.items() for J in d["Y"]}
            Xb = x_cells(D, kind, 5, W, SEL)
            for w, dsf in WHO:
                for j in (0, 3): reps[(w, dsf, j)].append(x_summ(Xb, w, j, dsf))
            for J in JX:
                for st in STS:
                    for j in (0, 3): reps[("J", J, st, j)].append(xj(Xb, J, st, j))
        null = defaultdict(list)
        for b in range(NP):
            Yp = {}
            for k, d in D.items():
                labs = [J for J in d["Y"] for _ in range(len(d["Y"][J]))]; rows = np.vstack([d["Y"][J] for J in d["Y"]]) if d["Y"] else None; Yp[k] = {}
                if rows is None: continue
                perm = rng.permutation(len(labs))
                for lab, jj in zip(labs, perm): Yp[k].setdefault(lab, []).append(rows[jj])
                Yp[k] = {J: np.array(v_) for J, v_ in Yp[k].items()}
            Xp = x_cells(D, kind, 5, Yover=Yp)
            for w in ("pooled",) + STS: null[w].append(x_summ(Xp, w))
        lab = "X_FR" if kind == "right" else "X_FA"
        P(f"  {lab} ({kind} units; cells |A|,|B| >= 5): {len(X)} cells, negative {sum(x[0] < 0 for x in X.values())}, positive {sum(x[0] > 0 for x in X.values())}")
        res = {}
        for w, dsf in WHO:
            e = x_summ(X, w, 0, dsf); ea = x_summ(X, w, 3, dsf)
            ncell = sum(1 for k in X if (dsf is None or k[1] in dsf) and (w == "pooled" or w in (k[0], k[1])))
            if ncell == 0: continue
            s = f"    {w:6s}{'' if dsf is None else '@' + '+'.join(dsf):7s} X {f3(e, pct(reps[(w, dsf, 0)]))}  X_adj(FWL) {f3(ea, pct(reps[(w, dsf, 3)]))}  cells {ncell}"
            if dsf is None and w in null:
                nv = np.array([x for x in null[w] if np.isfinite(x)])
                if len(nv) and np.isfinite(e): s += f"  perm p {(np.sum(np.abs(nv) >= abs(e)) + 1) / (len(nv) + 1):.4f}"
            P(s); res[f"{w}{'' if dsf is None else '@' + '+'.join(dsf)}"] = dict(X=e, ci=pct(reps[(w, dsf, 0)]), Xadj=ea, ci_adj=pct(reps[(w, dsf, 3)]), cells=ncell)
        pj = {}
        for J in JX:
            for st in STS:
                e = xj(X, J, st, 0)
                if np.isfinite(e): pj[f"{J}|{st}"] = dict(X=e, ci=pct(reps[("J", J, st, 0)]), Xadj=xj(X, J, st, 3), ci_adj=pct(reps[("J", J, st, 3)]),
                                                         cells=sum(1 for k in X if k[0] == st and J in k[2:]))
        P("    per judge (mean over cells containing it, per student): " + "; ".join(f"{k} X {f3(v_['X'], v_['ci'])} adj {f3(v_['Xadj'], v_['ci_adj'])} ({v_['cells']})" for k, v_ in pj.items()))
        P("    cells: " + "; ".join(f"{k[0]} {k[1]} {k[2]}-{k[3]} {x[0]:+.3f}/adj {x[3]:+.3f} (|A| {x[1]}, |B| {x[2]})" for k, x in sorted(X.items())))
        store[lab] = dict(summary=res, per_judge=pj, cells={"|".join(k): x for k, x in X.items()})

# ================================================================== (ii) readable-map slopes, v08 MAIN
def s_prep(cell, cut, uni):
    B = cell["B"]; st, ds = cell["st"], cell["ds"]; cd0 = B["cd"][:, 0]; v = np.isin(cd0, (1, 2)); s0 = B["son0"]; base = B["new"]
    masks = {"right": base & v & ((cd0 == 1) | (s0 >= 4)), "wrong": base & v & (cd0 == 2) & (s0 >= 1) & (s0 <= 3)}
    P2 = cell["PRB"]["v02"]; Y = {}
    for J in judge_order({J for (J, c) in cell["runs"] if c == cut and J in uni and f"rej_{J}_{cut}" in P2}):
        Y[J] = np.array([(r["o"] == 0).astype(float) for r in sorted(cell["runs"][(J, cut)], key=lambda r: r["run"])])
    pr = {J: logit(P2[f"rej_{J}_{cut}"]["p"]) for J in Y}; rej = {J: B["PJ"][J][:, 0] <= thr(J, st, ds, cut) for J in Y}
    G = np.column_stack([B["conf"], logit(P2["aud"]["p"]), logit(P2["sure"]["p"])])
    return dict(masks={k: np.where(m)[0] for k, m in masks.items()}, Y=Y, pr=pr, rej=rej, G=G, N=B["N"])

def _resid(M, G, w):
    X1 = np.column_stack([np.ones(len(G)), G]); WX = X1 * w[:, None]
    return M - X1 @ np.linalg.solve(X1.T @ WX + 1e-9 * np.eye(X1.shape[1]), WX.T @ M)

def s_pair(c, panel, K, J, w, sel, Yover=None):
    U = c["masks"][panel]; Yd = c["Y"] if Yover is None else Yover
    yK = (Yd[K] if sel is None else Yd[K][sel[K]]).mean(0)[U]; yJ = (Yd[J] if sel is None else Yd[J][sel[J]]).mean(0)[U]
    y = yK - yJ; x = c["pr"][K][U] - c["pr"][J][U]
    Rr = _resid(np.column_stack([x, y]), c["G"][U], w); x, y = Rr[:, 0], Rr[:, 1]
    sdx = np.sqrt((w * x * x).sum() / w.sum()); x = x / sdx
    slope = (w * x * y).sum() / (w * x * x).sum()
    b = wq_bins(x, w); curve = np.array([(w[b == j] * y[b == j]).sum() / w[b == j].sum() for j in range(5)])
    yr = yK - yJ; oK = c["rej"][K][U] & ~c["rej"][J][U]; oJ = c["rej"][J][U] & ~c["rej"][K][U]; wa, wb = w * oK, w * oJ
    X = (yr @ wa / wa.sum() - yr @ wb / wb.sum()) if (oK.sum() >= 5 and oJ.sum() >= 5 and wa.sum() > 0 and wb.sum() > 0) else np.nan
    return slope, curve, X

def s_cells(Cs, panel, W=None, SEL=None, YO=None):
    out = {}; pairs = {}
    for k, c in Cs.items():
        n = len(c["masks"][panel]); w = np.ones(n) if W is None else W[k][c["masks"][panel]]
        res = []
        for K, J in itertools.combinations(list(c["Y"]), 2):
            r = s_pair(c, panel, K, J, w, None if SEL is None else SEL[k], None if YO is None else YO[k]); res.append(r); pairs[k + (K, J)] = r[0]
        if not res: continue
        out[k] = (np.mean([r[0] for r in res]), np.mean([r[1] for r in res], 0), np.nanmean([r[2] for r in res]) if any(np.isfinite(r[2]) for r in res) else np.nan,
                  len(res), sum(r[0] > 0 for r in res))
    return out, pairs

def s_pool(S, i, who=None, dsf=None):
    per = []
    for s in ([who] if who else STS):
        v_ = [S[k][i] for k in S if k[0] == s and (dsf is None or k[1] in dsf) and np.all(np.isfinite(S[k][i]))]
        if v_: per.append(np.mean(v_, 0))
    return np.mean(per, 0) if per else np.nan

def s_judge(pairs, J, st):
    v_ = [x for k, x in pairs.items() if k[0] == st and J in (k[2], k[3]) and np.isfinite(x)]; return np.mean(v_) if v_ else np.nan

def run_slopes(cells, cut, uni, NB, NP, rng, P, store):
    Cs = {(c["st"], c["ds"]): s_prep(c, cut, uni) for c in cells}
    Cs = {k: c for k, c in Cs.items() if len(c["Y"]) >= 2}
    if not Cs: P("  S: no cell has two judges with runs at this cut -> S_FR / S_FA undefined"); return
    jud = sorted({J for c in Cs.values() for J in c["Y"]}, key=lambda J: (J not in CORE, CORE.index(J) if J in CORE else 0, J))
    for panel in ("right", "wrong"):
        S0, PR0 = s_cells(Cs, panel); Bs = []; BP = []
        for b in range(NB):
            W = {k: np.bincount(rng.integers(0, c["N"], c["N"]), minlength=c["N"]).astype(float) for k, c in Cs.items()}
            SEL = {k: {J: rng.integers(0, len(c["Y"][J]), len(c["Y"][J])) for J in c["Y"]} for k, c in Cs.items()}
            sb, pb = s_cells(Cs, panel, W, SEL); Bs.append(sb); BP.append(pb)
        NUL = []
        for b in range(NP):
            YO = {}
            for k, c in Cs.items():
                labs = [J for J in c["Y"] for _ in range(len(c["Y"][J]))]; rows = np.vstack([c["Y"][J] for J in c["Y"]]); perm = rng.permutation(len(labs)); d = {}
                for lab, j in zip(labs, perm): d.setdefault(lab, []).append(rows[j])
                YO[k] = {J: np.array(x) for J, x in d.items()}
            NUL.append(s_cells(Cs, panel, YO=YO)[0])
        lab = "S_FR" if panel == "right" else "S_FA"; res = {}
        sl = s_pool(S0, 0); slb = [s_pool(s, 0) for s in Bs]; nul = np.array([s_pool(s, 0) for s in NUL]); nul = nul[np.isfinite(nul)]
        pp = (np.sum(np.abs(nul) >= abs(sl)) + 1) / (len(nul) + 1) if len(nul) else np.nan
        cv = s_pool(S0, 1); cvb = np.array([s_pool(s, 1) for s in Bs]); lo, hi = np.nanpercentile(cvb, [2.5, 97.5], axis=0)
        npairs = sum(S0[k][3] for k in S0); npos = sum(S0[k][4] for k in S0)
        units = "/".join(f"{k[0]}:{k[1]} {len(c['masks'][panel])}" for k, c in Cs.items())
        P(f"  {lab} ({panel} units {units}): slope per SD {f3(sl, pct(slb), 4)}; pairs > 0: {npos}/{npairs}; placebo p {pp:.4f}")
        res["pooled"] = dict(slope=sl, ci=pct(slb), pairs=npairs, pos=npos, p=pp)
        for who in STS:
            e = s_pool(S0, 0, who)
            if np.isfinite(e): res[who] = dict(slope=e, ci=pct([s_pool(s, 0, who) for s in Bs])); P(f"    {who}: slope {f3(e, res[who]['ci'], 4)}")
        for ds in DSS:
            e = s_pool(S0, 0, None, (ds,))
            if np.isfinite(e): res["@" + ds] = dict(slope=e, ci=pct([s_pool(s, 0, None, (ds,)) for s in Bs])); P(f"    @{ds} (students averaged): slope {f3(e, res['@' + ds]['ci'], 4)}")
        if np.all(np.isfinite(cv)):
            P("    quintile curve (pooled): " + "  ".join(f"Q{j + 1} {cv[j]:+.3f} [{lo[j]:+.3f}, {hi[j]:+.3f}]" for j in range(5)) +
              f" | Q5 - Q1 {f3(cv[4] - cv[0], pct(cvb[:, 4] - cvb[:, 0]))}")
        P("    per cell: " + " ".join(f"{k[0]}/{k[1]} {S0[k][0]:+.4f} ({S0[k][4]}/{S0[k][3]})" for k in S0))
        P("    per judge (mean over its pairs, per student): " + "; ".join(
            f"{J}: " + ", ".join(f"{st} {f3(s_judge(PR0, J, st), pct([s_judge(pb, J, st) for pb in BP]), 4)}" for st in STS if np.isfinite(s_judge(PR0, J, st))) for J in jud))
        res["per_judge"] = {f"{J}|{st}": dict(slope=s_judge(PR0, J, st), ci=pct([s_judge(pb, J, st) for pb in BP])) for J in jud for st in STS if np.isfinite(s_judge(PR0, J, st))}
        res["pairs"] = {"|".join(k): v_ for k, v_ in PR0.items()}
        store[lab] = res

# ================================================================== (iii) content lift, stats/c1 MAIN at the cut
def c1_units(cells, cut, LK):
    UN = []; QN = {}
    for c in cells:
        B = c["B"]; st, ds = c["st"], c["ds"]; JF = LK[B["dsn"]]["JF"]; N = B["N"]; X = B["X"]
        panel_all = [J for J in J4 if J in B["PJ"] and J in JF]
        th = {J: thr(J, st, ds, cut) for J in set(panel_all) | {J for (J, cc) in c["runs"] if cc == cut}}
        valid = np.isin(B["cd"], (1, 2)); kl = B["cd"] == 1; wrong_s = (valid & ~kl)[:, 1:]
        def bJ(J):
            pj = B["PJ"][J][:, 1:]; paid = (pj > th[J]) & wrong_s & np.isfinite(pj); den = (wrong_s & np.isfinite(pj)).sum(1)
            return np.where(den > 0, paid.sum(1) / np.maximum(den, 1), np.nan)
        parts = {"home": np.where(B["train"])[0], "new": np.where(B["new"])[0]}
        for (J, cc), runs in c["runs"].items():
            if cc != cut or J not in JF or J not in B["PJ"] or th.get(J) is None: continue
            panel = panel_all if J in panel_all else panel_all + [J]
            if J not in panel or len(panel) < 2: continue
            bv_all = {K: bJ(K) for K in panel}
            for part, items in parts.items():
                QN[(st, ds, part)] = len(items); pos = {int(i): k for k, i in enumerate(items)}
                for r in sorted(runs, key=lambda r: r["run"]):
                    qp, Dm, nall = [], [], 0
                    for i in items:
                        x = r["f"][i]
                        if r["o"][i] != 2 or x == X[i][0]: continue
                        nall += 1
                        pv = []
                        for K in panel:
                            v_ = JF[K]["line"].get((int(i), r["fl"][i])); v_ = v_ if v_ is not None else JF[K]["norm"].get((int(i), x)); pv.append(v_)
                        bv = [bv_all[K][i] for K in panel]
                        if any(v_ is None for v_ in pv) or any(not np.isfinite(v_) for v_ in bv): continue
                        qp.append(pos[int(i)]); Dm.append([(pv[k] > th[panel[k]]) - bv[k] for k in range(len(panel))])
                    if len(qp) >= 5:
                        UN.append(dict(st=st, ds=ds, part=part, run=r["run"], own=J, panel=panel, cell=(st, ds, J, cut), qpos=np.array(qp), D=np.array(Dm, float), ncand=nall))
    return UN, QN

def c1_S(u, l):
    k = u["panel"].index(u["own"]); return l[k] - np.mean([l[j] for j in range(len(u["panel"])) if j != k])

def c1_est(UN, W=None, runsel=None, core=True):
    vals = defaultdict(list)
    idx = range(len(UN)) if runsel is None else [j for L in runsel.values() for j in L]
    for j in idx:
        u = UN[j]
        if core != (u["own"] in J4): continue
        w = np.ones(len(u["qpos"])) if W is None else W[(u["st"], u["ds"], u["part"])][u["qpos"]]
        if w.sum() == 0: continue
        l = (w[:, None] * u["D"]).sum(0) / w.sum(); vals[(u["st"], u["part"])].append(c1_S(u, l))
        vals[(u["st"], u["ds"], u["part"])].append(c1_S(u, l)); vals[(u["st"], u["own"], u["part"])].append(c1_S(u, l))
    out = {k: np.mean(v) for k, v in vals.items()}
    for part in ("home", "new"):
        if (STS[0], part) in out and (STS[1], part) in out: out[("pooled", part)] = (out[(STS[0], part)] + out[(STS[1], part)]) / 2
    return out

def run_c1(cells, cut, LK, NB, rng, P, store):
    UN, QN = c1_units(cells, cut, LK)
    for core in (True, False):
        sub = [u for u in UN if (u["own"] in J4) == core]
        if not sub: continue
        lab = "panel judges Q M G O (c1)" if core else "judges outside the panel (vs Q M G O)"
        est = c1_est(UN, core=core); cellidx = defaultdict(list)
        for j, u in enumerate(UN):
            if (u["own"] in J4) == core: cellidx[(u["cell"], u["part"])].append(j)
        reps = []
        for b in range(NB):
            W = {k: np.bincount(rng.integers(0, n, n), minlength=n).astype(float) for k, n in QN.items()}
            sel = {k: [L[t] for t in rng.integers(0, len(L), len(L))] for k, L in cellidx.items()}
            reps.append(c1_est(UN, W, sel, core))
        P(f"  content lift S ({lab}; changed key-wrong answers, all verdicts available; run-weighted):")
        res = {}
        keys = [("pooled",), ("qwen7",), ("llama8",)] + [(st, ds) for st in STS for ds in DSS] + sorted({(u["st"], u["own"]) for u in sub})
        for kk in keys:
            h = est.get(kk + ("home",)); n = est.get(kk + ("new",))
            if h is None and n is None: continue
            bh = [r.get(kk + ("home",), np.nan) for r in reps]; bn = [r.get(kk + ("new",), np.nan) for r in reps]
            br = [a / b if b else np.nan for a, b in zip(bn, bh)]
            nr = len({u["run"] for u in sub if u["part"] == "new" and (kk[0] == "pooled" or (u["st"] == kk[0] and (len(kk) == 1 or kk[1] in (u["ds"], u["own"]))))})
            h = np.nan if h is None else h; n = np.nan if n is None else n
            P(f"    {'/'.join(kk):14s} home {f3(h, pct(bh))}  new {f3(n, pct(bn))}  ratio new/home {f3(n / h if h else np.nan, pct(br))}  (runs on new {nr})")
            res["/".join(kk)] = dict(home=h, ci_home=pct(bh), new=n, ci_new=pct(bn), ratio=n / h if h else np.nan, ci_ratio=pct(br), runs_new=nr)
        cov = [(u["ncand"], len(u["qpos"])) for u in sub if u["part"] == "new"]
        if cov: P(f"    verdict coverage of changed key-wrong answers (new): {sum(b for a, b in cov)}/{sum(a for a, b in cov)}")
        store["content_" + ("panel" if core else "outside")] = res
    miss = sorted({(c["st"], c["ds"], J) for c in cells for (J, cc) in c["runs"] if cc == cut} - {(u["st"], u["ds"], u["own"]) for u in UN})
    if miss: P(f"    (no c1 rows: {', '.join('/'.join(m) for m in miss)} - fewer than 5 changed key-wrong answers with all verdicts per run, or no lookups)")

# ================================================================== (iv) + R1: costs (regimes/rg, r02, r02e)
def rg_runs(cell, LK, runs, J4list):
    """rg.build per-run arrays: o, aud, chg, pj (PYD (idx, norm) lookup), son."""
    B = cell["B"]; PYD = LK[B["dsn"]]["PYD"]; N = B["N"]; gr = B["gr"]; LV2 = B["LV2"]; out = []
    for r in runs:
        o = r["o"]; son = np.full(N, -1); chg = np.zeros(N, bool); pj = {J: np.full(N, np.nan) for J in J4list}
        for i in range(N):
            if o[i] not in (1, 2): continue
            x = r["f"][i]; son[i] = LV2.get((i, x), -1); chg[i] = not (gr[i] and same(x, gr[i]))
            for J in J4list:
                v_ = PYD.get(J, {}).get((i, x))
                if v_ is not None: pj[J][i] = v_
        out.append(dict(o=o, son=son, aud=(o == 1) | ((o == 2) & (son >= 4)), chg=chg, pj=pj, run=r["run"]))
    return out

def placebo(cell, U, target):
    best = None
    for g in PLACEBO_G:
        rr = [r for r in cell["refs"] if r["g"] == g]
        if not rr: continue
        a = np.mean([(r["o"][U] == 0).mean() for r in rr])
        if best is None or abs(a - target) < best[0]: best = (abs(a - target), g, rr, a)
    return best

def located_stats(yG, yR, nG, nR, rej, strat, ix, Ntot):
    """verbatim from regimes/rg.py (risk-difference parts dropped; excess and MH pieces unchanged)."""
    yG = yG[ix]; yR = yR[ix]; rej = rej[ix]; strat = strat[ix]
    num = den = 0.0; numR = denR = 0.0; ex = 0.0
    for s in np.unique(strat):
        m = strat == s; mr = m & rej; ma = m & ~rej
        if mr.sum() == 0 or ma.sum() == 0: continue
        A = yG[mr].sum() * nG; Bn = mr.sum() * nG - A; Cc = yG[ma].sum() * nG; D = ma.sum() * nG - Cc; T = (mr.sum() + ma.sum()) * nG
        num += A * D / T; den += Bn * Cc / T
        A2 = yR[mr].sum() * nR; B2 = mr.sum() * nR - A2; C2 = yR[ma].sum() * nR; D2 = ma.sum() * nR - C2; T2 = (mr.sum() + ma.sum()) * nR
        numR += A2 * D2 / T2; denR += B2 * C2 / T2
    lorR = np.log((numR + 0.5) / (denR + 0.5))
    for s in np.unique(strat):
        m = strat == s; mr = m & rej; ma = m & ~rej
        if mr.sum() == 0 or ma.sum() == 0: continue
        base = yG[ma].mean(); ex += mr.sum() * (yG[mr].mean() - sig(logit_rg(base) + lorR))
    return dict(excess100=100 * ex / Ntot, n_rej100=100 * rej.sum() / Ntot)

def cost_cell(cell, LK, J, cut):
    """per-question arrays for one (student, dataset, judge, cut): RW on new; LOC pieces; R1 / IHr on training questions."""
    B = cell["B"]; st, ds = cell["st"], cell["ds"]; runs = cell["runs"][(J, cut)]; t = thr(J, st, ds, cut)
    RR = rg_runs(cell, LK, runs, [J]); held = B["new"]; trn = B["train"]
    cd0 = B["cd"][:, 0]; gval = np.isin(cd0, (1, 2)); aud0 = B["aud_rg"] & gval
    idk = np.array([r["o"] == 0 for r in RR], float)
    RW = (idk * aud0).mean(0)
    paid = np.array([np.nan_to_num(r["pj"][J], nan=0.0) > t for r in RR]); scored = np.array([np.isfinite(r["pj"][J]) for r in RR])
    kw = np.array([(r["o"] == 2) & r["chg"] for r in RR]); aw = np.array([(r["o"] == 2) & ~r["aud"] & r["chg"] for r in RR])
    R1 = (kw & paid).mean(0); IHr = (aw & paid).mean(0); KW = kw.mean(0); KWs = (kw & scored).mean(0)
    U = held & gval & B["aud_rg"] & B["ks"][:, 0] & np.isfinite(B["PJ"][J][:, 0]); rej = (B["PJ"][J][:, 0] <= t)[U]
    yG = np.mean([r["o"][U] == 0 for r in runs], axis=0); pb = placebo(cell, U, yG.mean())
    loc = None
    if pb is not None:
        yR = np.mean([r["o"][U] == 0 for r in pb[2]], axis=0)
        loc = dict(uidx=np.where(U[held])[0], yG=yG, yR=yR, rej=rej, strat=np.digitize(B["conf"][U], CBINS), nG=len(runs), nR=len(pb[2]), pl=f"{pb[1]}({pb[3]:.2f}/{yG.mean():.2f})")
    return dict(rw=RW[held], loc=loc, r1=R1[trn], ihr=IHr[trn], kw=KW[trn], kws=KWs[trn], abst=float(idk[:, held].mean()), nrun=len(runs))

def cost_stat(Bs, sel, selT):
    rw = sum(b["rw"][s].sum() for b, s in zip(Bs, sel)); n = sum(len(s) for s in sel); ex = 0.0; nrej = 0.0
    for b, s in zip(Bs, sel):
        if b["loc"] is None: ex = nrej = np.nan; continue
        L = b["loc"]; pos = {q: k for k, q in enumerate(L["uidx"])}; ix = np.array([pos[q] for q in s if q in pos], int)
        if len(ix):
            o = located_stats(L["yG"], L["yR"], L["nG"], L["nR"], L["rej"], L["strat"], ix, 1); ex += o["excess100"] / 100; nrej += o["n_rej100"] / 100
    nt = sum(len(s) for s in selT); r1 = sum(b["r1"][s].sum() for b, s in zip(Bs, selT)); ih = sum(b["ihr"][s].sum() for b, s in zip(Bs, selT))
    kw = sum(b["kw"][s].sum() for b, s in zip(Bs, selT)); kws = sum(b["kws"][s].sum() for b, s in zip(Bs, selT))
    return [100 * rw / n, 100 * ex / n, 100 * nrej / n, ex / nrej if nrej > 0 else np.nan, ex / rw if rw > 0 and np.isfinite(ex) else np.nan,
            100 * r1 / nt, 100 * ih / nt, (100 * r1 / nt) * (kw / kws) if kws > 0 else np.nan]

def run_costs(cells, cut, LK, NB, rng, P, store):
    names = ["RW withheld/100 new", "LOC located/100", "FRJ judge false rejects/100", "inherited share LOC/FRJ", "located share LOC/RW",
             "R1 installed judge-paid key-wrong/100 train", "IHr (r02, audited-wrong)/100 train", "R1 coverage-corrected"]
    P("  columns: " + " | ".join(names) + " ; coverage = share of changed key-wrong training answers with an own-judge verdict")
    res = {}
    jud = sorted({J for c in cells for (J, cc) in c["runs"] if cc == cut}, key=lambda J: (J not in CORE, CORE.index(J) if J in CORE else 0, J))
    for J in jud:
        per = {}
        for c in cells:
            if (J, cut) in c["runs"]: per[(c["st"], c["ds"])] = cost_cell(c, LK, J, cut)
        groups = [((st, ds), [(st, ds)]) for st in STS for ds in DSS if (st, ds) in per] + [((st, "all"), [k for k in per if k[0] == st]) for st in STS] + \
                 [((st, "pq+nq"), [k for k in per if k[0] == st and k[1] in ("pq", "nq")]) for st in STS]
        for gk, ks in groups:
            if not ks or (gk[1] in ("all", "pq+nq") and len(ks) < 2): continue
            Bs = [per[k] for k in ks]; full = [np.arange(len(b["rw"])) for b in Bs]; fullT = [np.arange(len(b["r1"])) for b in Bs]
            e0 = cost_stat(Bs, full, fullT); bs = []
            for _ in range(NB):
                bs.append(cost_stat(Bs, [rng.integers(0, len(b["rw"]), len(b["rw"])) for b in Bs], [rng.integers(0, len(b["r1"]), len(b["r1"])) for b in Bs]))
            bs = np.array(bs, float); cis = [pct(bs[:, j]) for j in range(8)]
            cov = sum(b["kws"].sum() for b in Bs) / max(sum(b["kw"].sum() for b in Bs), 1e-9)
            nrun = "+".join(str(per[k]["nrun"]) for k in ks); pl = " ".join(f"{k[1]}:{per[k]['loc']['pl']}" for k in ks if per[k]["loc"])
            P(f"    {J}_{cut} {gk[0]} {gk[1]:5s} runs {nrun:7s} abst {np.mean([per[k]['abst'] for k in ks]):.2f} | " +
              " | ".join(f"{e0[j]:+.2f} [{cis[j][0]:+.2f},{cis[j][1]:+.2f}]" if j not in (3, 4) else f"{e0[j]:+.3f} [{cis[j][0]:+.3f},{cis[j][1]:+.3f}]" for j in range(8)) +
              f" | coverage {cov:.2f} | placebo {pl}")
            res[f"{J}|{gk[0]}|{gk[1]}"] = dict(zip(["RW", "LOC", "FRJ", "inh_share", "loc_share", "R1", "IHr", "R1_cc"], [(e0[j], cis[j]) for j in range(8)]), coverage=cov, runs=nrun)
    store["costs"] = res

# ================================================================== (v) dose, d04 grader-specific T-B with the run's own grader's lin16 probe
def dose_prep(cell, cut):
    B = cell["B"]; PR = cell["PRB"]["d03"]; v = np.isin(B["cd"][:, 0], (1, 2)); kl = B["cd"][:, 0] == 1
    aud = v & (kl | (B["son0"] >= 4)); sure = v & (B["conf"] >= 31 / 32); U = np.where(B["new"] & sure & aud)[0]
    arms = {}
    for (J, cc), runs in cell["runs"].items():
        if cc != cut or f"{J}_{cut}" not in PR: continue
        arms[J] = sorted(runs, key=lambda r: r["run"])
    ref = sorted([r for r in cell["refs"] if r["a"] < 0.95], key=lambda r: r["run"])
    if not arms or not ref: return None
    runs = []; arm_ix = {}
    for J in sorted(arms): arm_ix[J] = list(range(len(runs), len(runs) + len(arms[J]))); runs += arms[J]
    ref_ix = list(range(len(runs), len(runs) + len(ref))); runs += ref
    Y = np.column_stack([(r["o"][U] == 0).astype(float) for r in runs])
    G = np.column_stack([B["conf"][U], logit(PR["aud"]["p"][U]), logit(PR["sure"]["p"][U])])
    X = {J: logit(PR[f"{J}_{cut}"]["p"][U]) for J in arms}
    return dict(U=U, Y=Y, G=G, X=X, arm_ix=arm_ix, ref_ix=ref_ix, a=np.array([r["a"] for r in runs]))

def dose_runstats(d, J, w):
    X = d["X"][J]; G = d["G"]; Y = d["Y"]; ok = np.isfinite(X) & np.all(np.isfinite(G), 1) & (w > 0)
    if ok.sum() < 30 or np.std(X[ok]) < 1e-6: return None
    ww = w[ok]; x = X[ok]; Yk = Y[ok]
    Rr = _resid(np.column_stack([x, Yk]), G[ok], ww); xp = Rr[:, 0]; Yp = Rr[:, 1:]
    sxx = (ww * xp * xp).sum(); sd = np.sqrt(sxx / ww.sum()); sl = (ww[:, None] * xp[:, None] * Yp).sum(0) / sxx * sd
    yadj = Yp + (ww[:, None] * Yk).sum(0) / ww.sum(); b = wq_bins(xp, ww, 5); bins = np.full((5, Y.shape[1]), np.nan)
    for j in range(5):
        m = b == j
        if ww[m].sum() > 0: bins[j] = (ww[m][:, None] * yadj[m]).sum(0) / ww[m].sum()
    return dict(sl=sl, tb=bins[4] - bins[0])

def dose_eval(Ds, W=None, SEL=None, REFSEL=None):
    out = {}
    for k, d in Ds.items():
        w = np.ones(len(d["U"])) if W is None else W[k]; a = d["a"]
        pool = np.array(d["ref_ix"]) if REFSEL is None else np.array(d["ref_ix"])[REFSEL[k]]
        for J, ixs in d["arm_ix"].items():
            r = dose_runstats(d, J, w)
            if r is None: continue
            ix = np.array(ixs) if SEL is None else np.array(ixs)[SEL[k + (J,)]]
            nn = [pool[np.argsort(np.abs(a[pool] - a[i]), kind="stable")[:3]] for i in ix]
            out[k + (J,)] = (r["tb"][ix].mean() - np.mean([r["tb"][j].mean() for j in nn]), r["sl"][ix].mean() - np.mean([r["sl"][j].mean() for j in nn]),
                             r["tb"][ix].mean(), len(ix))
    return out

def dose_pool(E, i, J=None, who=None, dsf=None):
    per = []
    for s in ([who] if who else STS):
        v_ = [E[k][i] for k in E if k[0] == s and (J is None or k[2] == J) and (dsf is None or k[1] in dsf)]
        if v_: per.append(np.mean(v_))
    return np.mean(per) if per else np.nan

def run_dose(cells, cut, NB, rng, P, store):
    Ds = {}
    for c in cells:
        d = dose_prep(c, cut)
        if d is not None: Ds[(c["st"], c["ds"])] = d
    if not Ds: P("  dose: no arm with a probe and a reference pool"); return
    E0 = dose_eval(Ds); reps = []
    for b in range(NB):
        W = {k: np.bincount(rng.integers(0, len(d["U"]), len(d["U"])), minlength=len(d["U"])).astype(float) for k, d in Ds.items()}
        SEL = {k + (J,): rng.integers(0, len(ix), len(ix)) for k, d in Ds.items() for J, ix in d["arm_ix"].items()}
        REFSEL = {k: rng.integers(0, len(d["ref_ix"]), len(d["ref_ix"])) for k, d in Ds.items()}
        reps.append(dose_eval(Ds, W, SEL, REFSEL))
    P("  dose (d04): grader-specific T-B (refusal points, top minus bottom quintile of the own-grader probe, run minus 3 nearest no-location runs) | slope per SD")
    res = {}
    jud = sorted({k[2] for k in E0}, key=lambda J: (J not in CORE, CORE.index(J) if J in CORE else 0, J))
    for J in [None] + jud:
        for who in (None,) + STS:
            for dsf in (None, ("pq", "nq")):
                e = dose_pool(E0, 0, J, who, dsf)
                if not np.isfinite(e): continue
                ci = pct([dose_pool(r, 0, J, who, dsf) for r in reps]); es = dose_pool(E0, 1, J, who, dsf); cs = pct([dose_pool(r, 1, J, who, dsf) for r in reps])
                nu = sum(1 for k in E0 if (J is None or k[2] == J) and (who is None or k[0] == who) and (dsf is None or k[1] in dsf))
                lab = f"{J or 'all judges'}|{who or 'pooled'}|{'all' if dsf is None else 'pq+nq'}"
                P(f"    {lab:24s} T-B {f3(e, ci)}  slope {f3(es, cs)}  units {nu}"); res[lab] = dict(tb=e, ci=ci, sl=es, ci_sl=cs, units=nu)
    P("    units: " + " ".join(f"{k[0]}/{k[1]}/{k[2]} {v_[0]:+.3f} (n {v_[3]})" for k, v_ in sorted(E0.items())))
    store["dose"] = res

# ================================================================== driver
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=("build", "score")); ap.add_argument("st", nargs="?"); ap.add_argument("ds", nargs="?")
    ap.add_argument("--cuts", default="t50,a50,a35"); ap.add_argument("--parts", default="i,ii,iii,iv,v"); ap.add_argument("--nb", type=int, default=1000)
    ap.add_argument("--np", type=int, default=1000); ap.add_argument("--nbs", type=int, default=0, help="bootstrap draws for S (default = nb)")
    ap.add_argument("--nps", type=int, default=500); ap.add_argument("--nbc", type=int, default=500, help="draws for costs / dose")
    ap.add_argument("--judges", default="core+all", help="pairs: always the verified universe Q M G O R ('core'); a second pass with every tag that has runs ('all') is added when such tags exist, unless 'core-only'")
    ap.add_argument("--tag", default="run"); ap.add_argument("--cells", default="")
    a = ap.parse_args(); cuts = a.cuts.split(","); parts = a.parts.split(","); t00 = time.time()
    JR, RR = discover()
    if a.mode == "build":
        cl = [(a.st, a.ds)] if a.st else [(st, ds) for st in STS for ds in DSS]
        for st, ds in cl: load_cell(st, ds, JR, RR, CUTS, build_only=True)
        tags = set(J4) | set(CORE) | {r["J"] for r in JR}
        for ds in sorted({ds for _, ds in cl}): build_lookups(DSN[ds], tags)
        print(f"build done ({time.time() - t00:.0f}s)"); return
    sel = [tuple(x.split("/")) for x in a.cells.split(",")] if a.cells else [(st, ds) for st in STS for ds in DSS]
    cells = [load_cell(st, ds, JR, RR, CUTS) for st, ds in sel]
    LK = {DSN[ds]: build_lookups(DSN[ds], set(J4) | set(CORE) | {r["J"] for r in JR}) for ds in sorted({c["ds"] for c in cells})} if any(p in parts for p in ("iii", "iv", "v")) else {}
    L = []; P = lambda *x: (print(*x, flush=True), L.append(" ".join(str(y) for y in x)))
    P(__doc__); P(f"run at {time.strftime('%Y-%m-%d %H:%M:%S')}; md5 {hashlib.md5(open(__file__, 'rb').read()).hexdigest()}; judges universe for pairs: {a.judges}")
    P("\n== run inventory per (judge, cut, student, dataset): entered = complete ck120, scores + threshold present, new abstention < 0.95")
    for cut in cuts:
        for c in cells:
            for (J, cc), inv in sorted(c["inv"].items()):
                if cc != cut: continue
                P(f"  {cut} {J:3s} {c['st']:6s} {c['ds']}: entered {len(inv['entered'])} seeds {sorted(inv['entered'])}" +
                  (f"; dropped (abstention >= 0.95) {sorted(inv['dropped'])}" if inv["dropped"] else "") + (f"; unfinished {sorted(inv['unfinished'])}" if inv.get("unfinished") else "") +
                  ("; NO SCORES/THRESHOLD" if inv["noscore"] else ""))
        for c in cells:
            for n in c["PRB"]["notes"]: P(f"  probe note {c['st']} {c['ds']}: {n}")
    STORE = {}
    for k, cut in enumerate(cuts):
        rng = np.random.default_rng(32000 + k); st_ = STORE.setdefault(cut, {})
        P(f"\n================ CUT {cut} ================")
        extra = sorted({J for c in cells for (J, cc) in c["runs"] if cc == cut and J not in CORE})
        unis = [("core", set(CORE))] + ([("all", set(CORE) | set(extra))] if extra and a.judges != "core-only" else [])
        for un, uset in unis:
            sub = st_ if un == "core" else STORE.setdefault(cut + "@all", {})
            if un == "all": P(f"-- pairs including tags outside Q M G O R: {extra}")
            if "i" in parts: P(f"-- (i) crossovers (v05) [{un}]"); run_crossover(cells, cut, uset, a.nb, a.np, rng, P, sub)
            if "ii" in parts: P(f"-- (ii) readable-map slopes (v08 MAIN) [{un}]"); run_slopes(cells, cut, uset, a.nbs or a.nb, a.nps, rng, P, sub)
        if "iii" in parts: P("-- (iii) judge-specific content lift (stats/c1)"); run_c1(cells, cut, LK, a.nb, rng, P, st_)
        if "iv" in parts or "v" in parts: P("-- (iv)/(v-R1) costs per 100 questions (regimes r02 / r02e) and installed judge-paid wrong answers"); run_costs(cells, cut, LK, a.nbc, rng, P, st_)
        if "v" in parts: P("-- (v-R2) dose top-minus-bottom (dose/d04)"); run_dose(cells, cut, a.nbc, rng, P, st_)
        P(f"   ({time.time() - t00:.0f}s)")
    open(f"{OUT}/ladder_{a.tag}.txt", "w").write("\n".join(L) + "\n")
    def clean(o):
        if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, np.ndarray): return clean(o.tolist())
        return o
    json.dump(clean(STORE), open(f"{OUT}/ladder_{a.tag}.json", "w"), indent=1)
    print(f"done ({time.time() - t00:.0f}s) -> {OUT}/ladder_{a.tag}.txt")

if __name__ == "__main__":
    main()
