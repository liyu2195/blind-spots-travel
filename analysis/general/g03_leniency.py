"""(explorer 'general') Does abstention on NEW questions follow the grader's leniency L for the small students and for other
recipes?  L = share of the untrained student's valid (non-IDK, well-formed) sampled answers to TRAINING questions that the
grader accepts (32 untrained samples per question, base file, IDK prompt); for a uniform random grader with false-accept rate FP
and false-reject rate FN, L = (1 - FN) r + FP (1 - r), r = strict-key accuracy of those samples (the trainer's strict key).
Abstention = share of new (held-out, de-duplicated) questions answered "I don't know" at the final checkpoint (greedy; strict
reading for the strict-format recipes, first-line reading for the free-format recipe).
Fits p = sigmoid(a + b L). Discovery: qwen7 KL-0.2 uniform graders (as in the draft) and, separately, phi4m KL-0.2 uniform graders.
Confirmation: llama8, smol3, granite; other recipes as residuals against the same-student KL-0.2 curve.
Output: g03_leniency.txt, g03_points.json."""
import json, os, sys, pickle, re, glob
import numpy as np
from scipy.optimize import curve_fit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gcommon import setup, sa, IDK, C, RL, BASE, HERE, read_eval

rng = np.random.default_rng(3)
SETS = {}
def S(ds):
    if ds not in SETS:
        probs, H1, HELD = setup(ds); SETS[ds] = (probs, H1, np.array(HELD))
    return SETS[ds]

RCACHE = {}
def base_r(st, ds):
    """strict-key accuracy r of valid untrained samples on training questions, and the share of samples that are valid"""
    if (st, ds) in RCACHE: return RCACHE[(st, ds)]
    probs, H1, _ = S(ds)
    f = f"{BASE}/{st}_{ds}_1500_n32_s1.jsonl"
    if not os.path.exists(f): f = f"{BASE}/{st}_{ds}_1500_idk_n32_s1.jsonl"
    nv = ns = nl = nall = 0
    for l in open(f):
        r = json.loads(l)
        if r["s"] < 0 or r["idx"] not in H1: continue
        nall += 1; x = sa(r)
        if not x or IDK.match(x): continue
        al = set(probs[r["idx"]]["aliases"]); nv += 1; ns += bool(C.tq_strict(x, al)); nl += bool(C.tq_correct(x, al))
    RCACHE[(st, ds)] = (ns / nv, nl / nv, nv / nall); return RCACHE[(st, ds)]

def sig(L, a, b): return 1 / (1 + np.exp(-(a + b * L)))
def fit(L, y):
    p, _ = curve_fit(sig, np.asarray(L), np.asarray(y), p0=(0.0, -5.0), maxfev=20000); return p
def r2(y, p): y = np.asarray(y); p = np.asarray(p); return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()

# ------------------------------------------------------------------ points
pts = []
NAME7 = re.compile(r"^(qwen7|llama8)_FK20(sim\d\dfn\d\d)_idkc05_s(\d+)_(h1|pq|nq)$")
for d in sorted(glob.glob(f"{RL}/qwen7_FK20sim*") + glob.glob(f"{RL}/llama8_FK20sim*")):
    m = NAME7.match(os.path.basename(d))
    if not m or not os.path.exists(f"{d}/eval_rl_ck120.jsonl"): continue
    st, g, seed, dt = m.groups(); ds = {"h1": "tqa", "pq": "pqa", "nq": "nqo"}[dt]
    mm = re.fullmatch(r"sim(\d\d)fn(\d\d)", g); fp, fn = int(mm.group(1)) / 100, int(mm.group(2)) / 100
    probs, H1, HELD = S(ds); gr, _ = read_eval(os.path.basename(d), 120)
    xs = [sa(gr[i]) for i in HELD]; a = np.mean([bool(x) and bool(IDK.match(x)) for x in xs]); bad = np.mean([not x for x in xs])
    rs, rl, vshare = base_r(st, ds)
    pts.append(dict(run=os.path.basename(d), st=st, ds=ds, rec="FK20", fam="key" if fp == fn == 0 else "sim", fp=fp, fn=fn, credit=0.5,
                    L=(1 - fn) * rs + fp * (1 - rs), a=float(a), bad=float(bad), seed=int(seed), cell=f"{st}_{ds}_{g}"))
for st in ("phi4m", "smol3", "granite"):
    P = pickle.load(open(f"{HERE}/cache_{st}.pkl", "rb"))
    for run, d in P.items():
        m = d["meta"]
        if m["fam"] not in ("key", "sim", "det", "hash") or m["outlet"] != "idk" or d["f"] is None: continue
        if m["g0"] in ("gold",): continue
        probs, H1, HELD = S(m["ds"]); rs, rl, vshare = base_r(st, m["ds"])
        fp, fn = m["fp"], m["fn"]
        L = (1 - fn) * rs + fp * (1 - rs)            # det: expected reward of an answer (partial credit f on wrong answers)
        code = d["f"]["cs"] if m["fmt"] == 1 else d["f"]["cf"]
        a = float((code[HELD] == 0).mean()); bad = float((code[HELD] == 3).mean())
        a_strict = float((d["f"]["cs"][HELD] == 0).mean())
        rec = (m["pre"] or "old") + ("" if m["adv"] == "std" or m["pre"] == "FK20D" else "-dr")
        pts.append(dict(run=run, st=st, ds=m["ds"], rec=rec, fam=m["fam"], fp=fp, fn=fn, credit=m["credit"], L=float(L), a=a,
                        a_strict=a_strict, bad=bad, seed=m["seed"], kl=m["kl"], steps=m["steps"], temp=m["temp"],
                        cell=f"{st}_{m['ds']}_{rec}_{m['g0']}_c{m['credit']}"))
json.dump(pts, open(f"{HERE}/g03_points.json", "w"), indent=0)

out = open(f"{HERE}/g03_leniency.txt", "w")
def P_(*a): print(*a, file=out); print(*a)
for st in ("qwen7", "llama8", "phi4m", "smol3", "granite"):
    for ds in ("tqa", "pqa", "nqo"):
        if (st, ds) in RCACHE: P_(f"untrained {st} {ds}: strict-key accuracy of valid samples r = {RCACHE[(st, ds)][0]:.3f} (loose {RCACHE[(st, ds)][1]:.3f}), valid share {RCACHE[(st, ds)][2]:.3f}")

def sel(**kw):
    return [p for p in pts if all((p.get(k) == v) if not callable(v) else v(p.get(k)) for k, v in kw.items())]
def cluster_boot_r2(train, test, nb=2000):
    a, b = fit([p["L"] for p in train], [p["a"] for p in train])
    yt = np.array([p["a"] for p in test]); pt = sig(np.array([p["L"] for p in test]), a, b)
    cells = sorted({p["cell"] for p in test}); idx = {c: [k for k, p in enumerate(test) if p["cell"] == c] for c in cells}
    bs = []
    for _ in range(nb):
        cc = rng.choice(cells, len(cells)); k = [j for c in cc for j in idx[c]]
        if len(set(yt[k])) > 1: bs.append(r2(yt[k], pt[k]))
    return (a, b), r2(yt, pt), np.percentile(bs, [2.5, 97.5]), float(np.mean(yt - pt))

ok = lambda p: p["bad"] <= 0.2
q7 = sel(st="qwen7", rec="FK20"); l8 = sel(st="llama8", rec="FK20")
P_(f"\n[A] 7-8B curve (discovery = qwen7 KL-0.2 uniform graders, {len(q7)} runs) -> confirmation sets")
for name, test in (("llama8 FK20", l8), ("phi4m FK20", sel(st="phi4m", rec="FK20", fam=lambda f: f in ("key", "sim"), credit=0.5)),
                   ("smol3 FK20", sel(st="smol3", rec="FK20", fam=lambda f: f in ("key", "sim"), credit=0.5)),
                   ("granite FK20", sel(st="granite", rec="FK20", fam=lambda f: f in ("key", "sim"), credit=0.5))):
    test = [p for p in test if ok(p)]
    if len(test) < 3: P_(f"  {name}: {len(test)} runs (too few)"); continue
    (a, b), R2, ci, bias = cluster_boot_r2(q7, test)
    P_(f"  {name}: {len(test)} runs, R^2 {R2:.3f} [{ci[0]:.2f}, {ci[1]:.2f}] (cluster bootstrap over grader cells), mean residual (obs - curve) {bias:+.3f}; curve a {a:.2f} b {b:.2f}")

P_("\n[B] small-student curve (discovery = phi4m KL-0.2 uniform graders, credit 0.5) -> smol3, granite, 7-8B")
ph = [p for p in sel(st="phi4m", rec="FK20", fam=lambda f: f in ("key", "sim"), credit=0.5) if ok(p)]
for name, test in (("smol3 FK20", sel(st="smol3", rec="FK20", fam=lambda f: f in ("key", "sim"), credit=0.5)),
                   ("granite FK20", sel(st="granite", rec="FK20", fam=lambda f: f in ("key", "sim"), credit=0.5)),
                   ("qwen7 FK20", q7), ("llama8 FK20", l8)):
    test = [p for p in test if ok(p)]
    (a, b), R2, ci, bias = cluster_boot_r2(ph, test)
    P_(f"  {name}: {len(test)} runs, R^2 {R2:.3f} [{ci[0]:.2f}, {ci[1]:.2f}], mean residual {bias:+.3f}; phi4m curve a {a:.2f} b {b:.2f} (n={len(ph)})")
sm = [p for p in sel(st="smol3", rec="FK20", fam=lambda f: f in ("key", "sim"), credit=0.5) if ok(p)]
(a, b), R2, ci, bias = cluster_boot_r2(sm, ph); P_(f"  reverse: smol3 curve -> phi4m: R^2 {R2:.3f} [{ci[0]:.2f}, {ci[1]:.2f}], residual {bias:+.3f}")
for st in ("phi4m", "smol3"):
    for ds in ("tqa", "pqa"):
        t = [p for p in sel(st=st, rec="FK20", ds=ds, fam=lambda f: f in ("key", "sim"), credit=0.5) if ok(p)]
        if len(t) >= 2: P_(f"  {st} {ds}: {len(t)} runs, L range {min(p['L'] for p in t):.2f}-{max(p['L'] for p in t):.2f}, abstention range {min(p['a'] for p in t):.2f}-{max(p['a'] for p in t):.2f}")

P_("\n[C] other recipes: residual (observed abstention minus the same student's KL-0.2 curve at the run's L), uniform graders, by recipe and credit")
curves = {}
for st in ("phi4m", "smol3", "granite"):
    base = [p for p in sel(st=st, rec="FK20", fam=lambda f: f in ("key", "sim"), credit=0.5) if ok(p)]
    if len(base) >= 4: curves[st] = fit([p["L"] for p in base], [p["a"] for p in base])
groups = {}
for p in pts:
    if p["st"] not in curves or p["fam"] not in ("key", "sim", "det"): continue
    k = (p["rec"], p["fam"] if p["fam"] == "det" else "uniform", p["credit"])
    groups.setdefault(k, []).append(p)
for k in sorted(groups, key=str):
    G = groups[k]
    res = np.array([p["a"] - sig(p["L"], *curves[p["st"]]) for p in G]); bads = np.array([p["bad"] for p in G])
    bs = [res[rng.integers(0, len(res), len(res))].mean() for _ in range(2000)] if len(res) > 1 else [res.mean()]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    collapsed = sum((p["a"] > 0.97) or (p["bad"] > 0.5) for p in G)
    P_(f"  recipe {k[0]:8s} {k[1]:7s} credit {k[2]}: {len(G):3d} runs ({', '.join(sorted({p['st'] for p in G}))}), mean residual {res.mean():+.3f} [{lo:+.3f}, {hi:+.3f}], "
       f"median {np.median(res):+.3f}; collapsed (all-IDK or >50% malformed) {collapsed}; mean malformed {bads.mean():.3f}")

P_("\n[D] matched graders across KL weights (same student, grader, credit 0.5, seeds pooled): abstention on new questions")
for st in ("phi4m", "smol3"):
    for g in ((0.0, 0.0), (0.0, 0.3), (0.0, 0.2), (0.35, 0.0)):
        line = f"  {st} FP {g[0]:.2f} FN {g[1]:.2f}:"
        for rec in ("FK04", "FK10", "FK20", "FK40", "F", "old", "FL2", "FT1", "FK20D", "old-dr"):
            G = [p for p in pts if p["st"] == st and p["rec"] == rec and p["fp"] == g[0] and p["fn"] == g[1] and p["credit"] == 0.5 and p["ds"] == "tqa" and p["fam"] in ("key", "sim")]
            if G: line += f" {rec} {np.mean([p['a'] for p in G]):.2f} (n={len(G)}{', bad %.2f' % np.mean([p['bad'] for p in G]) if np.mean([p['bad'] for p in G]) > 0.05 else ''})"
        P_(line)
out.close()
