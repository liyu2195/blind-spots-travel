"""(dose) Tables from d04_dose.py (point estimates cache/d04_units.pkl + bootstrap draws cache/d04_boot*_<seed>.npy). See d04_dose.py for definitions.
Pooling: equal-weight mean over units (student x dataset x arm); 'pooled' = mean of the two students' means. Intervals: 2.5 / 97.5 percentiles
of the crossed bootstrap. Output: d04_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, glob, pickle, numpy as np
sys.path.insert(0, (_RA + "/dose"))
from d04_lib import *
U = pickle.load(open(f"{CACHE}/d04_units.pkl", "rb")); UNITS = U["UNITS"]; BUNITS = U["BUNITS"]; A1 = U["A1"]; A2 = U["A2"]; S = U["S"]
f1 = sorted(glob.glob(f"{CACHE}/d04_boot1_*.npy")); B1 = np.concatenate([np.load(f) for f in f1]) if f1 else np.zeros((0,) + A1.shape)
f2 = sorted(glob.glob(f"{CACHE}/d04_boot2_*.npy")); B2 = np.concatenate([np.load(f) for f in f2]) if f2 else np.zeros((0,) + A2.shape)
JUNITS = U["JUNITS"]; A3 = U["A3"]; f3 = sorted(glob.glob(f"{CACHE}/d04_boot3_*.npy")); B3 = np.concatenate([np.load(f) for f in f3]) if f3 else np.zeros((0,) + A3.shape)
JIX = {u: i for i, u in enumerate(JUNITS)}
P = Log(f"{HERE}/d04_out.txt"); P(__doc__); P(open(f"{HERE}/d04_dose.py").read().split('"""')[1]); P(f"bootstrap draws: {len(B1)} (files {len(f1)})")
UIX = {u: i for i, u in enumerate(UNITS)}; BIX = {u: i for i, u in enumerate(BUNITS)}
ST = {"sl": 0, "sl_ref": 1, "pc": 2, "pc_ref": 3, "tb": 4, "tb_ref": 5}
STRICT = [f"J{J}_a35" for J in J5]; MID = ["JQ_a50"]; DEF = [f"J{J}_t50" for J in J5]
def sel(arms, pt, k, who=None, dsf=None, U_=UNITS):
    return [u for u in U_ if u[2] in arms and u[3] == pt and u[4] == k and (who is None or u[0] == who) and (dsf is None or u[1] == dsf)]
def val(arr, idx, nm):
    """arr: (..., units, NST) -> (..., len(idx)) for stat nm ('x_d' = x - x_ref)"""
    if nm.endswith("_d"): b = nm[:-2]; return arr[..., idx, ST[b]] - arr[..., idx, ST[b + "_ref"]]
    return arr[..., idx, ST[nm]]
def pool(arms, nm, pt="new", k="lin16", who=None, dsf=None):
    """returns point, lo, hi, n units"""
    whos = [who] if who else list(STS); means0 = []; meansb = []; n = 0
    for w_ in whos:
        us = sel(arms, pt, k, w_, dsf)
        idx = [UIX[u] for u in us if np.isfinite(A1[UIX[u], ST[nm.replace('_d', '')]])]
        if not idx: continue
        n += len(idx); means0.append(np.nanmean(val(A1, idx, nm))); meansb.append(np.nanmean(val(B1, idx, nm), axis=1) if len(B1) else np.array([np.nan]))
    if not means0: return None
    v = np.mean(means0); b = np.mean(meansb, 0)
    return v, np.nanpercentile(b, 2.5), np.nanpercentile(b, 97.5), n
def fmt(r, d=3): return "n/a" if r is None else f"{r[0]:+.{d}f} [{r[1]:+.{d}f}, {r[2]:+.{d}f}]"
def poolbins(arms, pt="new", k="lin16", who=None, which="d"):
    whos = [who] if who else list(STS); m0 = []; mb = []
    for w_ in whos:
        idx = [BIX[u] for u in sel(arms, pt, k, w_, None, BUNITS)]
        idx = [i for i in idx if np.all(np.isfinite(A2[i]))]
        if not idx: continue
        f = (lambda A: A[..., idx, 0, :] - A[..., idx, 1, :]) if which == "d" else ((lambda A: A[..., idx, 0, :]) if which == "run" else (lambda A: A[..., idx, 1, :]))
        m0.append(np.nanmean(f(A2), 0)); mb.append(np.nanmean(f(B2), 1) if len(B2) else np.full((1, 5), np.nan))
    if not m0: return None
    v = np.mean(m0, 0); b = np.mean(mb, 0); return v, np.nanpercentile(b, 2.5, 0), np.nanpercentile(b, 97.5, 0)

P("\n== T1 PRIMARY curve: lin16 own-grader predicted rejection, strict judges (a35), NEW known-sure answers; quintiles of x_perp (generic removed)")
for who in (None,) + STS:
    r = poolbins(STRICT, who=who, which="run"); rr = poolbins(STRICT, who=who, which="ref"); d = poolbins(STRICT, who=who, which="d")
    P(f"  {who or 'pooled'} ({len(sel(STRICT, 'new', 'lin16', who))} units): adjusted refusal of trained runs | matched reference | grader-specific")
    for j in range(5):
        P(f"    Q{j + 1}: {r[0][j]:.3f} [{r[1][j]:.3f}, {r[2][j]:.3f}] | {rr[0][j]:.3f} [{rr[1][j]:.3f}, {rr[2][j]:.3f}] | {d[0][j]:+.3f} [{d[1][j]:+.3f}, {d[2][j]:+.3f}]")
    P(f"    T-B: trained {fmt(pool(STRICT, 'tb', who=who))}  reference {fmt(pool(STRICT, 'tb_ref', who=who))}  grader-specific {fmt(pool(STRICT, 'tb_d', who=who))}")
    P(f"    slope per SD: trained {fmt(pool(STRICT, 'sl', who=who))}  reference {fmt(pool(STRICT, 'sl_ref', who=who))}  grader-specific {fmt(pool(STRICT, 'sl_d', who=who))}")

P("\n== T2 grader-specific T-B and slope per SD (lin16) by operating point, student and dataset")
for lab, arms in (("strict a35 (Q M G O R)", STRICT), ("middle Q a50", MID), ("default t50 (Q all cells; M G O where run)", DEF)):
    r = pool(arms, "tb_d"); P(f"  {lab}: pooled T-B {fmt(r)} slope {fmt(pool(arms, 'sl_d'))} ({r[3] if r else 0} units); raw T-B of trained runs {fmt(pool(arms, 'tb'))}")
    for who in STS:
        parts = []
        for ds in DSS:
            r = pool(arms, "tb_d", who=who, dsf=ds)
            if r is None: continue
            parts.append(f"{ds}: T-B {fmt(r, 2)} slope {fmt(pool(arms, 'sl_d', who=who, dsf=ds), 3)} (n {r[3]})")
        P(f"    {who}: all T-B {fmt(pool(arms, 'tb_d', who=who))} | " + " | ".join(parts))
P("  per judge (strict, pooled over cells): " + "; ".join(f"{A[1]} T-B {fmt(pool([A], 'tb_d'), 2)} slope {fmt(pool([A], 'sl_d'), 3)}" for A in STRICT if pool([A], 'tb_d')))

P("\n== T3 comparisons (each arm's own target; lin16): key-trained, Sonnet-trained, planted simulated graders (PopQA)")
for A in ("key", "son", "fnB80", "fnB50", "fnA80", "fnH80", "fnH60", "plrealM", "plrealO", "plscrM", "plsharpM", "plsharpO"):
    r = pool([A], "tb_d")
    if r is None: continue
    P(f"  {A:9s} ({r[3]} units): T-B trained {fmt(pool([A], 'tb'))} reference {fmt(pool([A], 'tb_ref'))} specific {fmt(r)} | slope specific {fmt(pool([A], 'sl_d'))}"
      + "".join(f" | {w_} T-B {pool([A], 'tb_d', who=w_)[0]:+.3f}" for w_ in STS if pool([A], 'tb_d', who=w_)))
for A in ("fnB80", "fnH80", "plrealM", "plscrM"):
    c = poolbins([A], which="d"); r = poolbins([A], which="run")
    if c is not None: P(f"  {A} curve (trained, Q1..Q5): " + " ".join(f"{x:.2f}" for x in r[0]) + " | grader-specific: " + " ".join(f"{x:+.2f}" for x in c[0]))

P("\n== T4 predictor comparison on NEW known-sure answers")
P("   vAUC = AUROC for the grader's actual verdict on new questions (all valid greedy answers / known-sure answers only);")
P("   rAUC = mean over runs of AUROC for refusal (raw score / score residualised on G); pcor = partial correlation with refusal after G;")
P("   spec = grader-specific slope per SD (run minus matched reference); 'vs lin16' = paired bootstrap difference of pcor")
VK = ("lin16fr", "tfidffr", "lin16G2", "v_bin", "v_share", "v_py")
for lab, arms in (("strict a35", STRICT), ("middle Q a50", MID), ("key", ["key"]), ("Sonnet", ["son"]), ("default t50", DEF), ("fnB80 (type set)", ["fnB80"]), ("plrealM", ["plrealM"]), ("plscrM", ["plscrM"]), ("fnH80", ["fnH80"]), ("plsharpM", ["plsharpM"])):
    P(f"  -- {lab}")
    for k in KINDS + VK:
        r = pool(arms, "pc", k=k)
        if r is None: continue
        us = sel(arms, "new", k)
        va = [S[(u[0], u[1])]["vauc"].get((k, u[5]), (np.nan, np.nan)) for u in us]
        au = [U["AUC"].get((u[0], u[1], u[2], k), (np.nan, np.nan)) for u in us]
        dd = []
        for w_ in STS:
            a_ = [UIX[u] for u in sel(arms, "new", k, w_)]; b_ = [UIX[(u[0], u[1], u[2], u[3], "lin16", u[5])] for u in sel(arms, "new", k, w_)]
            if a_: dd.append((np.nanmean(val(A1, a_, "pc")) - np.nanmean(val(A1, b_, "pc")), np.nanmean(val(B1, a_, "pc"), 1) - np.nanmean(val(B1, b_, "pc"), 1)))
        d0 = np.mean([x[0] for x in dd]); db = np.mean([x[1] for x in dd], 0)
        pw = {w_: pool(arms, 'pc', k=k, who=w_) for w_ in STS}; pws = ", ".join(f"{w_} {pw[w_][0]:+.3f}" for w_ in STS if pw[w_])
        P(f"    {k:8s} vAUC {np.nanmean([x[0] for x in va]):.3f}/{np.nanmean([x[1] for x in va]):.3f} rAUC {np.nanmean([x[0] for x in au]):.3f}/{np.nanmean([x[1] for x in au]):.3f} | "
          f"pcor {fmt(r)} ({pws}) | spec {fmt(pool(arms, 'sl_d', k=k))} | "
          f"vs lin16 {d0:+.3f} [{np.nanpercentile(db, 2.5):+.3f}, {np.nanpercentile(db, 97.5):+.3f}]")

P("\n== T4b partial correlation of refusal with each predictor by dataset (strict a35 arms, new known-sure answers; students pooled, then per student)")
for ds in DSS:
    parts = []
    for k in ("lin16", "lin20", "lin16p64", "mlp16", "knn16", "tfidf", "knnE", "v_bin", "v_share"):
        r = pool(STRICT, "pc", k=k, dsf=ds)
        if r is None: continue
        pw = [pool(STRICT, "pc", k=k, who=w_, dsf=ds) for w_ in STS]
        parts.append(f"{k} {r[0]:+.2f} [{r[1]:+.2f},{r[2]:+.2f}] (" + "/".join(f"{x[0]:+.2f}" for x in pw if x) + ")")
    P(f"  {ds}: " + " | ".join(parts))

P("\n== T5 HOME (training questions, out-of-fold predictors) vs NEW: partial correlation of refusal with readable prediction vs actual verdict")
for lab, arms in (("strict a35", STRICT), ("middle Q a50", MID), ("key", ["key"])):
    for k in ("lin16", "lin16fr", "lin16p64", "mlp16", "tfidf", "knnE", "v_bin", "v_share"):
        h = pool(arms, "pc", "home", k); n_ = pool(arms, "pc", "new", k)
        if h is None: continue
        P(f"  {lab:11s} {k:8s} pcor home {fmt(h)} new {fmt(n_)} | spec slope home {fmt(pool(arms, 'sl_d', 'home', k))} new {fmt(pool(arms, 'sl_d', 'new', k))}"
          + "".join(f" | {w_} home {pool(arms, 'pc', 'home', k, w_)[0]:+.3f} new {pool(arms, 'pc', 'new', k, w_)[0]:+.3f}" for w_ in STS if pool(arms, 'pc', 'home', k, w_)))
for lab, arms in (("strict a35", STRICT),):
    for kr in ("lin16", "lin16p64"):
        for kv in ("v_bin", "v_share"):
            rr = []
            for pt in ("home", "new"):
                whos = []
                for w_ in STS:
                    a_ = [UIX[u] for u in sel(arms, pt, kr, w_)]; b_ = [UIX[(u[0], u[1], u[2], u[3], kv, u[5])] for u in sel(arms, pt, kr, w_)]
                    whos.append((np.nanmean(val(A1, a_, "pc")) - np.nanmean(val(A1, b_, "pc")), np.nanmean(val(B1, a_, "pc"), 1) - np.nanmean(val(B1, b_, "pc"), 1)))
                v0 = np.mean([x[0] for x in whos]); bb = np.mean([x[1] for x in whos], 0); rr.append((v0, bb))
            dd = rr[1][1] - rr[0][1]
            P(f"  {lab} pcor({kr}) - pcor({kv}): home {rr[0][0]:+.3f} [{np.nanpercentile(rr[0][1], 2.5):+.3f}, {np.nanpercentile(rr[0][1], 97.5):+.3f}]  "
              f"new {rr[1][0]:+.3f} [{np.nanpercentile(rr[1][1], 2.5):+.3f}, {np.nanpercentile(rr[1][1], 97.5):+.3f}]  change new - home {rr[1][0] - rr[0][0]:+.3f} [{np.nanpercentile(dd, 2.5):+.3f}, {np.nanpercentile(dd, 97.5):+.3f}]")

P("\n== T6 own judge vs other judges' predicted rejection (lin16): grader-specific slope per SD on new known-sure answers (same runs, same answers)")
for lab, arms in (("strict a35", STRICT), ("middle Q a50", MID)):
    d0 = []; db = []; o0 = []; ob = []; w0 = []; wb = []
    for w_ in STS:
        us = [u for u in sel(arms, "new", "lin16", w_) if any(x[:4] == u[:4] and x[4].startswith("x_") for x in UNITS)]
        if not us: continue
        own_i = [UIX[u] for u in us]; xs = [[UIX[x] for x in UNITS if x[:4] == u[:4] and x[4].startswith("x_")] for u in us]
        oth0 = np.array([np.nanmean(val(A1, x, "sl_d")) for x in xs]); othb = np.array([np.nanmean(val(B1, x, "sl_d"), 1) for x in xs])   # units, (units x draws)
        own0 = val(A1, own_i, "sl_d"); ownb = val(B1, own_i, "sl_d").T
        w0.append(np.nanmean(own0)); wb.append(np.nanmean(ownb, 0)); o0.append(np.nanmean(oth0)); ob.append(np.nanmean(othb, 0))
        d0.append(np.nanmean(own0 - oth0)); db.append(np.nanmean(ownb - othb, 0))
    if d0:
        f = lambda a, b: f"{np.mean(a):+.3f} [{np.nanpercentile(np.mean(b, 0), 2.5):+.3f}, {np.nanpercentile(np.mean(b, 0), 97.5):+.3f}]"
        P(f"  {lab}: own judge {f(w0, wb)}; other judges' predictions {f(o0, ob)}; own minus other {f(d0, db)}")
P.save()

P("\n== T7 joint model on known-sure answers: refusal ~ G + readable prediction + actual verdict (coefficients per SD, refusal points)")
P("   br / bv = readable / verdict coefficient of the trained runs; '-ref' = minus the matched reference runs; HOME uses out-of-fold readable predictions")
JS = {"br": 0, "br_ref": 1, "bv": 2, "bv_ref": 3, "corr": 4}
def jpool(arms, nm, pt, kk, who=None):
    whos = [who] if who else list(STS); m0 = []; mb = []
    for w_ in whos:
        idx = [JIX[u] for u in JUNITS if u[2] in arms and u[3] == pt and u[4] == kk and u[0] == w_ and np.isfinite(A3[JIX[u], 0])]
        if not idx: continue
        if nm.endswith("_d"): b = nm[:-2]; f = lambda A: A[..., idx, JS[b]] - A[..., idx, JS[b + "_ref"]]
        elif nm == "gap": f = lambda A: A[..., idx, 0] - A[..., idx, 2]
        elif nm == "gap_d": f = lambda A: (A[..., idx, 0] - A[..., idx, 1]) - (A[..., idx, 2] - A[..., idx, 3])
        else: f = lambda A: A[..., idx, JS[nm]]
        m0.append(np.nanmean(f(A3))); mb.append(np.nanmean(f(B3), 1) if len(B3) else np.array([np.nan]))
    if not m0: return None
    b = np.mean(mb, 0); return np.mean(m0), np.nanpercentile(b, 2.5), np.nanpercentile(b, 97.5)
for lab, arms in (("strict a35", STRICT), ("middle Q a50", MID), ("key", ["key"])):
    for kk in ("lin16+v_share", "lin16+v_py", "lin16fr+v_share", "lin16p64+v_share"):
        for pt in ("home", "new"):
            r = jpool(arms, "br", pt, kk)
            if r is None: continue
            P(f"  {lab:11s} {kk:16s} {pt:4s}: readable {fmt(r)} verdict {fmt(jpool(arms, 'bv', pt, kk))} | readable-ref {fmt(jpool(arms, 'br_d', pt, kk))} verdict-ref {fmt(jpool(arms, 'bv_d', pt, kk))} | "
              f"readable minus verdict {fmt(jpool(arms, 'gap', pt, kk))} | corr(read, verdict) {jpool(arms, 'corr', pt, kk)[0]:.2f}"
              + "".join(f" | {w_} {jpool(arms, 'br', pt, kk, w_)[0]:+.3f}/{jpool(arms, 'bv', pt, kk, w_)[0]:+.3f}" for w_ in STS if jpool(arms, 'br', pt, kk, w_)))
P.save()

# plot data for d06_fig.py
PLOT = dict(curve={who or "pooled": dict(run=poolbins(STRICT, who=who, which="run"), ref=poolbins(STRICT, who=who, which="ref"), d=poolbins(STRICT, who=who, which="d")) for who in (None,) + STS},
            ops={lab: poolbins(arms, which="d") for lab, arms in (("strict a35", STRICT), ("middle Q a50", MID), ("default t50", DEF))},
            arms={A: pool([A], "tb_d") for A in ("key", "son", "fnB80", "fnB50", "fnH80", "fnH60", "plrealM", "plrealO", "plscrM", "plsharpM", "plsharpO")},
            strict_tb=pool(STRICT, "tb_d"), mid_tb=pool(MID, "tb_d"), def_tb=pool(DEF, "tb_d"),
            pcor_new={k: pool(STRICT, "pc", "new", k) for k in KINDS + ("v_bin", "v_share", "v_py")},
            pcor_home={k: pool(STRICT, "pc", "home", k) for k in ("lin16", "lin16p64", "mlp16", "tfidf", "knnE", "v_bin", "v_share")})
pickle.dump(PLOT, open(f"{HERE}/d04_plot.pkl", "wb"))
