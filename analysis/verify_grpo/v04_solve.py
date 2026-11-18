"""(verify_grpo) My own solver for the shared abstention rule (claim 2). Inputs per run: alpha_q (v02: the run's grader on the untrained replies),
the untrained IDK share and self-agreement (v01), and (c, beta, normaliser) from the configuration. No trained-run output is read.
Drift per training question (my derivation, v03): G_q(p) = F(p, alpha_q; c, norm) - kappa * beta * k * (p - pref_q), F = E[sum of IDK advantages],
evaluated EXACTLY by enumerating the 45 group compositions with the trainer's float32 advantage (no interpolation table).
Rules: u_q = logit(pref_q) + theta . phi_q;  conf: phi = [1, z_q], z = logit(agreement) standardised on the training questions;
bias: phi = [1]; conf2: [1, z, standardised logit pref]; free: every question its own 1-D fixed point (flow from pref_q).
Stationary theta: flow d theta/dt = mean_{q in training} G_q phi_q from theta = 0 (RK-free Euler with step control) until |g| < 1e-4, then
Newton polish with the analytic Jacobian; reports how many distinct roots a multi-start Newton finds (bistability of the shared rule).
Variant 'tcorr': KL term with the sampler mismatch of v03 section 6 (T 0.8 sampling, T 1 log-probs). Prediction: final greedy IDK iff p_q > 1/2
('half'), or p_q > (1 - p_q) * agreement_q ('greedy').
Smoothing as in the claim: pref = (nIDK + 0.5) / 33, agreement = (modal + 0.5) / (n_nonIDK + 1); 'sm2' = (nIDK + 1) / 34 and (modal + 1) / (n + 2).
Usage: nice -n 10 python3 -I v04_solve.py <st> <ds> -> cache/vp_<st>_<ds>.pkl"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, math, pickle, hashlib, numpy as np
st, ds = sys.argv[1], sys.argv[2]
PI, PN = (int(sys.argv[3].split("/")[0]), int(sys.argv[3].split("/")[1])) if len(sys.argv) > 3 else (0, 1)   # optional part i/N
HERE = (_RA + "/verify_grpo"); k = 8
B = dict(np.load(f"{HERE}/cache/vb_{st}_{ds}.npz")); D = pickle.load(open(f"{HERE}/cache/va_{st}_{ds}.pkl", "rb"))
home, new = B["home"].astype(bool), B["new"].astype(bool)
COMP = [(m, a, k - m - a) for m in range(k + 1) for a in range(k - m + 1)]
M_ = np.array([x[0] for x in COMP], float); A_ = np.array([x[1] for x in COMP], float); R_ = np.array([x[2] for x in COMP], float)
LC = np.array([math.lgamma(k + 1) - math.lgamma(m + 1) - math.lgamma(a + 1) - math.lgamma(r + 1) for m, a, r in COMP])
_S = {}
def Stab(c, norm):
    key = (round(c, 6), norm)
    if key not in _S:
        v = []
        for m, a, r in COMP:
            if m == 0: v.append(0.0); continue
            rw = np.array([c] * m + [1.0] * a + [0.0] * r, np.float32); sd = rw.std(ddof=1)
            if sd < 1e-6: v.append(0.0); continue
            adv = (rw - rw.mean()) / ((sd + np.float32(1e-4)) if norm == "std" else np.float32(1.0)); v.append(float(adv[:m].sum()))
        _S[key] = np.array(v)
    return _S[key]
def F_dF(p, al, c, norm):
    """F and dF/dp, vectorised over questions (p, al 1-D)."""
    p = np.clip(p, 1e-12, 1 - 1e-12)[:, None]; al = al[:, None]
    q1 = (1 - p) * al; q0 = (1 - p) * (1 - al)
    with np.errstate(divide="ignore"):
        lw = LC + M_ * np.log(p) + A_ * np.log(np.where(q1 > 0, q1, 1e-300)) + R_ * np.log(np.where(q0 > 0, q0, 1e-300))
    w = np.exp(lw); w = np.where(((A_ > 0) & (al <= 0)) | ((R_ > 0) & (al >= 1)), 0.0, w)
    S = Stab(c, norm); dlw = M_ / p - (A_ + R_) / (1 - p)
    return (w * S).sum(1), (w * dlw * S).sum(1)
sig = lambda u: 1 / (1 + np.exp(-u))
def kl_term(p, pref, beta, tcorr):
    """returns (value, d/dp) of the restoring term beta k (...)"""
    if not tcorr: return beta * k * (p - pref), beta * k * np.ones_like(p)
    pr1 = sig(0.8 * np.log(pref / (1 - pref)))
    def v(u):
        pp = sig(u); p1 = sig(0.8 * u); h = pp * (1 - p1) / p1 + (1 - pp) * p1 / (1 - p1); return beta * k * (p1 - pr1) * h
    pc = np.clip(p, 1e-12, 1 - 1e-12); u = np.clip(np.log(pc / (1 - pc)), -30, 30); du = 1e-4
    return v(u), (v(u + du) - v(u - du)) / (2 * du) / np.maximum(pc * (1 - pc), 1e-300)
def solve(alpha, u0, pref, Phi, c, norm, beta, tcorr=False, multistart=False):
    H = home & np.isfinite(alpha); al = np.where(np.isfinite(alpha), alpha, 0.5)
    def g_J(th):
        p = sig(u0[H] + Phi[H] @ th); Fv, dF = F_dF(p, al[H], c, norm); kv, dk = kl_term(p, pref[H], beta, tcorr)
        G = Fv - kv; dG = (dF - dk) * p * (1 - p)
        return Phi[H].T @ G / H.sum(), (Phi[H].T * dG) @ Phi[H] / H.sum()
    th = np.zeros(Phi.shape[1]); dt = 0.5; g, _ = g_J(th); it = 0; tflow = 0.0
    while np.abs(g).max() > 1e-4 and it < 20000 and tflow < 4000.0:     # flow time capped at 4000 (as the claim's solver: 8000 x 0.5)
        tn = th + dt * g; gn, _ = g_J(tn)
        if np.abs(gn).max() > 1.5 * np.abs(g).max() and dt > 1e-3: dt *= 0.5; continue
        th, g = tn, gn; it += 1; tflow += dt
        if it % 200 == 0: dt = min(dt * 1.5, 4.0)
    for _ in range(30):                                     # Newton polish
        g, J = g_J(th)
        if np.abs(g).max() < 1e-10: break
        try: step = np.linalg.solve(J, -g)
        except np.linalg.LinAlgError: break
        th = th + np.clip(step, -1, 1)
    g, J = g_J(th); stable = bool(np.isfinite(J).all() and np.all(np.real(np.linalg.eigvals(J)) < 0)) if np.isfinite(J).all() else False
    nroots = None
    if multistart:
        roots = []
        for s in np.linspace(-6, 6, 9):
            t2 = np.zeros(Phi.shape[1]); t2[0] = s
            for _ in range(60):
                g2, J2 = g_J(t2)
                if np.abs(g2).max() < 1e-9: break
                try: t2 = t2 + np.clip(np.linalg.solve(J2, -g2), -1, 1)
                except np.linalg.LinAlgError: break
            if np.abs(g_J(t2)[0]).max() < 1e-7 and not any(np.abs(t2 - r).max() < 1e-4 for r in roots): roots.append(t2)
        nroots = len(roots)
    return sig(u0 + Phi @ th), th, it, float(np.abs(g).max()), stable, nroots
UG = np.linspace(-12, 12, 481)
def solve_free(alpha, pref, c, norm, beta):
    """each question's own attractor of du/dt = G(sigmoid(u)) reached from u0 = logit(pref): grid scan in u, first sign change in the flow direction"""
    al = np.where(np.isfinite(alpha), alpha, 0.5); u0 = np.log(pref / (1 - pref)); n = len(al)
    Gg = np.empty((len(UG), n))
    for i, u in enumerate(UG):
        p = np.full(n, sig(u)); Fv, _ = F_dF(p, al, c, norm); Gg[i] = Fv - beta * k * (p - pref)
    G0 = F_dF(sig(u0), al, c, norm)[0] - beta * k * (sig(u0) - pref)
    uf = np.empty(n)
    for q in range(n):
        if G0[q] >= 0:
            idx = np.where((UG > u0[q]) & (Gg[:, q] < 0))[0]
            if len(idx) == 0: uf[q] = 12.0; continue
            j = idx[0]; ua, ga = (UG[j - 1], Gg[j - 1, q]) if UG[j - 1] > u0[q] else (u0[q], G0[q]); ub, gb = UG[j], Gg[j, q]
        else:
            idx = np.where((UG < u0[q]) & (Gg[:, q] > 0))[0]
            if len(idx) == 0: uf[q] = -12.0; continue
            j = idx[-1]; ua, ga = UG[j], Gg[j, q]; ub, gb = (UG[j + 1], Gg[j + 1, q]) if UG[j + 1] < u0[q] else (u0[q], G0[q])
        uf[q] = ua - ga * (ub - ua) / (gb - ga) if gb != ga else ua
    return sig(uf)
def feats(fmt, sm):
    I = B["i1"] if fmt == "1" else B["i0"]; m = B["m1"] if fmt == "1" else B["m0"]; nI = I.sum(1); nn = 32 - nI
    if sm == "sm2": pref = (nI + 1.0) / 34.0; agr = (m + 1.0) / (nn + 2.0)
    else: pref = (nI + 0.5) / 33.0; agr = (m + 0.5) / (nn + 1.0)
    return pref, agr
out = {}; cache = {}
for ri, (run, r) in enumerate(sorted(D["rows"].items())):
    if r["alpha"] is None or r["prompt"] != "idk": continue
    a = r["alpha"].astype(float); key = hashlib.md5(a.tobytes() + f"{r['c']}|{r['adv']}|{r['kl']}|{r['fmt']}".encode()).hexdigest()
    if int(key, 16) % PN != PI: continue
    if key in cache: out[run] = cache[key]; continue
    rec = {}
    for sm in ("sm1", "sm2"):
        pref, agr = feats(r["fmt"], sm); u0 = np.log(pref / (1 - pref)); z = np.log(agr / (1 - agr)); z = (z - z[home].mean()) / z[home].std()
        ur = (u0 - u0[home].mean()) / u0[home].std(); one = np.ones(len(u0))
        rules = {"conf": np.column_stack([one, z])} if (sm == "sm2" or r["kl"] <= 0) else {"bias": one[:, None], "conf": np.column_stack([one, z]), "conf2": np.column_stack([one, z, ur])}
        if sm == "sm2" and r["kl"] <= 0: continue
        for rule, Phi in rules.items():
            kaps = (1.0,) if (sm == "sm2" or rule != "conf" or r["kl"] <= 0) else (0.5, 0.75, 1.0, 1.5, 2.0)
            for kap in kaps:
                p, th, it, gres, stable, nr = solve(a, u0, pref, Phi, r["c"], r["adv"], kap * r["kl"], multistart=(rule == "conf" and kap == 1.0 and sm == "sm1" and r["kl"] > 0))
                tag = (sm, rule, kap)
                rec[tag] = dict(n_half=float((p[new] > 0.5).mean()), h_half=float((p[home] > 0.5).mean()), n_greedy=float((p[new] > (1 - p[new]) * agr[new]).mean()),
                                h_greedy=float((p[home] > (1 - p[home]) * agr[home]).mean()), n_meanp=float(p[new].mean()), theta=th.tolist(), iters=it, gres=gres,
                                stable=stable, nroots=nr)
                if rule == "conf" and kap == 1.0 and sm == "sm1":
                    rec["p_conf"] = p.astype(np.float32)
            if rule == "conf" and sm == "sm1" and r["kl"] > 0:
                p, th, it, gres, stable, _ = solve(a, u0, pref, Phi, r["c"], r["adv"], r["kl"], tcorr=True)
                rec[("sm1", "conf_tcorr", 1.0)] = dict(n_half=float((p[new] > 0.5).mean()), h_half=float((p[home] > 0.5).mean()), theta=th.tolist(), stable=stable, gres=gres)
        if sm == "sm1":
            p = solve_free(a, pref, r["c"], r["adv"], r["kl"])
            rec[("sm1", "free", 1.0)] = dict(n_half=float((p[new] > 0.5).mean()), h_half=float((p[home] > 0.5).mean()))
    cache[key] = rec; out[run] = rec
pickle.dump(out, open(f"{HERE}/cache/vp_{st}_{ds}" + (f"_p{PI}of{PN}" if PN > 1 else "") + ".pkl", "wb"))
nr = [v[("sm1", "conf", 1.0)]["nroots"] for v in out.values()]
print(f"{st} {ds}: {len(out)} runs ({len(cache)} distinct inputs); conf rule: unstable or unconverged solutions "
      f"{sum(1 for v in cache.values() if not v[('sm1', 'conf', 1.0)]['stable'] or v[('sm1', 'conf', 1.0)]['gres'] > 1e-6)}; "
      f"distinct roots found by multistart: {dict((x, nr.count(x)) for x in set(nr))}")
