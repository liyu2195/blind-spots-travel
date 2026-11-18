"""(verify_grpo) Risky out-of-sample test 1: runs with one FORCED 'I don't know.' per group (FORCE=1, force_mode train: the forced reply is in the
loss with its own advantage and in the KL term; trainer lines 289-298, 392-396). Never used by the claim. The drift changes in two ways that the
theory fixes with no new constant:
  PG: the group always holds >= 1 IDK: E[S_I] over (1 forced + m sampled IDK, a paid, r unpaid), m + a + r = 7 ~ Multinomial(7; p, (1-p)alpha, (1-p)(1-alpha))
  KL: 7 sampled replies give 7 (p - p_ref) in expectation; the forced one is NOT sampled from pi: (1 - p_ref/p)(1 - p) (zero if log(p_ref/p) > 10, the clamp)
Shared rule conf, threshold 1/2, as in v04. Compared with the same judge's non-forced runs (judgeQt50, PopQA, same seeds), both students.
Usage: nice -n 10 python3 -I v07_force.py > out_v07_force.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, math, pickle, numpy as np
HERE = (_RA + "/verify_grpo")
res = []
for st in ("qwen7", "llama8"):
    sys.argv = ["x", st, "pq"]; g = {}
    exec(open(f"{HERE}/v04_solve.py").read().split("out = {}; cache = {}")[0], g)
    D, home, new, sig, Stab_ = g["D"], g["home"], g["new"], g["sig"], g["Stab"]
    k = 8
    C7 = [(m, a, 7 - m - a) for m in range(8) for a in range(8 - m)]
    M7 = np.array([x[0] for x in C7], float); A7 = np.array([x[1] for x in C7], float); R7 = np.array([x[2] for x in C7], float)
    L7 = np.array([math.lgamma(8) - math.lgamma(m + 1) - math.lgamma(a + 1) - math.lgamma(r + 1) for m, a, r in C7])
    def S_forced(c, norm):
        v = []
        for m, a, r in C7:
            rw = np.array([c] * (m + 1) + [1.0] * a + [0.0] * r, np.float32); sd = rw.std(ddof=1)
            if sd < 1e-6: v.append(0.0); continue
            adv = (rw - rw.mean()) / ((sd + np.float32(1e-4)) if norm == "std" else 1.0); v.append(float(adv[:m + 1].sum()))
        return np.array(v)
    def F_forced(p, al, S):
        p = np.clip(p, 1e-12, 1 - 1e-12)[:, None]; al = al[:, None]; q1 = (1 - p) * al; q0 = (1 - p) * (1 - al)
        lw = L7 + M7 * np.log(p) + A7 * np.log(np.where(q1 > 0, q1, 1e-300)) + R7 * np.log(np.where(q0 > 0, q0, 1e-300))
        w = np.exp(lw); w = np.where(((A7 > 0) & (al <= 0)) | ((R7 > 0) & (al >= 1)), 0.0, w); return (w * S).sum(1)
    pref, agr = g["feats"]("1", "sm1"); u0 = np.log(pref / (1 - pref)); z = np.log(agr / (1 - agr)); z = (z - z[home].mean()) / z[home].std()
    Phi = np.column_stack([np.ones(len(u0)), z])
    def solve_forced(alpha, c, norm, beta, kl_forced_as_sampled=False):
        S = S_forced(c, norm); H = home & np.isfinite(alpha); al = np.where(np.isfinite(alpha), alpha, 0.5)
        def gfun(th):
            p = sig(u0[H] + Phi[H] @ th); ratio = pref[H] / p
            klf = (p - pref[H]) if kl_forced_as_sampled else np.where(np.log(ratio) > 10, 0.0, (1 - ratio) * (1 - p))
            G = F_forced(p, al[H], S) - beta * (7 * (p - pref[H]) + klf)
            return Phi[H].T @ G / H.sum()
        th = np.zeros(2); dt = 0.25; gv = gfun(th)
        for it in range(40000):
            tn = th + dt * gv; gn = gfun(tn)
            if np.abs(gn).max() > 1.5 * np.abs(gv).max() and dt > 1e-4: dt *= 0.5; continue
            th, gv = tn, gn
            if np.abs(gv).max() < 1e-7: break
            if it % 300 == 0: dt = min(dt * 1.3, 2.0)
        return sig(u0 + Phi @ th), th, float(np.abs(gv).max())
    P = {}
    for pf in __import__("glob").glob(f"{HERE}/cache/vp_{st}_pq*.pkl"): P.update(pickle.load(open(pf, "rb")))
    for seed in (0, 1):
        rf = f"{st}_FK20judgeQt50f1_idkc05_s{seed}_pq"; r0 = f"{st}_FK20judgeQt50_idkc05_s{seed}_pq"
        if rf not in D["rows"] or r0 not in D["rows"]: print("missing", rf, r0); continue
        Rf, R0 = D["rows"][rf], D["rows"][r0]
        assert np.allclose(np.nan_to_num(Rf["alpha"]), np.nan_to_num(R0["alpha"]))
        pf_, th, gres = solve_forced(Rf["alpha"].astype(float), 0.5, "std", 0.2)
        pa_, _, _ = solve_forced(Rf["alpha"].astype(float), 0.5, "std", 0.2, kl_forced_as_sampled=True)
        y_f = float((Rf["out"][new] == 0).mean()); y_0 = float((R0["out"][new] == 0).mean())
        yh_f = float((Rf["out"][home] == 0).mean()); yh_0 = float((R0["out"][home] == 0).mean())
        pred0 = P[r0][("sm1", "conf", 1.0)]
        res.append(dict(st=st, seed=seed, act_forced=y_f, act_plain=y_0, pred_forced=float((pf_[new] > 0.5).mean()), pred_plain=pred0["n_half"], pred_forced_pgonly=float((pa_[new] > 0.5).mean()),
                        act_forced_h=yh_f, act_plain_h=yh_0, pred_forced_h=float((pf_[home] > 0.5).mean()), pred_plain_h=pred0["h_half"], gres=gres, theta=th.round(3).tolist()))
print("== forced-IDK runs (judgeQt50, PopQA, KL 0.2, c 0.5, std): actual vs predicted final abstention, new (and training) questions")
for r in res:
    print(f"   {r['st']:6s} s{r['seed']}: NEW actual plain {r['act_plain']:.3f} -> forced {r['act_forced']:.3f} (change {r['act_forced'] - r['act_plain']:+.3f}) | "
          f"predicted plain {r['pred_plain']:.3f} -> forced {r['pred_forced']:.3f} (change {r['pred_forced'] - r['pred_plain']:+.3f}) || "
          f"[forced PG, forced KL treated as sampled: {r['pred_forced_pgonly']:.3f}] || TRAIN actual {r['act_plain_h']:.3f} -> {r['act_forced_h']:.3f} | predicted {r['pred_plain_h']:.3f} -> {r['pred_forced_h']:.3f}  (theta {r['theta']}, |g| {r['gres']:.1e})")
