"""(grpo) Mechanistic prediction of each run's final abstention from the UNTRAINED student's 32 samples and the grader's verdicts only.
Drift (g02_theory): per visit, the IDK logit u_q moves by lambda * G_q(p_q), G_q(p) = F(p, alpha_q; c, normaliser) - beta * k * (p - p_ref,q).
The policy cannot set u_q freely per question; it shifts the untrained IDK logit by a rule shared by all questions:
    u_q = logit(p_ref,q) + theta . phi_q
The rule's weights follow the expected update, d theta/dt = mean_{q in training set} G_q(p_q) phi_q (each training question is visited equally
often), integrated from theta = 0 (the untrained model) to its stationary point. Nothing is fitted to trained-run outcomes.
Rules (features phi_q, all from the untrained samples):  bias = [1];  conf = [1, z_q] with z_q = logit of the modal non-IDK answer's share
(self-agreement);  conf2 = [1, z_q, logit p_ref,q] (the untrained IDK logit also gets a free slope);  free = each question its own fixed point
(perfectly readable; flow from p_ref). Features are standardised on the training questions (affects the path only, not the stationary point).
Prediction: final greedy answer is IDK iff p_q > 1/2 ('half') or p_q > (1 - p_q) * agreement_q ('greedy': IDK beats the modal answer).
Optional single global constant kappa: effective KL = kappa * beta (kappa = 1 is the trainer's value).
Usage: nice -n 10 python3 -I g05_model.py <st> <ds> [kappa ...]  -> cache/pred_<st>_<ds>.pkl"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, json, pickle, numpy as np
from collections import Counter
sys.path.insert(0, (_RA + "/grpo"))
from g02_theory import F as F_exact
HERE = (_RA + "/grpo"); K = 8
st, ds = sys.argv[1], sys.argv[2]; KAPPAS = [float(x) for x in sys.argv[3:] if not x.startswith("part")] or [1.0]
PART = [x for x in sys.argv[3:] if x.startswith("part")]          # optional 'partI/N': only runs with index % N == I (output file suffixed)
PI, PN = (int(PART[0][4:].split("/")[0]), int(PART[0][4:].split("/")[1])) if PART else (0, 1)
PG = np.linspace(0, 1, 201); AG = np.linspace(0, 1, 201); _T = {}
def Ftab(c, norm):
    key = (round(c, 4), norm)
    if key not in _T: _T[key] = F_exact(PG[:, None], AG[None, :], c, norm)
    return _T[key]
def Fi(p, a, c, norm):
    T = Ftab(c, norm); x = np.clip(p, 0, 1) * 200; y = np.clip(a, 0, 1) * 200
    i = np.minimum(x.astype(int), 199); j = np.minimum(y.astype(int), 199); fx = x - i; fy = y - j
    return T[i, j] * (1 - fx) * (1 - fy) + T[i + 1, j] * fx * (1 - fy) + T[i, j + 1] * (1 - fx) * fy + T[i + 1, j + 1] * fx * fy
sig = lambda u: 1 / (1 + np.exp(-u))
B = dict(np.load(f"{HERE}/cache/base_{st}_{ds}.npz")); X = json.load(open(f"{HERE}/cache/base_{st}_{ds}_str.json"))["X"]
D = pickle.load(open(f"{HERE}/cache/runs_{st}_{ds}.pkl", "rb")); home, new = D["home"], D["new"]
def feats(idk):
    n_idk = idk.sum(1); pref = (n_idk + 0.5) / 33.0
    agr = np.array([(Counter(x for j, x in enumerate(X[k]) if not idk[k, j]).most_common(1)[0][1] if (~idk[k]).any() else 0) for k in range(len(X))])
    nn = (~idk).sum(1); a = (agr + 0.5) / (nn + 1.0)
    return pref, a
FE = {"1": feats(B["idk"]), "0": feats(B["idk_old"])}
def solve(alpha, pref, agr, c, norm, beta, rule, T=None):
    """stationary rule weights by integrating the expected drift; returns p on all questions."""
    u0 = np.log(pref / (1 - pref)); z = np.log(agr / (1 - agr)); z = (z - z[home].mean()) / (z[home].std() + 1e-9)
    if rule == "free":
        u = u0.copy(); dt = 0.25
        for it in range(8000):
            g = Fi(sig(u), alpha, c, norm) - beta * K * (sig(u) - pref); u = u + dt * g
            if np.abs(g).max() < 1e-6: break
        return sig(u), it
    ur = (u0 - u0[home].mean()) / (u0[home].std() + 1e-9)
    Phi = np.ones((len(u0), 1)) if rule == "bias" else (np.column_stack([np.ones(len(u0)), z]) if rule == "conf" else np.column_stack([np.ones(len(u0)), z, ur]))
    th = np.zeros(Phi.shape[1]); dt = 0.5; H = home & np.isfinite(alpha)
    for it in range(8000):
        p = sig(u0 + Phi @ th); g = Fi(p[H], alpha[H], c, norm) - beta * K * (p[H] - pref[H])
        step = Phi[H].T @ g / H.sum(); th = th + dt * step
        if np.abs(step).max() < 1e-7: break
    return sig(u0 + Phi @ th), it
out = {}
for ri, (run, r) in enumerate(sorted(D["rows"].items())):
    if r["alpha"] is None or ri % PN != PI: continue
    if r["kl"] <= 0 and r["fmt"] != "1": pass
    pref, agr = FE[r["fmt"]]
    alpha = r["alpha"].astype(float); ok = np.isfinite(alpha)
    rec = dict(run=run)
    for kap in KAPPAS:
        for rule in ("bias", "conf", "conf2", "free"):
            p, it = solve(np.where(ok, alpha, 0.5), pref, agr, r["c"], r["adv"], kap * r["kl"], rule)
            for thr in ("half", "greedy"):
                idk_pred = p > 0.5 if thr == "half" else p > (1 - p) * agr
                for part, m in (("h", home), ("n", new)):
                    rec[(kap, rule, thr, part)] = float(idk_pred[m].mean())
                    rec[(kap, rule, "mean_p", part)] = float(p[m].mean())
            rec[(kap, rule, "iters")] = it
    out[run] = rec
pickle.dump(out, open(f"{HERE}/cache/pred_{st}_{ds}" + (f"_p{PI}of{PN}" if PN > 1 else "") + ".pkl", "wb"))
print(f"{st} {ds}: predictions for {len(out)} runs, kappas {KAPPAS}")
