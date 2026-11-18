"""(grpo) Expected per-visit drift of the IDK logit under the trainer's update (st_grpo_b.py lines 385-416), exact by enumeration.
Model: on question q the policy says IDK with probability p (logit u); otherwise it answers, and an answer gets reward 1 with probability alpha
(the grader's acceptance of the policy's current answers, invalid replies = 0); IDK gets the credit c. Group of k = 8.
Trainer: adv = (r - mean) / (std_unbiased + 1e-4) ('std') or (r - mean) ('none'); adv = 0 if std < 1e-6; loss = -sum adv * logp / ntok
+ beta * sum k3 / ntok, k3 = exp(d) - d - 1, d = logp_ref - logp.
Derivation (two-level softmax: u decides IDK vs answer, the answer distribution is separate):
  d/du of sum_i A_i log pi(y_i) = (1-p) S_I - p sum_{ans} A_i = S_I     (sum_i A_i = 0),  S_I = sum of the IDK samples' advantages
  E over sampling of the k3 gradient on u = k (p - p_ref)                  (the k3 loss gradient is the REVERSE KL gradient)
  => E[du per visit] = lambda * [ F(p, alpha, c) - beta k (p - p_ref) ],  F = E_composition[S_I].
  'none': F = (k-1) p (1-p) (c - alpha) exactly (unbiased REINFORCE with leave-in mean); 'std': enumerate (m IDK, a accepted, r rejected).
Usage: python3 -I g02_theory.py > out_g02_theory.txt"""
import numpy as np, math, itertools
k = 8; EPS = 1e-4
def comps():
    for m in range(k + 1):
        for a in range(k - m + 1):
            yield m, a, k - m - a
CMP = list(comps())
def SI(m, a, r, c, norm):
    if m == 0: return 0.0
    rw = np.array([c] * m + [1.0] * a + [0.0] * r); mu = rw.mean(); sd = rw.std(ddof=1)
    if sd < 1e-6: return 0.0
    return m * (c - mu) / ((sd + EPS) if norm == "std" else 1.0)
def multi(m, a, r, p, al):
    return math.factorial(k) / (math.factorial(m) * math.factorial(a) * math.factorial(r)) * p ** m * ((1 - p) * al) ** a * ((1 - p) * (1 - al)) ** r
_cache = {}
def table(c, norm):
    key = (round(c, 6), norm)
    if key not in _cache: _cache[key] = np.array([SI(m, a, r, c, norm) for m, a, r in CMP])
    return _cache[key]
M_ = np.array([x[0] for x in CMP]); A_ = np.array([x[1] for x in CMP]); R_ = np.array([x[2] for x in CMP])
LOGC = np.array([math.lgamma(k + 1) - math.lgamma(m + 1) - math.lgamma(a + 1) - math.lgamma(r + 1) for m, a, r in CMP])
def F(p, al, c, norm):
    """E[S_I]; p, al broadcastable arrays."""
    p = np.asarray(p, float)[..., None]; al = np.asarray(al, float)[..., None]
    with np.errstate(divide="ignore", invalid="ignore"):
        lw = LOGC + M_ * np.log(np.clip(p, 1e-300, 1)) + A_ * np.log(np.clip((1 - p) * al, 1e-300, 1)) + R_ * np.log(np.clip((1 - p) * (1 - al), 1e-300, 1))
    w = np.exp(lw); w = np.where(((A_ > 0) & (al <= 0)) | ((R_ > 0) & (al >= 1)) | ((M_ > 0) & (p <= 0)), 0.0, w)
    return (w * table(c, norm)).sum(-1)
def G(p, al, c, norm, beta, pref):
    return F(p, al, c, norm) - beta * k * (np.asarray(p) - pref)
def fixed_points(al, c, norm, beta, pref, n=4001):
    """roots of G in (0, 1) with their stability; the end points count as stable attractors when G points into them."""
    ps = np.linspace(1e-6, 1 - 1e-6, n); g = G(ps, al, c, norm, beta, pref); out = []
    if g[0] < 0: out.append((0.0, "stable"))
    for i in range(n - 1):
        if g[i] * g[i + 1] < 0:
            p0 = ps[i] - g[i] * (ps[i + 1] - ps[i]) / (g[i + 1] - g[i]); out.append((p0, "stable" if g[i] > 0 else "unstable"))
    if g[-1] > 0: out.append((1.0, "stable"))
    return out
def flow(al, c, norm, beta, pref, p0, T=4000.0, dt=0.05):
    """integrate du/dt = G(sigmoid(u)) from p0 (vectorised over al / pref arrays)."""
    u = np.log(p0 / (1 - p0)) * np.ones(np.broadcast(al, pref).shape)
    for _ in range(int(T / dt) if False else 0): pass
    t = 0.0
    while t < T:
        p = 1 / (1 + np.exp(-u)); du = G(p, al, c, norm, beta, pref); u = u + dt * du; t += dt
        if np.max(np.abs(du)) < 1e-7: break
    return 1 / (1 + np.exp(-u))

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=160)
    print("== 0. checks")
    for c in (0.3, 0.5, 0.7):          # closed form for 'none'
        for p, al in ((0.1, 0.2), (0.4, 0.7), (0.8, 0.5)):
            print(f"   none c {c} p {p} alpha {al}: enumerated F {F(p, al, c, 'none'):+.5f}  closed form 7p(1-p)(c-alpha) {7 * p * (1 - p) * (c - al):+.5f}")
    rng = np.random.default_rng(0)        # Monte Carlo with the trainer's own formulas (float32 as in torch)
    for c, p, al in ((0.5, 0.2, 0.3), (0.7, 0.3, 0.6)):
        S = []
        for _ in range(20000):
            idk = rng.random(k) < p; acc = rng.random(k) < al
            r = np.where(idk, c, acc.astype(float)).astype(np.float32); sd = r.std(ddof=1)
            A = (r - r.mean()) / (sd + 1e-4) if sd >= 1e-6 else np.zeros(k)
            S.append(A[idk].sum())
        print(f"   std c {c} p {p} alpha {al}: Monte Carlo E[S_I] {np.mean(S):+.4f} (se {np.std(S) / np.sqrt(len(S)):.4f})  enumerated {F(p, al, c, 'std'):+.4f}")
    # k3 gradient identity on a first-token softmax: E_pi[(1 - pi_ref/pi) d log pi / du] = p - p_ref
    for p, pr in ((0.3, 0.05), (0.05, 0.3)):
        e = p * (1 - pr / p) * (1 - p) + (1 - p) * (1 - (1 - pr) / (1 - p)) * (-p)
        print(f"   k3 loss gradient on u, p {p} p_ref {pr}: {e:+.4f}  = p - p_ref {p - pr:+.4f}")

    print("\n== 1. group-level: advantage of ONE IDK in a group with 7 answers, a of them accepted (std | none)")
    for c in (0.1, 0.3, 0.5, 0.7, 0.9):
        print(f"   c {c}: std " + " ".join(f"{SI(1, a, 7 - a, c, 'std'):+.2f}" for a in range(8)) + " | none " + " ".join(f"{SI(1, a, 7 - a, c, 'none'):+.3f}" for a in range(8)))
    print("   two-level groups (no accepted, or no rejected answer): the std advantage does not depend on c; m IDK + (8-m) rejected:",
          " ".join(f"m{m} {SI(m, 0, k - m, 0.3, 'std'):+.3f}/{SI(m, 0, k - m, 0.8, 'std'):+.3f}" for m in (1, 2, 4, 7)))

    print("\n== 2. F(p, alpha, c) = E[S_I] (std) on a grid; rows alpha, columns p 0.05 0.2 0.5 0.8 0.95")
    for c in (0.3, 0.5, 0.7, 0.9):
        print(f"  c {c}")
        for al in (0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0):
            print(f"    alpha {al:.2f}: std " + " ".join(f"{F(p, al, c, 'std'):+.3f}" for p in (0.05, 0.2, 0.5, 0.8, 0.95)) +
                  "   none " + " ".join(f"{F(p, al, c, 'none'):+.3f}" for p in (0.05, 0.2, 0.5, 0.8, 0.95)))

    print("\n== 3. escape from rare IDK: s1 = E[S_I | one IDK in the group] must exceed beta (KL 'force' beta*k*p vs PG k*p*s1)")
    als = np.linspace(0, 1, 1001)
    s1 = lambda al, c, norm: sum(math.comb(7, a) * al ** a * (1 - al) ** (7 - a) * SI(1, a, 7 - a, c, norm) for a in range(8))
    for norm in ("std", "none"):
        for beta in (0.0, 0.2, 0.5):
            row = []
            for c in (0.1, 0.3, 0.5, 0.7, 0.85, 0.9):
                v = np.array([s1(a, c, norm) for a in als]); ok = als[v > beta]
                row.append(f"c {c}: {ok.max() if len(ok) else float('nan'):.3f}")
            print(f"   {norm:4s} beta {beta}: escape iff alpha < " + "  ".join(row))

    print("\n== 4. stable fixed points p* of G(p) = F - beta k (p - p_ref), p_ref 0.02; alpha* = largest alpha with an upper fixed point p* > 0.5")
    for norm in ("std", "none"):
        for beta in (0.0, 0.04, 0.1, 0.2, 0.4, 0.5, 1.0):
            row = []
            for c in (0.1, 0.3, 0.5, 0.7, 0.85, 0.9):
                best = np.nan
                for al in np.linspace(0, 1, 201):
                    fp = [x for x, s in fixed_points(al, c, norm, beta, 0.02, n=801) if s == "stable" and x > 0.5]
                    if fp: best = al
                row.append(f"{best:.3f}")
            print(f"   {norm:4s} beta {beta:4.2f}: alpha*(c = 0.1 0.3 0.5 0.7 0.85 0.9) = " + " ".join(row))
    print("   p* at alpha 0 (upper stable point), c 0.5: std", {b: round(max([x for x, s in fixed_points(0.0, 0.5, 'std', b, 0.02) if s == 'stable']), 3) for b in (0.0, 0.2, 0.5, 1.0)},
          "none", {b: round(max([x for x, s in fixed_points(0.0, 0.5, 'none', b, 0.02) if s == 'stable']), 3) for b in (0.0, 0.2, 0.5, 1.0)})
    print("   bistability check (std, beta 0.2, c 0.5, p_ref 0.02): fixed points by alpha:")
    for al in (0.3, 0.4, 0.45, 0.5, 0.55, 0.6):
        print(f"     alpha {al}: " + ", ".join(f"{x:.3f} {s}" for x, s in fixed_points(al, 0.5, "std", 0.2, 0.02)))

    print("\n== 5. credit sensitivity of the boundary: d alpha*/dc (finite difference c 0.5 -> 0.7 / 0.85), beta 0.2")
    for norm in ("std", "none"):
        res = {}
        for c in (0.5, 0.7, 0.85):
            best = np.nan
            for al in np.linspace(0, 1, 401):
                fp = [x for x, s in fixed_points(al, c, norm, 0.2, 0.02, n=801) if s == "stable" and x > 0.5]
                if fp: best = al
            res[c] = best
        print(f"   {norm}: alpha* {res}; slope 0.5->0.7 {(res[0.7] - res[0.5]) / 0.2:.2f}, 0.5->0.85 {(res[0.85] - res[0.5]) / 0.35:.2f}")
