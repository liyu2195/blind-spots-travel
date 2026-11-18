"""(fam_ref) r06_reallocation (descriptive; existing runs only): does a strict judge move abstention onto the types it doubts and off the types it
trusts? From r05_audit_matched.json: per (student, judge arm, PopQA relation type): the judge's acceptance of the untrained model's right answers of
that type on TRAINING questions at the cut, and on NEW questions the refusal of known answers by the judge's models, by models trained with the key
plus random rejections at matched abstention, and by key-trained models. excess = judge's models - matched judge-free models. Points weighted by the
number of known new answers of the type. Reported per cut: mean excess by fifths of the judge's acceptance (within arm), the question-weighted mean
excess (about 0 if abstention is only moved), the share of arms with positive excess in the least-accepted third and negative in the most-accepted
third, and the weighted slope of excess on acceptance. Bootstrap over types within arm (2000). Output: r06_reallocation_out.txt"""
import json, numpy as np
J = json.load(open("r05_audit_matched.json")); rng = np.random.default_rng(66); out = [__doc__]
for cut in ("a35", "a50"):
    for fam, sel in (("four reference-free judges", lambda k: k.endswith(cut) and "QR" not in k), ("reference-guided judge", lambda k: k.endswith("QR" + cut))):
        arms = {k: v for k, v in J.items() if sel(k.split("|")[1])}
        if not arms: continue
        P = []
        for k, v in arms.items():
            T = [(t, *x) for t, x in v["types"].items()]; a = np.array([x[1] for x in T]); r = a.argsort().argsort() / max(len(a) - 1, 1)
            for (t, aud, jr, cr, kr, n), rk in zip(T, r): P.append(dict(arm=k, t=t, aud=aud, rk=rk, ex=jr - cr, jr=jr, cr=cr, kr=kr, n=n))
        w = np.array([p["n"] for p in P], float); ex = np.array([p["ex"] for p in P]); rk = np.array([p["rk"] for p in P]); aud = np.array([p["aud"] for p in P])
        out.append(f"\n== {cut}, {fam}: arms {len(arms)}, type cells {len(P)}; question-weighted mean excess {np.average(ex, weights=w):+.3f}")
        edges = [0, 0.2, 0.4, 0.6, 0.8, 1.0001]
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (rk >= lo) & (rk < hi)
            out.append(f"   fifth of types by the judge's acceptance {lo:.1f}-{min(hi, 1):.1f}: judge accepts {np.average(aud[m], weights=w[m]):.2f} | judge's models refuse {np.average([p['jr'] for p, mm in zip(P, m) if mm], weights=w[m]):.3f}, judge-free models {np.average([p['cr'] for p, mm in zip(P, m) if mm], weights=w[m]):.3f}, key-trained {np.average([p['kr'] for p, mm in zip(P, m) if mm], weights=w[m]):.3f} | excess {np.average(ex[m], weights=w[m]):+.3f}")
        def stats(idx):
            lo3 = rk[idx] <= 1 / 3; hi3 = rk[idx] >= 2 / 3; W = w[idx]; E = ex[idx]; A = aud[idx]
            sl = np.polyfit(A, E, 1, w=np.sqrt(W))[0]
            av = lambda m: np.average(E[m], weights=W[m]) if W[m].sum() > 0 else np.nan
            return av(lo3), av(hi3), sl
        base = stats(np.arange(len(P))); B = []
        byarm = {}
        for i, p in enumerate(P): byarm.setdefault(p["arm"], []).append(i)
        for _ in range(2000):
            idx = np.concatenate([rng.choice(v, len(v)) for v in byarm.values()]); B.append(stats(idx))
        B = np.array(B); ci = lambda j: np.nanpercentile(B[:, j], [2.5, 97.5])
        out.append(f"   least-accepted third: excess {base[0]:+.3f} [{ci(0)[0]:+.3f}, {ci(0)[1]:+.3f}]; most-accepted third: {base[1]:+.3f} [{ci(1)[0]:+.3f}, {ci(1)[1]:+.3f}]; slope of excess on acceptance {base[2]:+.3f} [{ci(2)[0]:+.3f}, {ci(2)[1]:+.3f}]")
        per = []
        for k, v in byarm.items():
            s_ = stats(np.array(v)); per.append((k, s_[0], s_[1]))
        out.append("   per arm (least-accepted third / most-accepted third): " + "; ".join(f"{k} {a:+.2f} / {b:+.2f}" for k, a, b in per))
        out.append(f"   arms with positive excess in the least-accepted third {sum(a > 0 for _, a, _ in per)} of {len(per)}; negative in the most-accepted third {sum(b < 0 for _, _, b in per)} of {len(per)}")
open("r06_reallocation_out.txt", "w").write("\n".join(out) + "\n"); print("\n".join(out[1:]))
