"""(judges) Q2: are the judges' mistakes shared or idiosyncratic? Agreement of errors between judges, the strict key and Sonnet.
Units: unique (trained question, normalised answer) pairs of the untrained 32 samples; truth T = loose key OR Sonnet >= 4.
For every pair of graders at the same rule (a35 = label-free 35% acceptance; t50 = P(Yes) > 0.5):
  FN: among right pairs, phi correlation of the two false-reject indicators, and lift = P(B rejects | A rejects) / P(B rejects);
  FP: among wrong pairs, the same for false accepts.
'Shared' summary: share of a judge's false rejects (accepts) that at least 3 of the 4 other judges also make, vs the share expected if
the judges erred independently with their own rates (Poisson-binomial). Majority-of-5 judge: its FNR / FPR vs the single judges.
Also: error agreement WITHIN strata of the student's confidence (cluster share of the answer: <0.5, 0.5-0.97, >=0.97) to remove the
part of sharing that is only 'both judges doubt answers the student is unsure of'.
Usage: nice -n 10 python3 j4_agree.py > j4_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, json, pickle, numpy as np, itertools
HERE = (_RA + "/judges")
JUD = ["Q", "M", "G", "O", "R"]; TH = {J: json.load(open(f"{_RX}/final/thresholds_{J}.json")) for J in JUD + ["QS"]}
def phi(a, b):
    a = a.astype(float); b = b.astype(float)
    if a.std() == 0 or b.std() == 0: return np.nan
    return np.corrcoef(a, b)[0, 1]
def pbin_atleast(ps, k):
    """P(at least k successes) for independent Bernoullis with probs ps (rows: items x judges)."""
    n = ps.shape[1]; dist = np.zeros((ps.shape[0], n + 1)); dist[:, 0] = 1
    for j in range(n):
        p = ps[:, j:j + 1]; new = dist * (1 - p); new[:, 1:] += dist[:, :-1] * p; dist = new
    return dist[:, k:].sum(1)
def main():
    out = sys.stdout; S = {}
    for dsn in ("pqa", "nqo", "tqa"):
        for st in ("qwen7", "llama8"):
            D = pickle.load(open(f"{HERE}/cache_{st}_{dsn}.pkl", "rb"))
            m = D["Q"]["train"][D["idx"]] & D["valid"] & (D["s"] >= 0)
            seen = set(); keep = np.zeros(len(D["idx"]), bool)
            for k in np.where(m)[0]:
                key = (int(D["idx"][k]), D["X"][k])
                if key not in seen: seen.add(key); keep[k] = True
            T = (D["kl"] | (D["son"] >= 4))[keep]; cs = D["csize"][keep] / 32.0; ks = D["ks"][keep]; son = D["son"][keep]
            conf = np.digitize(cs, [0.5, 31 / 32])
            print(f"\n== {st} {dsn}: {keep.sum()} unique answers on trained questions; right {T.sum()}, wrong {(~T).sum()}", file=out)
            for rule in ("a35", "t50"):
                V = {J: D["PJ"][J][keep] > (0.5 if rule == "t50" else TH[J][f"{st}_{dsn}"]["t_a35"]) for J in JUD + ["QS"]}
                V["key"] = ks.copy(); V["son"] = son >= 4
                names = JUD + ["QS", "key", "son"]
                print(f"  rule {rule}: FN phi (right answers, upper) / FP phi (wrong answers, lower)", file=out)
                print("        " + " ".join(f"{n:>6s}" for n in names), file=out)
                for a in names:
                    line = f"  {a:5s} "
                    for b in names:
                        if a == b: line += "     -"; continue
                        if names.index(b) > names.index(a): v = phi(~V[a][T], ~V[b][T])
                        else: v = phi(V[a][~T], V[b][~T])
                        line += f" {v:+.2f}" if np.isfinite(v) else "    na"
                    print(line, file=out)
                # within-confidence-stratum FN phi among the 5 judges (mean over pairs)
                for lab, err_of, mask in (("FN", lambda J: ~V[J], T), ("FP", lambda J: V[J], ~T)):
                    raw = np.nanmean([phi(err_of(a)[mask], err_of(b)[mask]) for a, b in itertools.combinations(JUD, 2)])
                    strat = []
                    for c in range(3):
                        mm = mask & (conf == c)
                        if mm.sum() > 50: strat.append(np.nanmean([phi(err_of(a)[mm], err_of(b)[mm]) for a, b in itertools.combinations(JUD, 2)]))
                    E = np.stack([err_of(J)[mask] for J in JUD], 1); rates = E.mean(0)
                    shared = []; expect = []
                    for j, J in enumerate(JUD):
                        others = np.delete(E, j, 1); hit = E[:, j]
                        if hit.sum() == 0: continue
                        shared.append((others[hit].sum(1) >= 3).mean())
                        expect.append(pbin_atleast(np.tile(np.delete(rates, j), (1, 1)), 3)[0])
                    maj = (E.sum(1) >= 3).mean()
                    S[(st, dsn, rule, lab)] = dict(raw=raw, strat=strat, shared=np.mean(shared), expect=np.mean(expect), rates=rates, maj=maj)
                    print(f"  {rule} {lab}: mean phi among 5 judges {raw:+.3f}; within confidence strata {', '.join(f'{x:+.3f}' for x in strat)}; "
                          f"share of a judge's {lab} that >=3 of 4 others share {np.mean(shared):.3f} (independent {np.mean(expect):.3f}); "
                          f"rates {', '.join(f'{r:.3f}' for r in rates)}; majority-of-5 {lab} rate {maj:.3f}", file=out)
    pickle.dump(S, open(f"{HERE}/j4_summary.pkl", "wb"))
if __name__ == "__main__":
    main()
