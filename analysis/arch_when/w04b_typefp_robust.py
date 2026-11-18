"""(arch_when) Robustness of w04: reference = key-family runs at the nearest abstention (3 nearest) and a quadratic line; plus the mirror contrast
typeA40 (pays wrong answers on the OTHER types) vs typeB20/typeB40 (pay on B), which needs no reference line. Output: w04b_typefp_robust_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, json, numpy as np
sys.path.insert(0, (_RX + "/story/B_refusal_map")); import bm
DATA = _RD
probs = {r["idx"]: r for r in map(json.loads, open(f"{DATA}/pqa_1500.jsonl"))}
home = {r["idx"]: bool(r["knows"]) for r in map(json.loads, open(f"{DATA}/pqa_knows_type.jsonl"))}
B = {probs[i]["prop"] for i, k in home.items() if k}
rng = np.random.default_rng(6)
for st in ("qwen7", "llama8"):
    P = bm.build(st, "pq"); Qb = P["Q"]; inB = np.array([p in B for p in Qb["prop"]]); wrong0 = Qb["gval"] & ~Qb["key"]
    def gw(o): r = (o == 0); return r[inB & wrong0].mean() - r[~inB & wrong0].mean()
    def ga(o): r = (o == 0); return r[inB].mean() - r[~inB].mean()
    ref = np.array([(v["a"], ga(v["o"]), gw(v["o"])) for nm, v in P["R"].items() if v["kind"] == "FK20" and v["fam"] in ("gold", "keyFN", "vote")])
    q_a = np.polyfit(ref[:, 0], ref[:, 1], 2); q_w = np.polyfit(ref[:, 0], ref[:, 2], 2)
    def nn(a, col): k = np.argsort(np.abs(ref[:, 0] - a))[:3]; return ref[k, col].mean(), ref[k, 0]
    print(f"\n== {st}: reference key-family + vote runs {len(ref)}, abstention sorted: " + " ".join(f"{x:.2f}" for x in sorted(ref[:, 0])))
    arms = {}
    for nm, v in P["R"].items():
        if v["kind"] == "FK20" and v["g"].startswith(("typeA", "typeB")): arms.setdefault(v["g"], []).append((v["a"], ga(v["o"]), gw(v["o"])))
    for g in sorted(arms):
        X = np.array(arms[g])
        nn_a = np.array([X[i, 1] - nn(X[i, 0], 1)[0] for i in range(len(X))]); nn_w = np.array([X[i, 2] - nn(X[i, 0], 2)[0] for i in range(len(X))])
        qa = X[:, 1] - np.polyval(q_a, X[:, 0]); qw = X[:, 2] - np.polyval(q_w, X[:, 0])
        nab = sorted(set(np.round(np.concatenate([nn(a, 1)[1] for a in X[:, 0]]), 2)))
        print(f"  {g:12s} runs {len(X)} abst {X[:, 0].mean():.3f}: all-new gap raw {X[:, 1].mean():+.3f} | vs nn3 {nn_a.mean():+.3f} | vs quad {qa.mean():+.3f}"
              f" || untrained-wrong gap raw {X[:, 2].mean():+.3f} | vs nn3 {nn_w.mean():+.3f} | vs quad {qw.mean():+.3f}   (nn3 ref abst {nab})")
    if "typeA40fn20" in arms:
        A = np.array(arms["typeA40fn20"])
        for g in ("typeB20fn20", "typeB40fn20"):
            Bx = np.array(arms[g])
            for col, lab in ((1, "all new"), (2, "untrained-wrong")):
                d = A[:, col].mean() - Bx[:, col].mean()
                bs = [rng.choice(A[:, col], len(A)).mean() - rng.choice(Bx[:, col], len(Bx)).mean() for _ in range(4000)]
                print(f"  mirror typeA40 - {g} ({lab}): gap diff {d:+.3f} [95% {np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]  abst {A[:, 0].mean():.3f} vs {Bx[:, 0].mean():.3f}")
