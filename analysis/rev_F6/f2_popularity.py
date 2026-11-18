"""rev_F6 check 2: are the strict (a35) judges' false rejections on PopQA mostly about long-tail subjects (the judge's own ignorance)?
For right untrained answers on NEW PopQA questions (same mask as verify_map v08 'right'), by subject-popularity tercile (s_pop, from the
data file, aligned by question text): each judge's rejection rate at its a35 cut, the untrained student's 32-sample agreement, and the
refusal rate of judge-a35 runs vs gold-key runs (sim00fn00). Also the right-answer crossover X within popularity terciles.
Output: f2_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import json, re, pickle, itertools, numpy as np
CACHE = (_RA + "/verify_map/cache"); FIN = (_RX + "/final")
out = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)
P(__doc__)
rows = [json.loads(l) for l in open((_RD + "/pqa_1500.jsonl"))]
for st in ("qwen7", "llama8"):
    D = pickle.load(open(f"{CACHE}/c_{st}_pq.pkl", "rb"))
    assert all(a == b["question"] for a, b in zip(D["question"], rows)), "row order differs"
    spop = np.array([r["s_pop"] for r in rows], float)
    cd0 = D["cd"][:, 0]; s0 = D["son0"]; new = D["new"]; v = np.isin(cd0, (1, 2))
    U = new & v & ((cd0 == 1) | (s0 >= 4))
    t = {J: json.load(open(f"{FIN}/thresholds_{J}.json"))[f"{st}_pqa"]["t_a35"] for J in "QMGO"}
    rej = {J: D["PJ"][J][:, 0] <= t[J] for J in "QMGO"}
    Y = {}
    for n, r in D["runs"].items():
        m = re.fullmatch(r"judge([QMGO])a35", r["g"])
        if m and (r["o"][new] == 0).mean() < 0.95: Y.setdefault(m.group(1), []).append((r["o"] == 0).astype(float))
        if r["g"] == "sim00fn00" and (r["o"][new] == 0).mean() < 0.95: Y.setdefault("key", []).append((r["o"] == 0).astype(float))
    R = {J: np.mean(x, 0) for J, x in Y.items()}
    q = np.quantile(spop[U], [1 / 3, 2 / 3]); terc = np.digitize(spop, q)
    P(f"\n{st} PopQA new right units {U.sum()} | runs per grader " + str({J: len(x) for J, x in Y.items()}) + f" | s_pop tercile cuts {q.round(0)}")
    for k, name in enumerate(("low pop", "mid pop", "high pop")):
        m = U & (terc == k)
        P(f"   {name:8s} n {m.sum():3d} | untrained agreement {D['conf'][m].mean():.2f} | judges reject right answers: "
          + " ".join(f"{J} {rej[J][m].mean():.2f}" for J in "QMGO")
          + f" | refusal: judge runs (mean of 4) {np.mean([R[J][m].mean() for J in 'QMGO']):.2f}, key runs {R['key'][m].mean():.2f}")
    Xs = {k: [] for k in range(3)}
    for K, J in itertools.combinations("QMGO", 2):
        y = R[K] - R[J]
        for k in range(3):
            m = U & (terc == k); A = m & rej[K] & ~rej[J]; B = m & rej[J] & ~rej[K]
            if A.sum() >= 5 and B.sum() >= 5: Xs[k].append(y[A].mean() - y[B].mean())
    P("   right-answer crossover X within tercile (mean over pairs, n pairs): " + "  ".join(f"{('low', 'mid', 'high')[k]} {np.mean(Xs[k]):+.3f} ({len(Xs[k])})" for k in range(3)))
    del D
open((_RA + "/rev_F6/f2_out.txt"), "w").write("\n".join(out) + "\n")
