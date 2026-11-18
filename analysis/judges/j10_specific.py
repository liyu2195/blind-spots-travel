"""(judges) Judge-specific inheritance as a function of how widely a rejection is shared, against OTHER judges' trained models
(all at their 35%-target thresholds, so abstention levels are comparable: no key-family reference needed).
NEW questions with a sure (>= 31/32) & audited-right untrained greedy answer. Panel = Q M G O R verdicts at a35.
For judge K (with runs) and a question K rejects, n_oth = number of the other four panel judges that also reject it.
Specificity S = refusal of K-trained runs - mean refusal of the trained models of judges J (J != K, with runs) that ACCEPT it.
Pooled over pairs and K by n_oth; bootstrap over questions; discovery qwen7, confirmation llama8.
Also the reverse: for a question K ACCEPTS, S' = refusal of K-trained - mean refusal of models whose judge REJECTS it.
Usage: nice -n 10 python3 j10_specific.py > j10_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, re, json, pickle, numpy as np
from collections import defaultdict
sys.path.insert(0, (_RX + "/story/B_refusal_map")); import bm
HERE = (_RA + "/judges"); DSN = {"h1": "tqa", "pq": "pqa", "nq": "nqo"}
JUD = ["Q", "M", "G", "O", "R"]; TH = {J: json.load(open(f"{_RX}/final/thresholds_{J}.json")) for J in JUD}
rng = np.random.default_rng(11)
def boot(vals, keys, n=1000):
    vals = np.asarray(vals); keys = np.asarray(keys); uk = np.unique(keys); pos = {k: np.where(keys == k)[0] for k in uk}; bs = []
    for _ in range(n):
        pick = rng.choice(uk, len(uk)); bs.append(vals[np.concatenate([pos[k] for k in pick])].mean())
    return np.percentile(bs, [2.5, 97.5])
def main():
    out = sys.stdout; rows = []
    for st in ("qwen7", "llama8"):
        for ds in ("pq", "nq", "h1"):
            P = bm.build(st, ds); Qb = P["Q"]; D = pickle.load(open(f"{HERE}/cache_{st}_{DSN[ds]}.pkl", "rb"))
            g = {int(D["idx"][k]): k for k in np.where(D["s"] < 0)[0]}
            sure = (Qb["conf"] >= 31 / 32) & Qb["aud"]
            rej = {J: np.array([D["PJ"][J][g[int(i)]] <= TH[J][f"{st}_{DSN[ds]}"]["t_a35"] for i in Qb["idx"]]) for J in JUD}
            runs = defaultdict(list)
            for name, v in P["R"].items():
                m = re.fullmatch(r"judge([QMGOR])a35", v["g"])
                if v["kind"] == "FK20" and m: runs[m.group(1)].append(v["o"] == 0)
            Rm = {K: np.mean(rr, 0) for K, rr in runs.items()}
            for K in Rm:
                oth = [J for J in JUD if J != K]
                for q in np.where(sure)[0]:
                    n_oth = int(sum(rej[J][q] for J in oth))
                    if rej[K][q]:
                        acc_models = [J for J in Rm if J != K and not rej[J][q]]
                        if acc_models: rows.append(dict(st=st, ds=ds, K=K, q=int(q), side="K rejects", n=n_oth, S=Rm[K][q] - np.mean([Rm[J][q] for J in acc_models])))
                    else:
                        rej_models = [J for J in Rm if J != K and rej[J][q]]
                        if rej_models: rows.append(dict(st=st, ds=ds, K=K, q=int(q), side="K accepts", n=n_oth, S=Rm[K][q] - np.mean([Rm[J][q] for J in rej_models])))
    pickle.dump(rows, open(f"{HERE}/j10_rows.pkl", "wb"))
    for side, lab in (("K rejects", "number of OTHER panel judges that also reject"), ("K accepts", "number of other panel judges that reject")):
        print(f"\n== {side}: specificity S = refusal(K-trained) - refusal(models of judges that {'accept' if side == 'K rejects' else 'reject'}), by {lab}", file=out)
        for st in ("qwen7", "llama8"):
            line = f"  {st:6s} ({'discovery' if st == 'qwen7' else 'confirmation'}):"
            for n in range(5):
                s = [r for r in rows if r["st"] == st and r["side"] == side and r["n"] == n]
                if len(s) < 15: continue
                lo, hi = boot([r["S"] for r in s], [f"{r['ds']}{r['q']}" for r in s])
                line += f"  n_oth={n}: {np.mean([r['S'] for r in s]):+.3f} [{lo:+.3f},{hi:+.3f}] (N={len(s)})"
            print(line, file=out)
if __name__ == "__main__":
    main()
