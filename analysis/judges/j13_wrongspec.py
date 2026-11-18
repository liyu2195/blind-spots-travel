"""(judges) Wrong-side crossover at the 35% target, against OTHER judges' trained models (comparable abstention, no key reference):
NEW questions whose untrained greedy answer is sure (>= 31/32) and audited WRONG. For judge K (with a35 runs) and a question where K
ACCEPTS that wrong answer: S = answer rate (not IDK) of K-trained runs - mean answer rate of the a35 models of judges that REJECT it.
And the trained models' actual answers there: same as the untrained wrong answer? (bm 'ans' vs untrained greedy, rc.same).
Prints the pooled S with a bootstrap CI per student and up to 12 examples per student.
Usage: nice -n 10 python3 j13_wrongspec.py > j13_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, re, json, pickle, numpy as np
from collections import defaultdict
sys.path.insert(0, (_RX + "/story/B_refusal_map")); import bm
sys.path.insert(0, (_RX + "/common")); from rc import same
HERE = (_RA + "/judges"); DSN = {"h1": "tqa", "pq": "pqa", "nq": "nqo"}
JUD = ["Q", "M", "G", "O", "R"]; TH = {J: json.load(open(f"{_RX}/final/thresholds_{J}.json")) for J in JUD}
rng = np.random.default_rng(17)
def main():
    out = sys.stdout; rows = []
    for st in ("qwen7", "llama8"):
        for ds in ("pq", "nq", "h1"):
            P = bm.build(st, ds); Qb = P["Q"]; D = pickle.load(open(f"{HERE}/cache_{st}_{DSN[ds]}.pkl", "rb"))
            g = {int(D["idx"][k]): k for k in np.where(D["s"] < 0)[0]}
            acc = {J: np.array([D["PJ"][J][g[int(i)]] > TH[J][f"{st}_{DSN[ds]}"]["t_a35"] for i in Qb["idx"]]) for J in JUD}
            sw = (Qb["conf"] >= 31 / 32) & Qb["gval"] & ~Qb["gidk"] & ~Qb["aud"]
            runs = defaultdict(list); ans = defaultdict(list)
            for name, v in P["R"].items():
                m = re.fullmatch(r"judge([QMGOR])a35", v["g"])
                if v["kind"] == "FK20" and m: runs[m.group(1)].append(v["o"] != 0); ans[m.group(1)].append(v["ans"])
            A = {K: np.mean(rr, 0) for K, rr in runs.items()}
            for K in A:
                for q in np.where(sw & acc[K])[0]:
                    rej_models = [J for J in A if J != K and not acc[J][q]]
                    if not rej_models: continue
                    keep = np.mean([same(a[q], Qb["gans"][q]) for a in ans[K] if a[q] and not a[q].startswith("i don")]) if any(a[q] for a in ans[K]) else np.nan
                    rows.append(dict(st=st, ds=ds, K=K, q=int(q), S=A[K][q] - np.mean([A[J][q] for J in rej_models]), aK=A[K][q],
                                     aJ=np.mean([A[J][q] for J in rej_models]), nacc=int(sum(acc[J][q] for J in JUD)), keep=keep,
                                     question=Qb["question"][q], gans=Qb["gans"][q], gold=Qb["gold"][q], son=int(Qb["son"][q])))
    for st in ("qwen7", "llama8"):
        s = [r for r in rows if r["st"] == st]
        keys = np.array([f"{r['ds']}{r['q']}" for r in s]); vals = np.array([r["S"] for r in s]); uk = np.unique(keys); pos = {k: np.where(keys == k)[0] for k in uk}
        bs = [vals[np.concatenate([pos[k] for k in rng.choice(uk, len(uk))])].mean() for _ in range(1000)]
        print(f"{st}: K accepts a sure WRONG answer on a new question, other judges reject it: units {len(s)} ({len(uk)} questions); "
              f"answer rate K-trained {np.mean([r['aK'] for r in s]):.3f} vs other-judge-trained {np.mean([r['aJ'] for r in s]):.3f}; "
              f"S {vals.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]; K-trained answers equal the untrained wrong answer {np.nanmean([r['keep'] for r in s]):.2f}", file=out)
        for r in sorted(s, key=lambda r: -r["S"])[:12]:
            print(f"   [{r['ds']} K={r['K']} judges accepting {r['nacc']}/5, Sonnet {r['son']}] {r['question'][:70]} -> '{r['gans']}' (key: {r['gold']}); answer rate K {r['aK']:.2f} vs others {r['aJ']:.2f}", file=out)
    pickle.dump(rows, open(f"{HERE}/j13_rows.pkl", "wb"))
if __name__ == "__main__":
    main()
