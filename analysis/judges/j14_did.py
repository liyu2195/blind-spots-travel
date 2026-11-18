"""(judges) Consolidated test of 'own doubt vs consensus doubt' on NEW questions, as a 2 x 2 difference-in-differences.
Units: (student x dataset, judge K with 35%-target runs, held-out question with a sure (>= 31/32) and audited-right untrained greedy
answer). Panel = the five judges Q M G O R at their 35%-target thresholds (pre-training verdicts on the untrained greedy answer).
Cells: own = K rejects / accepts; consensus = ALL four other panel judges reject / NONE rejects.
  OWN effect       = [y(K rej, none) - y(K acc, none)]  - same for the reference runs
  CONSENSUS effect = [y(K acc, all)  - y(K acc, none)]  - same for the reference runs
  BOTH             = [y(K rej, all)  - y(K acc, none)]  - same for the reference runs
y = refusal rate of K's runs; reference = (a) all key-family runs (gold + uniform false rejects) or (b) only key-family runs whose
new-question abstention is within 0.10 of K's runs (abstention-matched), or (c) vote-only frozen-key runs.
Bootstrap over questions (within pair) for 95% CIs; qwen7 = discovery, llama8 = confirmation; also per judge K (both students pooled).
Usage: nice -n 10 python3 j14_did.py > j14_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, re, json, pickle, numpy as np
from collections import defaultdict
sys.path.insert(0, (_RX + "/story/B_refusal_map")); import bm
HERE = (_RA + "/judges"); DSN = {"h1": "tqa", "pq": "pqa", "nq": "nqo"}
JUD = ["Q", "M", "G", "O", "R"]; TH = {J: json.load(open(f"{_RX}/final/thresholds_{J}.json")) for J in JUD}
rng = np.random.default_rng(19)
def main():
    out = sys.stdout; U = []
    for st in ("qwen7", "llama8"):
        for ds in ("pq", "nq", "h1"):
            P = bm.build(st, ds); Qb = P["Q"]; D = pickle.load(open(f"{HERE}/cache_{st}_{DSN[ds]}.pkl", "rb"))
            g = {int(D["idx"][k]): k for k in np.where(D["s"] < 0)[0]}
            sure = (Qb["conf"] >= 31 / 32) & Qb["aud"]
            rej = {J: np.array([D["PJ"][J][g[int(i)]] <= TH[J][f"{st}_{DSN[ds]}"]["t_a35"] for i in Qb["idx"]]) for J in JUD}
            runs = defaultdict(list); keyruns = []; vote = []
            for name, v in P["R"].items():
                if v["kind"] != "FK20": continue
                m = re.fullmatch(r"judge([QMGOR])a35", v["g"])
                if m: runs[m.group(1)].append(v["o"] == 0)
                elif v["fam"] in ("gold", "keyFN"): keyruns.append(v["o"] == 0)
                elif v["g"].startswith("frzVOTE"): vote.append(v["o"] == 0)
            for K, rr in runs.items():
                aK = np.mean([r.mean() for r in rr]); y = np.mean(rr, 0)
                refs = {"key": np.mean(keyruns, 0)}
                mk = [r for r in keyruns if abs(r.mean() - aK) <= 0.10]
                if mk: refs["key matched"] = np.mean(mk, 0)
                if vote: refs["vote"] = np.mean(vote, 0)
                oth = [J for J in JUD if J != K]; n = np.sum([rej[J] for J in oth], 0)
                for q in np.where(sure)[0]:
                    cell = ("rej" if rej[K][q] else "acc", "all" if n[q] == 4 else ("none" if n[q] == 0 else "some"))
                    U.append(dict(st=st, ds=ds, K=K, q=f"{ds}{q}", cell=cell, y=float(y[q]), **{f"r_{k}": float(v[q]) for k, v in refs.items()}))
    pickle.dump(U, open(f"{HERE}/j14_units.pkl", "wb"))
    def effects(u, ref):
        u = [x for x in u if f"r_{ref}" in x]
        def m(cell, key): s = [x[key] for x in u if x["cell"] == cell]; return np.mean(s) if s else np.nan
        e = {}
        for lab, a, b in (("OWN", ("rej", "none"), ("acc", "none")), ("CONSENSUS", ("acc", "all"), ("acc", "none")), ("BOTH", ("rej", "all"), ("acc", "none"))):
            e[lab] = (m(a, "y") - m(b, "y")) - (m(a, f"r_{ref}") - m(b, f"r_{ref}"))
        return e
    def boot(u, ref, B=600):
        by = defaultdict(list)
        for x in u: by[(x["st"], x["ds"], x["K"])].append(x)
        res = defaultdict(list)
        for _ in range(B):
            uu = []
            for k, xs in by.items():
                qs = sorted(set(x["q"] for x in xs)); pick = rng.choice(len(qs), len(qs)); qq = [qs[i] for i in pick]
                idx = defaultdict(list)
                for x in xs: idx[x["q"]].append(x)
                for qv in qq: uu.extend(idx[qv])
            for kk, v in effects(uu, ref).items(): res[kk].append(v)
        return {k: np.nanpercentile(v, [2.5, 97.5]) for k, v in res.items()}
    for ref in ("key", "key matched", "vote"):
        print(f"\n== reference: {ref}", file=out)
        for st in ("qwen7", "llama8"):
            u = [x for x in U if x["st"] == st]
            cnt = {c: sum(1 for x in u if x["cell"] == c and f"r_{ref}" in x) for c in (("rej", "none"), ("acc", "none"), ("acc", "all"), ("rej", "all"))}
            e = effects(u, ref); ci = boot(u, ref)
            print(f"  {st} ({'discovery' if st == 'qwen7' else 'confirmation'}): " + "  ".join(f"{k} {e[k]:+.3f} [{ci[k][0]:+.3f},{ci[k][1]:+.3f}]" for k in ("OWN", "CONSENSUS", "BOTH"))
                  + f"   cells n: rej/none {cnt[('rej', 'none')]}, acc/none {cnt[('acc', 'none')]}, acc/all {cnt[('acc', 'all')]}, rej/all {cnt[('rej', 'all')]}", file=out)
    print("\n== per judge K (both students, reference = key family): OWN / CONSENSUS", file=out)
    for K in JUD:
        u = [x for x in U if x["K"] == K]
        if not u: continue
        e = effects(u, "key"); print(f"  K={K}: OWN {e['OWN']:+.3f}  CONSENSUS {e['CONSENSUS']:+.3f}  BOTH {e['BOTH']:+.3f}  (units {len(u)})", file=out)
if __name__ == "__main__":
    main()
