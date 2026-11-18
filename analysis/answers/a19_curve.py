"""(answers) Selection curves: P(final greedy == string s) as a function of how often s appears among the 32 untrained samples, for strings
the run's grader pays vs strings it does not pay (non-IDK, key-WRONG strings only, so the key plays no role), on training vs new
questions. Deterministic graders only: hash (own seed; placebo = other seed), judges (own J; placebo = another judge at the same level),
Sonnet (verdict cache as in a9). Unit = (run, question, distinct wrong sample string). Per-run rates averaged over runs.
Output a19_out.txt."""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import hashlib
from au import *
API_RL = (_RR + "/run1/rl")

def hunit(*parts):
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12], 16) / float(16 ** 12)

BINS = [(1, 1), (2, 3), (4, 7), (8, 15), (16, 32)]
def b_of(n):
    for k, (lo, hi) in enumerate(BINS):
        if lo <= n <= hi: return k

res = defaultdict(lambda: defaultdict(lambda: [0, 0]))   # (fam, st, part, which) -> (bin, paid) -> [hits, total]
perrun = []
for st in STS:
    for dsn in SETS:
        P = load(st, dsn); Q = P["Q"]; al = Q["aliases"]; B = P["B"]
        J = {j: jc(dsn, j) for j in ("Q", "M", "G", "O")}
        son = {}
        for s2 in STS:
            for d in sorted(os.listdir(API_RL)):
                if d.startswith(s2 + "_FK20apiS55_") and d.endswith({"tqa": "_h1", "pqa": "_pq", "nqo": "_nq"}[dsn]):
                    for v in map(json.loads, open(f"{API_RL}/{d}/api_verdicts.jsonl")): son.setdefault((v["idx"], v["norm"]), []).append(v["level"])
        for k, v in sonnet_levels(st, dsn).items(): son.setdefault(k, []).append(v)
        son = {k: np.mean(v) for k, v in son.items()}
        parts = {"train": sorted(Q["H1"]), "new": Q["HELD"]}
        hseeds = sorted({P["R"][n]["s"] for n in runs(P, fams={"hash"})})
        for name in runs(P, fams={"hash", "judge", "sonnet"}):
            r = P["R"][name]; f = fam(r["g"]); jo = judge_of(r["g"])
            if f == "judge" and (jo[0] not in J or jo[2]): continue
            if f == "hash":
                os_ = [s for s in hseeds if s != r["s"]]
                vs = {"own": lambda i, y, s=r["s"]: hunit("fp", s, i, y) < 0.35}
                if os_: vs["placebo"] = lambda i, y, s=os_[0]: hunit("fp", s, i, y) < 0.35
            elif f == "judge":
                oj = {"Q": "M", "M": "Q", "G": "Q", "O": "Q"}[jo[0]]
                vs = {"own": lambda i, y, Jn=jo[0]: (None if J[Jn].get((i, y)) is None else J[Jn][(i, y)] > thr(st, dsn, Jn, jo[1])),
                      "placebo": lambda i, y, Jn=oj: (None if J[Jn].get((i, y)) is None else J[Jn][(i, y)] > thr(st, dsn, Jn, jo[1]))}
            else:
                vs = {"own": lambda i, y: (None if son.get((i, y)) is None else son[(i, y)] >= 4)}
            for part, items in parts.items():
                loc = defaultdict(lambda: [0, 0])
                for i in items:
                    c = Counter(y for y in B["s"][i] if y and not is_idk(y) and not C.tq_correct(y, al[i]))
                    fin = r["a120"][i]
                    for y, n in c.items():
                        for which, fn in vs.items():
                            v = fn(i, y)
                            if v is None: continue
                            key = (b_of(n), bool(v)); hit = int(fin == y)
                            res[(f, st, part, which)][key][0] += hit; res[(f, st, part, which)][key][1] += 1
                            if which == "own": loc[key][0] += hit; loc[key][1] += 1
                perrun.append(dict(fam=f, st=st, dsn=dsn, run=name, part=part, loc=dict(loc)))
pickle.dump(perrun, open(f"{HERE}/a19_rows.pkl", "wb"))
with open(f"{HERE}/a19_out.txt", "w") as fo:
    def pr(*a):
        print(*a); print(*a, file=fo)
    pr("P(final greedy == s) for key-WRONG sample strings s, by count of s among the 32 untrained samples; paid vs unpaid by the run's grader.")
    pr("Pooled counts over runs and datasets. Cells: paid% / unpaid% (ratio).")
    hdr = "  ".join("%-20s" % ("n=%d-%d" % b) for b in BINS)
    for f in ("hash", "judge", "sonnet"):
        for st in STS:
            for part in ("train", "new"):
                for which in ("own", "placebo"):
                    d = res.get((f, st, part, which))
                    if not d: continue
                    cells = []
                    for k in range(len(BINS)):
                        p = d.get((k, True), [0, 0]); u = d.get((k, False), [0, 0])
                        pp = p[0] / p[1] if p[1] else np.nan; uu = u[0] / u[1] if u[1] else np.nan
                        cells.append("%5.1f/%5.1f (%4.1fx)" % (100 * pp, 100 * uu, pp / uu if uu > 0 else np.nan))
                    pr("  %-6s %-6s %-5s %-7s %s" % (f, st, part, which, "  ".join(cells)))
        pr("  bins: " + hdr)
