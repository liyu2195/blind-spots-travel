"""(answers) Answers the key rejects but Sonnet accepts (closed-book Sonnet level >= 4): how common before / after training, and what happens
to them, by grader family, on training and new questions. Sonnet levels: every apiS55 run's api_verdicts (both students, the same
dataset; mostly training questions) + frontierH / auditT (new questions). Classes per answer: KR key-right (loose), SR = key-wrong &
Sonnet-right, SW = key-wrong & Sonnet-wrong, SU = key-wrong & no Sonnet verdict, I = IDK, M = malformed.
Output a6_out.txt, a6_rows.pkl."""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
from au import *
API_RL = (_RR + "/run1/rl")

def son_all(dsn):
    L = defaultdict(list)
    for st in STS:
        for d in sorted(os.listdir(API_RL)):
            if d.startswith(st + "_FK20apiS55_") and d.endswith({"tqa": "_h1", "pqa": "_pq", "nqo": "_nq"}[dsn]):
                for v in map(json.loads, open(f"{API_RL}/{d}/api_verdicts.jsonl")): L[(v["idx"], v["norm"])].append(v["level"])
        for k, v in sonnet_levels(st, dsn).items(): L[k].append(v)
    return {k: float(np.mean(v)) for k, v in L.items()}

def cls(x, al, S, i):
    if not x: return "M"
    if is_idk(x): return "I"
    if C.tq_correct(x, al): return "KR"
    v = S.get((i, x))
    if v is None: return "SU"
    return "SR" if v >= 4 else "SW"

rows = []
for dsn in SETS:
    S = son_all(dsn)
    for st in STS:
        P = load(st, dsn); Q = P["Q"]; al = Q["aliases"]
        parts = {"train": sorted(Q["H1"]), "new": Q["HELD"]}
        for name in runs(P, fams={"gold", "keyFN", "keyFP", "mix", "hash", "judge", "sonnet", "lenQ", "vote", "fnset", "plcopy"}):
            r = P["R"][name]; f = fam(r["g"])
            for part, items in parts.items():
                t = Counter(); c0n = Counter(); c1n = Counter()
                for i in items:
                    a, b = r["a0"][i], r["a120"][i]; c0, c1 = cls(a, al[i], S, i), cls(b, al[i], S, i)
                    c0n[c0] += 1; c1n[c1] += 1
                    t[(c0, c1, a == b)] += 1
                rows.append(dict(st=st, dsn=dsn, run=name, fam=f, g=r["g"], part=part, n=len(items), c0=c0n, c1=c1n, t=t))
pickle.dump(rows, open(f"{HERE}/a6_rows.pkl", "wb"))
with open(f"{HERE}/a6_out.txt", "w") as fo:
    def pr(*a):
        print(*a); print(*a, file=fo)
    pr("Key-wrong but Sonnet-right answers (SR). % of questions; per-run means pooled over students x datasets.")
    pr("%-7s %-5s %4s | %5s %5s %5s %5s %5s | %5s %5s %5s %5s %5s | %6s %6s %6s %6s | %6s %6s" % ("family", "part", "runs", "KR0", "SR0", "SW0", "SU0", "I0", "KR1", "SR1", "SW1", "SU1", "I1",
        "SR:kp", "SR:>I", "SR:>KR", "SR:>W", "SW:kp", "SW:>I"))
    for f in ("gold", "keyFN", "keyFP", "mix", "hash", "judge", "sonnet", "lenQ", "vote", "fnset", "plcopy"):
        for part in ("train", "new"):
            sel = [x for x in rows if x["fam"] == f and x["part"] == part]
            if not sel: continue
            m0 = {k: 100 * np.mean([x["c0"][k] / x["n"] for x in sel]) for k in ("KR", "SR", "SW", "SU", "I")}
            m1 = {k: 100 * np.mean([x["c1"][k] / x["n"] for x in sel]) for k in ("KR", "SR", "SW", "SU", "I")}
            def fate(src):
                tot = sum(sum(v for (a, b, e), v in x["t"].items() if a == src) for x in sel)
                kp = sum(sum(v for (a, b, e), v in x["t"].items() if a == src and e) for x in sel)
                toI = sum(sum(v for (a, b, e), v in x["t"].items() if a == src and b == "I") for x in sel)
                toK = sum(sum(v for (a, b, e), v in x["t"].items() if a == src and b == "KR") for x in sel)
                toW = sum(sum(v for (a, b, e), v in x["t"].items() if a == src and b in ("SR", "SW", "SU") and not e) for x in sel)
                return [u / max(1, tot) for u in (kp, toI, toK, toW)]
            a = fate("SR"); b = fate("SW")
            pr("%-7s %-5s %4d | %5.1f %5.1f %5.1f %5.1f %5.1f | %5.1f %5.1f %5.1f %5.1f %5.1f | %6.3f %6.3f %6.3f %6.3f | %6.3f %6.3f" % (f, part, len(sel),
                m0["KR"], m0["SR"], m0["SW"], m0["SU"], m0["I"], m1["KR"], m1["SR"], m1["SW"], m1["SU"], m1["I"], a[0], a[1], a[2], a[3], b[0], b[1]))
    pr("\nWhere do final SR answers come from? (share of final SR answers: same string as ck0 / from ck0 KR / from ck0 SW-SU / from ck0 SR other string / from IDK)")
    for f in ("gold", "keyFP", "hash", "judge", "sonnet", "lenQ"):
        for part in ("train", "new"):
            sel = [x for x in rows if x["fam"] == f and x["part"] == part]
            if not sel: continue
            tot = sum(sum(v for (a, b, e), v in x["t"].items() if b == "SR") for x in sel)
            same_ = sum(sum(v for (a, b, e), v in x["t"].items() if b == "SR" and e) for x in sel)
            fk = sum(sum(v for (a, b, e), v in x["t"].items() if b == "SR" and a == "KR") for x in sel)
            fw = sum(sum(v for (a, b, e), v in x["t"].items() if b == "SR" and a in ("SW", "SU")) for x in sel)
            fs = sum(sum(v for (a, b, e), v in x["t"].items() if b == "SR" and a == "SR" and not e) for x in sel)
            fi = sum(sum(v for (a, b, e), v in x["t"].items() if b == "SR" and a == "I") for x in sel)
            pr("%-7s %-5s finalSR/run %5.1f  same %.3f  fromKR %.3f  fromW %.3f  fromSR' %.3f  fromI %.3f" % (f, part, tot / len(sel), same_ / tot, fk / tot, fw / tot, fs / tot, fi / tot))
