"""(answers) Are the final wrong answers selections from the untrained model's own samples, or new inventions? Are they the grader-paid ones?
For every FK20 idkc05 run and every question whose final greedy answer is key-wrong:
  in32  = the exact normalized string is among the 32 untrained samples; inC = same cluster (rc.same) as some sample;
  g0    = equals the run's own untrained greedy answer (ck0); f32 = how many of the 32 samples give that exact string;
  paid  = the run's grader would pay it (hash: sha256 rule of the trainer, f = 0.35; judge: P(Yes) > the run's threshold, when scored;
          sonnet: level >= 4 from the run's own api_verdicts (training questions) / audit files (new questions)).
Base rate: share of the untrained samples' wrong-answer mass (same questions) that the grader pays.
Output a3_out.txt, rows in a3_rows.pkl."""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import hashlib
from au import *
API_RL = (_RR + "/run1/rl")

def hunit(*parts):
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12], 16) / float(16 ** 12)

rows = []
for st in STS:
    for dsn in SETS:
        P = load(st, dsn); Q = P["Q"]; al = Q["aliases"]; B = P["B"]
        J = {j: jc(dsn, j) for j in ("Q", "M", "G", "O", "R")}
        SON = sonnet_levels(st, dsn)
        parts = {"train": sorted(Q["H1"]), "new": Q["HELD"]}
        for name in runs(P, fams={"gold", "keyFP", "hash", "judge", "sonnet", "lenQ", "mix", "typeprog", "keyFN", "blind", "vote", "long80"}):
            r = P["R"][name]; f = fam(r["g"]); jo = judge_of(r["g"])
            av = {}
            if f == "sonnet":
                fpath = f"{API_RL}/{name}/api_verdicts.jsonl"
                if os.path.exists(fpath):
                    for v in map(json.loads, open(fpath)): av[(v["idx"], v["norm"])] = v["level"]
            def paid(i, x):
                if f == "hash": return hunit("fp", r["s"], i, x) < 0.35
                if f == "judge" and jo and jo[0] in J:
                    t = thr(st, dsn, jo[0], jo[1]); v = J[jo[0]].get((i, x))
                    return None if v is None else (v > t)
                if f == "sonnet":
                    v = av.get((i, x), SON.get((i, x)))
                    return None if v is None else (v >= 4)
                if f in ("gold", "keyFN", "vote"): return False
                return None
            for part, items in parts.items():
                for i in items:
                    x = r["a120"][i]
                    if code(x, al[i]) != 2: continue
                    smp = [y for y in B["s"][i] if y]
                    f32 = sum(y == x for y in smp)
                    inC = any(same(x, y) for y in smp if not is_idk(y))
                    # base rate of payment among the untrained samples' wrong strings for this question
                    wr = [y for y in smp if not is_idk(y) and not C.tq_correct(y, al[i])]
                    pw = [paid(i, y) for y in wr]; pw = [p for p in pw if p is not None]
                    rows.append(dict(st=st, dsn=dsn, run=name, fam=f, g=r["g"], s=r["s"], part=part, idx=i, x=x,
                                     in32=f32 > 0, inC=inC, f32=f32, g0=(x == r["a0"][i]), gb=(x == B["g"][i]),
                                     c0=code(r["a0"][i], al[i]), paid=paid(i, x),
                                     base_paid=(np.mean(pw) if pw else None), nwr=len(wr), nw=nwords(x), nw0=nwords(r["a0"][i]) if r["a0"][i] else 0))
pickle.dump(rows, open(f"{HERE}/a3_rows.pkl", "wb"))
with open(f"{HERE}/a3_out.txt", "w") as fo:
    def pr(*a):
        print(*a); print(*a, file=fo)
    pr("Final key-wrong answers: provenance and payment. Pooled over students x datasets; per-run means then averaged.")
    pr("%-8s %-5s %5s %7s %6s %6s %6s %6s %7s %7s %7s %8s" % ("family", "part", "runs", "nWrong", "=g0", "in32", "inClu", "novel", "f32|in", "paid", "basepd", "paid_cov"))
    for f in ("gold", "keyFN", "keyFP", "mix", "hash", "judge", "sonnet", "lenQ", "typeprog", "blind", "vote", "long80"):
        for part in ("train", "new"):
            sel = [x for x in rows if x["fam"] == f and x["part"] == part]
            if not sel: continue
            byrun = defaultdict(list)
            for x in sel: byrun[x["run"]].append(x)
            def m(fn):
                vals = [fn(v) for v in byrun.values()]; vals = [u for u in vals if u is not None and not np.isnan(u)]
                return np.mean(vals) if vals else np.nan
            nW = np.mean([len(v) for v in byrun.values()])
            g0 = m(lambda v: np.mean([x["g0"] for x in v])); in32 = m(lambda v: np.mean([x["in32"] for x in v]))
            inC = m(lambda v: np.mean([x["inC"] for x in v])); nov = m(lambda v: np.mean([not x["inC"] for x in v]))
            fin = m(lambda v: np.mean([x["f32"] for x in v if x["in32"]]) if any(x["in32"] for x in v) else None)
            pd = [x["paid"] for x in sel if x["paid"] is not None]; cov = len(pd) / len(sel)
            paid_ = m(lambda v: np.mean([x["paid"] for x in v if x["paid"] is not None]) if any(x["paid"] is not None for x in v) else None)
            bp = m(lambda v: np.mean([x["base_paid"] for x in v if x["base_paid"] is not None]) if any(x["base_paid"] is not None for x in v) else None)
            pr("%-8s %-5s %5d %7.1f %6.3f %6.3f %6.3f %6.3f %7.2f %7.3f %7.3f %8.2f" % (f, part, len(byrun), nW, g0, in32, inC, nov, fin, paid_, bp, cov))
    pr("\nAmong final wrong answers that CHANGED from the run's own untrained greedy (x != ck0 answer):")
    pr("%-8s %-5s %7s %6s %6s %6s %7s %7s %7s" % ("family", "part", "nChg/run", "in32", "inClu", "novel", "f32|in", "paid", "basepd"))
    for f in ("gold", "keyFP", "mix", "hash", "judge", "sonnet", "lenQ", "typeprog", "long80"):
        for part in ("train", "new"):
            sel = [x for x in rows if x["fam"] == f and x["part"] == part and not x["g0"]]
            if not sel: continue
            byrun = defaultdict(list)
            for x in sel: byrun[x["run"]].append(x)
            def m(fn):
                vals = [fn(v) for v in byrun.values()]; vals = [u for u in vals if u is not None and not np.isnan(u)]
                return np.mean(vals) if vals else np.nan
            pr("%-8s %-5s %7.1f %6.3f %6.3f %6.3f %7.2f %7.3f %7.3f" % (f, part, np.mean([len(v) for v in byrun.values()]),
                m(lambda v: np.mean([x["in32"] for x in v])), m(lambda v: np.mean([x["inC"] for x in v])), m(lambda v: np.mean([not x["inC"] for x in v])),
                m(lambda v: np.mean([x["f32"] for x in v if x["in32"]]) if any(x["in32"] for x in v) else None),
                m(lambda v: np.mean([x["paid"] for x in v if x["paid"] is not None]) if any(x["paid"] is not None for x in v) else None),
                m(lambda v: np.mean([x["base_paid"] for x in v if x["base_paid"] is not None]) if any(x["base_paid"] is not None for x in v) else None)))
