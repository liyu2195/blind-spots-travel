"""fix_sc f1: RL minus self-consistency SC_m at MATCHED COVERAGE on NEW questions, per grader family x operating point x student,
including the capped runs (runs that answer more questions than the untrained greedy answers).

SC_m (m = generations per question; the RL policy costs 1): the untrained greedy answer plus m-1 of the 32 untrained samples
(drawn at random per question without replacement; D = 24 nested draws, i.e. SC_4's samples contain SC_2's).
  answer  = the untrained greedy answer if it is an answer; otherwise the plurality answer of the m-1 drawn samples (extension for
            capped runs; ties between clusters broken at random); no answer if neither exists.
  score   = vote share of that answer among all m generations: (1 + #samples agreeing with the greedy answer) / m (rc.same), or
            #samples in the plurality cluster / m when the greedy declined (its IDK is a generation that does not vote for the answer).
  m = 1   = greedy only, no score: refuse at random among the greedy answers (k <= #greedy answers) / refuse nothing (otherwise).
  m = 33  = greedy + all 32 samples (the rule of the earlier section, selfcons s2), reported for continuity.
  SC answers the k highest-scoring questions, k = the run's answer count (boundary ties split = random tie-breaking).
  If SC_m has fewer than k answers ('unreachable'), the missing slots count as WRONG (main, 'lb': conservative for any claim that RL
  does not beat SC); 'ub' instead compares at the largest coverage SC_m reaches. SC-lex = same rule but greedy answers always ranked
  before sampled ones (= selfcons s2's SC whenever k <= #greedy answers).
Truth = audited (loose key OR Sonnet level >= 4; key-wrong answers without a Sonnet verdict count wrong); 'key' = loose key only
for both RL and SC. Gap = RL selective accuracy - SC_m selective accuracy at the same k (pp); at matched coverage this equals
-(RL - SC wrong answers per 100 answered).
Split of the m-specific gap: hybrid H = RL's answered set answered with SC_m's answer (0 where SC_m has none):
  selection = H - SC_m ; changed answers = RL - H.
Equivalent sample count m*: where the (group-mean) gap curve over m = 1, 2, 4, 8, 16, 32 crosses 0, log2-linear interpolation;
'<1' = RL below random refusal; '>32' = RL above SC_32.
Intervals: crossed bootstrap - 1,000 multinomial resamples of each dataset's new questions (shared by all runs of that dataset; the rule
is re-applied at the run's resampled answer count inside every resample) x resampling runs with replacement within dataset.
Runs answering < 5% of new questions are dropped (listed).
Output: f1_scm.txt, f1_rows.pkl.   Usage: python3 f1_scm.py"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, re, pickle, zlib, numpy as np
from collections import defaultdict
sys.path.insert(0, (_RA + "/selfcons"))
from sc_lib import RuleBoot, frac_select, boot_weights, ci, grp, STS, DSS, DSNAME
OUT = (_RA + "/fix_sc"); SCD = (_RA + "/selfcons")
MS = [1, 2, 4, 8, 16, 32]; MX = MS + [33]; D = 24; B = 1000
lines = []
def pr(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); lines.append(s)

def fam2(fam):
    m = re.fullmatch(r"J([A-Z]+?)_(t50|a50|a35)(f1|q)?", fam)
    if m: return f"judge {m.group(1)} {m.group(2)}" + ({"f1": " forcedIDK", "q": " quorum"}.get(m.group(3), ""))
    return {"key": "key", "keyFN": "key+FN noise", "keyFP": "key+FA noise", "keyHash": "key+FA noise", "keyLenQ": "key+FA noise",
            "keyMix": "key+mixed noise", "vote": "own vote", "sonnet": "Sonnet",
            "pl_fnType": "planted FR", "pl_fnHalf": "planted FR", "pl_real": "planted FR", "pl_scr": "planted FR", "pl_sharp": "planted FR",
            "pl_typeFP": "planted FA", "pl_popFP": "planted FA", "keyLong": "planted FA", "pl_blind": "blind (pays any answer on a subset)",
            "frzOR": "other frozen keys", "peer": "other frozen keys", "jveto": "other frozen keys"}.get(fam, "other:" + fam)

ROWS = ["key", "key+noise (all)", "key+FN noise", "key+FA noise", "key+mixed noise", "own vote",
        "judge Q t50", "judge Q a50", "judge Q a35", "judge M t50", "judge M a35", "judge G t50", "judge G a35", "judge O t50", "judge O a35",
        "judge R a35", "judge Q t50 forcedIDK", "judge Q t50 quorum", "Sonnet", "planted (all)", "planted FR", "planted FA",
        "blind (pays any answer on a subset)", "other frozen keys"]
def rows_of(f2):
    out = [f2]
    if f2.startswith("key+"): out.append("key+noise (all)")
    if f2.startswith("planted") or f2.startswith("blind"): out.append("planted (all)")
    return out

def sc_rules(S, M, rng):
    """yields (m, d, score, valid, lab_aud, lab_key, lab_hi, lex_score) for every m and draw; m = 1 and 33 deterministic (d = 0)"""
    gval = S["gval"][M]; gaud = S["gaud"][M]; gkey = S["gkey"][M]; ghi = S["ghi"][M]
    kind = S["kind"][M]; agr = S["agr"][M]; clu = S["clu"][M]; cA = S["cl_aud"][M]; cK = S["cl_key"][M]; cH = S["cl_hi"][M]
    N = len(gval); C_ = cA.shape[1]; ar = np.arange(N)
    def make(agree, cnt, L, tiebreak):
        top = np.argmax(cnt + tiebreak, 1); has = cnt[ar, top] > 0
        sc = np.where(gval, (agree + 1.0) / (L + 1.0), np.where(has, cnt[ar, top] / (L + 1.0), 0.0)); val = gval | has
        la = np.where(gval, gaud, cA[ar, top] & has); lk = np.where(gval, gkey, cK[ar, top] & has); lh = np.where(gval, ghi, cH[ar, top] & has)
        return sc, val, la, lk, lh
    yield (1, 0, np.zeros(N), gval.copy(), gaud.copy(), gkey.copy(), ghi.copy(), np.zeros(N))
    cnt = np.zeros((N, C_)); ok = clu >= 0
    for j in range(32): cnt[ar[ok[:, j]], clu[ok[:, j], j]] += 1
    sc, val, la, lk, lh = make(agr.sum(1).astype(float), cnt, 32.0, np.zeros((N, C_)))   # all 32: first max (= s0 plurality)
    yield (33, 0, sc, val, la, lk, lh, sc + 2.0 * gval)
    for d in range(D):
        perm = np.argsort(rng.random((N, 32)), 1); cnt = np.zeros((N, C_)); agree = np.zeros(N); tb = rng.random((N, C_)) * 1e-3
        for j in range(31):
            col = perm[:, j]; cj = clu[ar, col]; okj = cj >= 0
            cnt[ar[okj], cj[okj]] += 1; agree += agr[ar, col]
            L = j + 1
            if L + 1 in MS:
                sc, val, la, lk, lh = make(agree, cnt, float(L), tb)
                yield (L + 1, d, sc, val, la, lk, lh, sc + 2.0 * gval)

def main():
    pr(__doc__)
    rows = []; BOOT = {}; TINY = []
    for st in STS:
        for ds in DSS:
            P = pickle.load(open(f"{SCD}/cache_{st}_{ds}.pkl", "rb")); S = pickle.load(open(f"{OUT}/samp_{st}_{ds}.pkl", "rb"))
            S["ghi"] = P["Q"]["gaud"] | P["Q"]["gaud_miss"]
            M = P["Q"]["ho"].copy(); N = int(M.sum()); gval = S["gval"][M]; V = int(gval.sum())
            W = boot_weights(N, B, seed=zlib.crc32(f"{st}{ds}fixsc".encode()))
            rng = np.random.default_rng(zlib.crc32(f"{st}{ds}draws".encode()))
            RUL = []
            for (m, d, sc, val, la, lk, lh, lex) in sc_rules(S, M, rng):
                labs = {"a": la.astype(float), "k": lk.astype(float), "h": lh.astype(float)}
                rb = RuleBoot(sc, val, labs, W)
                rbl = RuleBoot(lex, val, {"a": labs["a"]}, W) if m in (32, 33) else None
                RUL.append((m, d, sc, val, labs, lex, rb, rbl))
            maxcov32 = np.mean([r[3].sum() for r in RUL if r[0] == 32]); maxcov33 = [r[3].sum() for r in RUL if r[0] == 33][0]
            pr(f"[{st} {ds}] new {N}, greedy answers {V}, SC_32 can answer {maxcov32:.1f} (all 32 samples: {maxcov33}), rule objects {len(RUL)}")
            for name in sorted(P["R"]):
                r = P["R"][name]; o = r["o"][M]; ans = (o == 1) | (o == 2); k = int(ans.sum())
                if k < 0.05 * N: TINY.append(f"{st}:{ds}:{grp(r['tag'], r['fam'], r['credit'])}"); continue
                aud = r["aud"][M].astype(float); key = (o == 1).astype(float); hi = (r["aud"][M] | r["miss"][M]).astype(float)
                a_ = ans.astype(float)
                d = dict(st=st, ds=ds, name=name, tag=r["tag"], credit=r["credit"], fam=r["fam"], f2=fam2(r["fam"]), s=r["s"],
                         grp=grp(r["tag"], r["fam"], r["credit"]), N=N, k=k, V=V, cov=k / N, cap=bool(k > V),
                         accRL=float(aud[ans].mean()), accRL_key=float(key[ans].mean()), accRL_hi=float(hi[ans].mean()),
                         n_beyond=int((ans & ~RUL[1][3]).sum()), acc_beyond=float(aud[ans & ~RUL[1][3]].mean()) if (ans & ~RUL[1][3]).any() else np.nan,
                         n_ongidk=int((ans & ~gval).sum()))
                kb = W @ a_; rl_b = W @ (a_ * aud); rk_b = W @ (a_ * key); accRL_b = rl_b / kb
                acc = defaultdict(list); bt = defaultdict(list)
                for (m, dd, sc, val, labs, lex, rb, rbl) in RUL:
                    w = frac_select(sc, val, k); nans = w.sum()
                    ra = (w * labs["a"]).sum(); rk = (w * labs["k"]).sum(); rh = (w * labs["h"]).sum()
                    hl = np.where(gval, labs["a"], labs["a"] * val)          # hybrid: SC_m's answer on RL's answered set (0 if none)
                    hacc = float(hl[ans].mean())
                    acc[(m, "lb")].append(ra / k); acc[(m, "ub")].append(ra / max(nans, 1e-9)); acc[(m, "key")].append(rk / k)
                    acc[(m, "hi")].append(rh / k); acc[(m, "H")].append(hacc); acc[(m, "reach")].append(float(nans >= k - 1e-6))
                    if rbl is not None:
                        wl = frac_select(lex, val, k); acc[(m, "lex")].append((wl * labs["a"]).sum() / k)
                    rr, nn = rb.right(kb, "a"); bt[(m, "lb")].append(accRL_b - rr / kb); bt[(m, "ub")].append(accRL_b - rr / np.maximum(nn, 1e-9))
                    h_b = W @ (a_ * hl); bt[(m, "sel")].append((h_b - rr) / kb); bt[(m, "ans")].append((rl_b - h_b) / kb)
                    if m in (32, 33):
                        rkk, _ = rb.right(kb, "k"); bt[(m, "key")].append(rk_b / kb - rkk / kb)
                for m in MX:
                    d[f"lb{m}"] = d["accRL"] - np.mean(acc[(m, "lb")]); d[f"ub{m}"] = d["accRL"] - np.mean(acc[(m, "ub")])
                    d[f"sel{m}"] = np.mean(acc[(m, "H")]) - np.mean(acc[(m, "lb")]); d[f"ans{m}"] = d["accRL"] - np.mean(acc[(m, "H")])
                    d[f"key{m}"] = d["accRL_key"] - np.mean(acc[(m, "key")]); d[f"hi{m}"] = d["accRL_hi"] - np.mean(acc[(m, "hi")])
                    d[f"reach{m}"] = np.mean(acc[(m, "reach")])
                    if (m, "lex") in acc: d[f"lex{m}"] = d["accRL"] - np.mean(acc[(m, "lex")])
                rows.append(d)
                BOOT[name] = {kx: np.stack([np.mean(bt[(m, kx)], 0) for m in MX]).astype(np.float32) for kx in ("lb", "ub", "sel", "ans")}
                BOOT[name]["key"] = np.stack([np.mean(bt[(m, "key")], 0) for m in (32, 33)]).astype(np.float32)
            del P, S, RUL
    pickle.dump(dict(rows=rows, BOOT=BOOT, TINY=TINY, MX=MX), open(f"{OUT}/f1_rows.pkl", "wb"))
    report(rows, BOOT, TINY)

def mstar(g):
    """g: gap curve over MS (pp or fraction); returns m* as float (0.5 = '<1', 64 = '>32')"""
    if g[0] <= 0: return 0.5
    for j in range(1, len(MS)):
        if g[j] <= 0:
            x = (j - 1) + g[j - 1] / (g[j - 1] - g[j]); return 2.0 ** x
    return 64.0
def fm(x): return "<1" if x < 1 else (">32" if x > 32 else f"{x:.1f}")

def crossed(L, BOOT, key, rng, nb=B):
    """crossed bootstrap of the group mean: resample runs within dataset (with replacement) x question resample b.  returns (nb, n_m)"""
    A = np.stack([BOOT[x["name"]][key] for x in L])                  # n_runs x n_m x B
    byds = defaultdict(list)
    for i, x in enumerate(L): byds[x["st"] + x["ds"]].append(i)
    idx = np.concatenate([rng.choice(v, size=(nb, len(v)), replace=True) for v in byds.values()], 1)   # nb x n_runs
    return A[idx, :, np.arange(nb)[:, None]].mean(1)                  # nb x n_m

def group_stats(L, BOOT, rng):
    mi = {m: i for i, m in enumerate(MX)}
    g = {m: np.mean([x[f"lb{m}"] for x in L]) for m in MX}
    cb = crossed(L, BOOT, "lb", rng); cs = crossed(L, BOOT, "sel", rng); ca = crossed(L, BOOT, "ans", rng); cu = crossed(L, BOOT, "ub", rng)
    ms = mstar([g[m] for m in MS]); msb = np.array([mstar(cb[b, :len(MS)]) for b in range(cb.shape[0])])
    return dict(n=len(L), ncap=sum(x["cap"] for x in L), cov=np.mean([x["cov"] for x in L]), acc=np.mean([x["accRL"] for x in L]), g=g,
                ci32=ci(cb[:, mi[32]]), ci33=ci(cb[:, mi[33]]), ci1=ci(cb[:, mi[1]]), ci4=ci(cb[:, mi[4]]), mstar=ms, mci=(np.percentile(msb, 2.5), np.percentile(msb, 97.5)),
                sel32=np.mean([x["sel32"] for x in L]), ans32=np.mean([x["ans32"] for x in L]), selci=ci(cs[:, mi[32]]), ansci=ci(ca[:, mi[32]]),
                ub32=np.mean([x["ub32"] for x in L]), ub1=np.mean([x["ub1"] for x in L]), ubci=ci(cu[:, mi[32]]), key32=np.mean([x["key32"] for x in L]),
                hi32=np.mean([x["hi32"] for x in L]), lex32=np.mean([x["lex32"] for x in L]), lex33=np.mean([x["lex33"] for x in L]),
                pos32=np.mean([x["lb32"] > 0 for x in L]), reach32=np.mean([x["reach32"] for x in L]), mstar_runs=np.median([mstar([x[f"lb{m}"] for m in MS]) for x in L]),
                beyond=sum(x["n_beyond"] for x in L), acc_beyond=np.nanmean([x["acc_beyond"] for x in L]) if any(x["n_beyond"] for x in L) else np.nan)

def report(rows, BOOT, TINY):
    rng = np.random.default_rng(20261116)
    main = [d for d in rows if d["tag"] == "FK20" and d["credit"] == "idkc05"]
    hdr = (f"{'family / operating point':34s} {'n':>3s} {'cap':>3s} {'cov':>4s} {'accRL':>5s} | RL - SC_m, pp (missing SC slots wrong): "
           + " ".join(f"{'m='+str(m):>6s}" for m in MX) + f" | {'RL-SC_32 [95% CI]':20s} | {'m* [95% CI]':17s} {'runs':>4s} | {'select':>6s} {'answer':>6s} | "
           f"{'ub32':>5s} {'key32':>5s} {'hi32':>5s} {'lex32':>5s} | {'>0':>4s} {'reach':>5s} | beyond-reach answers (acc)")
    SUM = {}
    for st in STS:
        pr(f"\n\n################ {st} ({'DISCOVERY' if st == 'qwen7' else 'CONFIRMATION'}): FK20, IDK credit 0.5, new questions, capped runs INCLUDED ################")
        for scope in ["ALL"] + list(DSS):
            pr(f"\n=== {st} {'pooled over datasets' if scope == 'ALL' else DSNAME[scope]}"); pr(hdr)
            for rname in ROWS:
                L = [d for d in main if d["st"] == st and (scope == "ALL" or d["ds"] == scope) and rname in rows_of(d["f2"])]
                if not L: continue
                G = group_stats(L, BOOT, rng)
                if scope == "ALL": SUM[(st, rname)] = G
                pr(f"{rname:34s} {G['n']:3d} {G['ncap']:3d} {G['cov']:4.2f} {100*G['acc']:5.1f} | " + " ".join(f"{100*G['g'][m]:+6.1f}" for m in MX)
                   + f" | {100*G['g'][32]:+5.1f} [{100*G['ci32'][0]:+5.1f},{100*G['ci32'][1]:+5.1f}] | {fm(G['mstar']):>4s} [{fm(G['mci'][0])},{fm(G['mci'][1])}]"
                   + f"{'':3s} {fm(G['mstar_runs']):>4s} | {100*G['sel32']:+6.1f} {100*G['ans32']:+6.1f} | {100*G['ub32']:+5.1f} {100*G['key32']:+5.1f} {100*G['hi32']:+5.1f} "
                   f"{100*G['lex32']:+5.1f} | {G['pos32']:4.2f} {G['reach32']:5.2f} | {G['beyond']} ({G['acc_beyond']:.2f})")
        # capped vs not capped
        pr(f"\n--- {st}: capped vs not capped (FK20 idkc05), RL - SC_32 lb / ub, selection / answers")
        for cap in (False, True):
            L = [d for d in main if d["st"] == st and d["cap"] == cap]
            if not L: continue
            G = group_stats(L, BOOT, rng)
            pr(f"   {'capped' if cap else 'not capped':10s} runs {G['n']:3d}: lb {100*G['g'][32]:+.2f} [{100*G['ci32'][0]:+.2f},{100*G['ci32'][1]:+.2f}]  ub {100*G['ub32']:+.2f} "
               f"[{100*G['ubci'][0]:+.2f},{100*G['ubci'][1]:+.2f}]  m=1 lb {100*G['g'][1]:+.2f} ub {100*G['ub1']:+.2f}  sel {100*G['sel32']:+.2f} ans {100*G['ans32']:+.2f}  "
               f"families: {sorted(set(d['f2'] for d in L))}")
        # continuity with selfcons s2 (not capped, SC-lex with all 32 samples == s2's SC)
        L = [d for d in main if d["st"] == st and not d["cap"]]
        pr(f"--- {st}: continuity, not-capped FK20 idkc05 runs ({len(L)}): RL - SC-lex(all 32) {100*np.mean([d['lex33'] for d in L]):+.2f} (selfcons s2 reported -0.77 / -1.36), "
           f"RL - SC-ext(all 32) {100*np.mean([d['lb33'] for d in L]):+.2f}, RL - SC_32 (31 samples) {100*np.mean([d['lb32'] for d in L]):+.2f}; "
           f"max |ext - lex| over runs at m=33 {100*max(abs(d['lb33'] - d['lex33']) for d in L):.2f} pp")
    # other recipes
    pr("\n\n=== other recipes (not FK20 or IDK credit != 0.5), pooled over datasets; m = 1..33 gaps (lb)")
    pr(hdr)
    oth = defaultdict(list)
    for d in rows:
        if not (d["tag"] == "FK20" and d["credit"] == "idkc05"): oth[(d["st"], d["grp"])].append(d)
    for (st, g_), L in sorted(oth.items()):
        G = group_stats(L, BOOT, rng)
        pr(f"{st+' '+g_:34s} {G['n']:3d} {G['ncap']:3d} {G['cov']:4.2f} {100*G['acc']:5.1f} | " + " ".join(f"{100*G['g'][m]:+6.1f}" for m in MX)
           + f" | {100*G['g'][32]:+5.1f} [{100*G['ci32'][0]:+5.1f},{100*G['ci32'][1]:+5.1f}] | {fm(G['mstar']):>4s} [{fm(G['mci'][0])},{fm(G['mci'][1])}]"
           + f"{'':3s} {fm(G['mstar_runs']):>4s} | {100*G['sel32']:+6.1f} {100*G['ans32']:+6.1f} | {100*G['ub32']:+5.1f} {100*G['key32']:+5.1f} {100*G['hi32']:+5.1f} "
           f"{100*G['lex32']:+5.1f} | {G['pos32']:4.2f} {G['reach32']:5.2f} | {G['beyond']} ({G['acc_beyond']:.2f})")
    pr(f"\n--- runs dropped for coverage < 5%: {len(TINY)} " + ", ".join(sorted(set(TINY))))
    pickle.dump(SUM, open(f"{OUT}/f1_summary.pkl", "wb"))
    open(f"{OUT}/f1_scm.txt", "w").write("\n".join(lines) + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        X = pickle.load(open(f"{OUT}/f1_rows.pkl", "rb")); pr(__doc__); report(X["rows"], X["BOOT"], X["TINY"])
    else: main()
