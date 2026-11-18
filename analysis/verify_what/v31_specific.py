"""(verify_what) Item 3: does WHICH wrong answer a grader pays transfer to new questions? FK20, idkc05, qwen7 / llama8, 3 sets.
Unit = (run, question) where the final greedy answer x is key-wrong (loose key), well-formed, not IDK, and CHANGED from the untrained answer.
  'changed' variants: vs base greedy (main), vs the run's own ck0 re-decode (the explorers' choice), vs both.
Verdict of judge J at level lv (a35 / a50 from thresholds_<J>.json, t50 = 0.5): P_J(idx, first line as the judge sees it) if scored,
else the mean P_J over scored strings with the same normalised answer, else missing.
lift_J = mean over units of [paid_J(x) - b_J(q)], b_J(q) = paid share among the untrained samples' key-wrong strings on q.
specific = lift_own - mean lift over the other judges of {Q, M, G, O} at the same level.
  support 'common': units where all four judges have a verdict for x and for >= 1 untrained wrong sample (main); 'avail': per judge.
Placebos: (P1) fake own judge: for each judge run, each other judge J* treated as own (the true own judge left out);
  (P2) key-trained runs (sim..fn.. graders, never saw a judge): per judge J, lift_J - mean others; difference-in-differences
       DiD = specific(J-run) - mean specific_J(key runs) of the same student, set, level;
  (P3) hash35 runs: own seed's hash table vs the other seed's table (my re-implementation of the trainer's hash).
Positive control: the 32 TriviaQA 'new' questions that duplicate a training question (same text, other idx): a judge sees the same text,
  so a memorised answer should carry the home lift there; for hash runs the training copy's idx table should predict, the copy's own should not.
Decode noise: share of questions where the run's ck0 differs from the base greedy (two untrained decodes); decode-stable questions =
  ck0 == base greedy in every run of the (student, set); the specific lift recomputed on decode-stable questions only.
Output v31_specific.txt, rows v31_rows.pkl"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import re, json, pickle, hashlib, sys, numpy as np
from collections import defaultdict
sys.path.insert(0, _RC)
import st_common as C
OUT = (_RA + "/verify_what")
IDK_FULL = re.compile(r"^(i don t know|i do not know|i dont know|idk|i m not sure|not sure|i cannot answer|cannot answer|no idea|i am not sure|unknown)$")
JS = ("Q", "M", "G", "O")
TH = {J: json.load(open(f"{_RX}/final/thresholds_{J}.json")) for J in JS}
def hunit(*parts): return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12], 16) / float(16 ** 12)
def outcome(x, al):
    if x is None or x == "": return 3
    if IDK_FULL.match(x): return 0
    return 1 if C.tq_correct(x, al) else 2
L = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); L.append(s)
def boot(v, n=4000):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if len(v) < 2: return (np.nan, np.nan)
    bs = np.random.default_rng(0).choice(v, (n, len(v))).mean(1); return tuple(np.percentile(bs, [2.5, 97.5]))
P(__doc__)
ROWS = []; NOISE = []
for st in ("qwen7", "llama8"):
    for ds, dsn in (("h1", "tqa"), ("pq", "pqa"), ("nq", "nqo")):
        Dd = pickle.load(open(f"{OUT}/cache_{st}_{ds}.pkl", "rb")); Q = Dd["Q"]; B = Dd["B"]; R = Dd["R"]; N = len(Q["train"]); al = Q["aliases"]
        JL = pickle.load(open(f"{OUT}/jl_{dsn}.pkl", "rb"))
        thr = {(J, lv): (0.5 if lv == "t50" else TH[J][f"{st}_{dsn}"][f"t_{lv}"]) for J in JS for lv in ("a35", "a50", "t50")}
        def pyes(J, i, line, x, how="line"):
            if how == "line":
                v = JL[J]["line"].get((i, line))
                if v is not None: return v
            return JL[J]["norm"].get((i, x))
        parts = {"home": [i for i in range(N) if Q["train"][i]], "new": [i for i in range(N) if not Q["train"][i] and not Q["dup"][i]],
                 "dup": [i for i in range(N) if Q["dup"][i]]}
        # untrained wrong samples per question
        wr = [[(B["sl"][i][k], B["s"][i][k]) for k in range(32) if outcome(B["s"][i][k], al[i]) == 2] for i in range(N)]
        bcache = {}
        def base_share(J, lv, i, how):
            k = (J, lv, i, how)
            if k not in bcache:
                v = [pyes(J, i, l, x, how) for l, x in wr[i]]; v = [u > thr[(J, lv)] for u in v if u is not None]
                bcache[k] = np.mean(v) if v else None
            return bcache[k]
        # decode-stability over all runs of this (student, set)
        stable = [all(r["ck0"][i] == B["g"][i] for r in R.values()) for i in range(N)]
        def units(r, items, chg):
            out = []
            for i in items:
                x = r["ck120"][i]
                if outcome(x, al[i]) != 2: continue
                c_base = x != B["g"][i]; c_ck0 = x != r["ck0"][i]
                if (chg == "base" and c_base) or (chg == "ck0" and c_ck0) or (chg == "both" and c_base and c_ck0): out.append(i)
            return out
        for name, r in sorted(R.items()):
            if r["tag"] != "FK20" or r["c"] != "05": continue
            g = r["g"]; m = re.fullmatch(r"judge([QMGO])(a35|a50|t50)", g)
            kind = "judge" if m else ("key" if re.fullmatch(r"sim\d\dfn\d\d", g) else ("hash" if g == "hash35fn00" else None))
            if kind is None: continue
            noise_new = np.mean([r["ck0"][i] != B["g"][i] for i in parts["new"]])
            NOISE.append(dict(st=st, ds=ds, run=name, kind=kind, noise_new=noise_new,
                              chg_new=np.mean([outcome(r["ck120"][i], al[i]) == 2 and r["ck120"][i] != B["g"][i] for i in parts["new"]]),
                              chg_home=np.mean([outcome(r["ck120"][i], al[i]) == 2 and r["ck120"][i] != B["g"][i] for i in parts["home"]]),
                              stable_new=np.mean([stable[i] for i in parts["new"]])))
            for part, items in parts.items():
                if part == "dup" and ds != "h1": continue
                for chg in ("base", "ck0", "both"):
                    U = units(r, items, chg)
                    for stab in ((False, True) if chg == "base" else (False,)):
                        UU = [i for i in U if stable[i]] if stab else U
                        if kind == "hash":
                            s0 = r["seed"]; other = 1 - s0
                            def hpaid(seed, i, x): return hunit("fp", seed, i, x) < 0.35
                            tabs = {"own": s0, "other": other}
                            lifts = {}
                            for tn, sd in tabs.items():
                                d = [hpaid(sd, i, r["ck120"][i]) - np.mean([hpaid(sd, i, x) for l, x in wr[i]]) for i in UU if wr[i]]
                                lifts[tn] = np.mean(d) if d else np.nan
                            if part == "dup":   # training partner's idx table, own seed
                                d = [hpaid(s0, Q["dup_of"][i], r["ck120"][i]) - np.mean([hpaid(s0, Q["dup_of"][i], x) for l, x in wr[i]]) for i in UU if wr[i]]
                                lifts["partner"] = np.mean(d) if d else np.nan
                            ROWS.append(dict(st=st, ds=ds, run=name, kind=kind, part=part, chg=chg, stab=stab, support="hash", lv="hash", own="own", n=len(UU),
                                             lift=lifts, spec=lifts["own"] - lifts["other"]))
                            continue
                        levels = [m.group(2)] if kind == "judge" else ["a35", "a50", "t50"]
                        for lv in levels:
                            for support, how in (("common", "line"), ("avail", "line"), ("avail_norm", "norm")):
                                vals = defaultdict(list)
                                for i in UU:
                                    x = r["ck120"][i]; line = r["ck120_line"][i]
                                    pv = {J: pyes(J, i, line, x, how) for J in JS}; bv = {J: base_share(J, lv, i, how) for J in JS}
                                    if support == "common" and any(pv[J] is None or bv[J] is None for J in JS): continue
                                    for J in JS:
                                        if pv[J] is None or bv[J] is None: continue
                                        vals[J].append((pv[J] > thr[(J, lv)]) - bv[J])
                                lift = {J: (np.mean(vals[J]) if len(vals[J]) >= 5 else np.nan) for J in JS}
                                own = m.group(1) if kind == "judge" else None
                                ROWS.append(dict(st=st, ds=ds, run=name, kind=kind, part=part, chg=chg, stab=stab, support=support, lv=lv, own=own,
                                                 n=min(len(v) for v in vals.values()) if vals else 0, lift=lift))
        print(st, ds, "done", len(ROWS), flush=True)
pickle.dump(dict(rows=ROWS, noise=NOISE), open(f"{OUT}/v31_rows.pkl", "wb"))
# ---------- summaries ----------
def spec_of(row, own):
    l = row["lift"]; o = [l[J] for J in JS if J != own and np.isfinite(l[J])]
    return l[own] - np.mean(o) if np.isfinite(l[own]) and o else np.nan
keyspec = defaultdict(list)   # (st, ds, part, chg, stab, support, lv, J) -> specific_J values in key runs
for x in ROWS:
    if x["kind"] == "key":
        for J in JS: keyspec[(x["st"], x["ds"], x["part"], x["chg"], x["stab"], x["support"], x["lv"], J)].append(spec_of(x, J))
P("\n=== Judge-trained runs: specific lift = lift(own judge) - mean lift(other judges), per run; mean [95% bootstrap CI over runs]")
for chg, stab, support in (("base", False, "common"), ("base", False, "avail"), ("ck0", False, "avail_norm"), ("both", False, "common"), ("base", True, "common")):
    P(f"-- changed vs {chg}{', decode-stable questions only' if stab else ''}; support {support}" + ("   <- closest to the claim's definition" if (chg, support) == ("ck0", "avail_norm") else ""))
    for st in ("qwen7", "llama8"):
        for part in ("home", "new", "dup"):
            X = [x for x in ROWS if x["kind"] == "judge" and x["st"] == st and x["part"] == part and x["chg"] == chg and x["stab"] == stab and x["support"] == support]
            if not X: continue
            sp = [spec_of(x, x["own"]) for x in X]
            fake = [np.nanmean([x["lift"][Js] - np.nanmean([x["lift"][J2] for J2 in JS if J2 not in (x["own"], Js)]) for Js in JS if Js != x["own"]]) for x in X]
            did = [spec_of(x, x["own"]) - np.nanmean(keyspec[(x["st"], x["ds"], part, chg, stab, support, x["lv"], x["own"])] or [np.nan]) for x in X]
            ownl = [x["lift"][x["own"]] for x in X]; oth = [np.nanmean([x["lift"][J] for J in JS if J != x["own"]]) for x in X]
            c1, c2, c3 = boot(sp), boot(fake), boot(did)
            P(f"   {st:6s} {part:4s} runs {sum(np.isfinite(sp)):3d} units/run {np.mean([x['n'] for x in X]):6.1f} | own lift {np.nanmean(ownl):+.3f} other {np.nanmean(oth):+.3f} | "
              f"specific {np.nanmean(sp):+.3f} [{c1[0]:+.3f}, {c1[1]:+.3f}] | P1 fake-own {np.nanmean(fake):+.3f} [{c2[0]:+.3f}, {c2[1]:+.3f}] | P2 DiD vs key runs {np.nanmean(did):+.3f} [{c3[0]:+.3f}, {c3[1]:+.3f}]")
P("\n=== Key-trained runs (never saw a judge): mean over runs of lift_J - mean others, per judge, level a35, changed vs base, common support")
for st in ("qwen7", "llama8"):
    for part in ("home", "new"):
        P(f"   {st:6s} {part:4s} " + "  ".join(f"{J} {np.nanmean([v for (a, b, p, c, s, sup, lv, JJ), vs in keyspec.items() for v in vs if a == st and p == part and c == 'base' and not s and sup == 'common' and lv == 'a35' and JJ == J]):+.3f}" for J in JS))
P("\n=== Hash35 runs: lift under the run's own seed table vs the other seed's table (changed vs base greedy)")
for stab in (False, True):
    for st in ("qwen7", "llama8"):
        for part in ("home", "new", "dup"):
            X = [x for x in ROWS if x["kind"] == "hash" and x["st"] == st and x["part"] == part and x["chg"] == "base" and x["stab"] == stab]
            if not X: continue
            sp = [x["spec"] for x in X]; c = boot(sp)
            extra = f" | training-partner idx table lift {np.nanmean([x['lift'].get('partner', np.nan) for x in X]):+.3f}" if part == "dup" else ""
            P(f"   {'stable ' if stab else ''}{st:6s} {part:4s} runs {len(X)} units/run {np.mean([x['n'] for x in X]):6.1f} | own {np.nanmean([x['lift']['own'] for x in X]):+.3f} "
              f"other {np.nanmean([x['lift']['other'] for x in X]):+.3f} | own - other {np.nanmean(sp):+.3f} [{c[0]:+.3f}, {c[1]:+.3f}]{extra}")
P("\n=== Decode noise: share of NEW questions where the run's ck0 differs from the base greedy (two untrained decodes) vs changed key-wrong finals")
for st in ("qwen7", "llama8"):
    for ds in ("h1", "pq", "nq"):
        X = [x for x in NOISE if x["st"] == st and x["ds"] == ds]
        P(f"   {st:6s} {ds}: ck0 != base greedy {100 * np.mean([x['noise_new'] for x in X]):.1f}% | changed key-wrong finals: new {100 * np.mean([x['chg_new'] for x in X]):.1f}% home "
          f"{100 * np.mean([x['chg_home'] for x in X]):.1f}% | decode-stable new questions (all runs' ck0 == base) {100 * X[0]['stable_new']:.1f}%")
open(f"{OUT}/v31_specific.txt", "w").write("\n".join(L) + "\n")
