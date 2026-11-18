"""(rev_R1) Check: does a judge's verdict on WRONG untrained answers (false accepts vs true rejects) shape where the trained
student answers on NEW questions as much as its verdict on RIGHT untrained answers (true accepts vs false rejects)? Same estimator as
verify_what/v33 (Q judge FK20 runs vs abstention-matched key-trained FK20 runs, |abstention gap| <= 0.05 on new questions):
  sel = P(run answers | untrained greedy judge-paid) - P(run answers | untrained greedy judge-rejected), on questions whose untrained greedy
  answer is key-WRONG (FA side, = v33) or key-RIGHT (FR side, new here). Read-only on verify_what caches. Variant (r3b): FA side restricted to key-wrong AND Sonnet-audit-wrong (level < 4) greedy answers, i.e. drops key misses the judge may
  rightly accept; FR side restricted to key-right greedy answers. Output: r3b_when_audited_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import re, json, pickle, sys, numpy as np, pandas as pd
sys.path.insert(0, _RC)
import st_common as C
V = (_RA + "/verify_what"); OUT = (_RA + "/rev_R1")
IDK_FULL = re.compile(r"^(i don t know|i do not know|i dont know|idk|i m not sure|not sure|i cannot answer|cannot answer|no idea|i am not sure|unknown)$")
TH = json.load(open((_RX + "/final/thresholds_Q.json")))
def outcome(x, al):
    if x is None or x == "": return 3
    if IDK_FULL.match(x): return 0
    return 1 if C.tq_correct(x, al) else 2
L = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); L.append(s)
P(__doc__)
rows = []
for st in ("qwen7", "llama8"):
    for ds, dsn in (("h1", "tqa"), ("pq", "pqa"), ("nq", "nqo")):
        SON = pickle.load(open(f"{_RA}/verify_where/cache/base_{st}_{dsn}.pkl", "rb"))["son"][:, 0]
        Dd = pickle.load(open(f"{V}/cache_{st}_{ds}.pkl", "rb")); Q = Dd["Q"]; B = Dd["B"]; R = Dd["R"]; N = len(Q["train"]); al = Q["aliases"]
        JQ = pickle.load(open(f"{V}/jl_{dsn}.pkl", "rb"))["Q"]
        def pyes(i, line, x):
            v = JQ["line"].get((i, line)); return v if v is not None else JQ["norm"].get((i, x))
        thr = {"a35": TH[f"{st}_{dsn}"]["t_a35"], "a50": TH[f"{st}_{dsn}"]["t_a50"], "t50": 0.5}
        go = [outcome(B["g"][i], al[i]) for i in range(N)]
        p0 = [pyes(i, B["gl"][i], B["g"][i]) if go[i] in (1, 2) else None for i in range(N)]
        new = [i for i in range(N) if not Q["train"][i] and not Q["dup"][i]]
        for name, r in R.items():
            if r["tag"] != "FK20" or r["c"] != "05": continue
            m = re.fullmatch(r"judgeQ(a35|a50|t50)", r["g"]); key = re.fullmatch(r"sim\d\dfn\d\d", r["g"])
            if not (m or key): continue
            o = [outcome(r["ck120"][i], al[i]) for i in range(N)]
            a_new = np.mean([o[i] == 0 for i in new])
            for lv in ([m.group(1)] if m else ["a35", "a50", "t50"]):
                t = thr[lv]
                for side, gcode in (("FA side (greedy wrong)", 2), ("FR side (greedy right)", 1)):
                    U = [i for i in new if p0[i] is not None and go[i] == gcode and (gcode == 1 or SON[i] < 4)]
                    pa = [o[i] in (1, 2) for i in U if p0[i] > t]; rj = [o[i] in (1, 2) for i in U if p0[i] <= t]
                    rows.append(dict(st=st, ds=ds, run=name, kind="judgeQ" if m else "key", lv=lv, side=side, a_new=a_new,
                                     sel=(np.mean(pa) - np.mean(rj)) if len(pa) >= 5 and len(rj) >= 5 else np.nan, npa=len(pa), nrj=len(rj)))
        del Dd, JQ
T = pd.DataFrame(rows)
for lv in ("a35", "a50", "t50"):
    for st in ("qwen7", "llama8"):
        for side in ("FA side (greedy wrong)", "FR side (greedy right)"):
            J = T[(T.kind == "judgeQ") & (T.lv == lv) & (T.st == st) & (T.side == side)]; K = T[(T.kind == "key") & (T.lv == lv) & (T.st == st) & (T.side == side)]
            dm = []
            for _, x in J.iterrows():
                mk = K[(K.ds == x.ds) & (np.abs(K.a_new - x.a_new) <= 0.05)]
                if len(mk) and np.isfinite(x.sel): dm.append(x.sel - mk.sel.mean())
            P(f"{lv} {st:6s} {side:23s}: judge runs {len(J):2d} sel {J.sel.mean():+.3f} | key runs sel {K.sel.mean():+.3f} | abstention-matched judge - key "
              f"({len(dm)} runs): {np.nanmean(dm) if dm else float('nan'):+.3f} | mean n paid/rejected per run {J.npa.mean():.0f}/{J.nrj.mean():.0f}")
open(f"{OUT}/r3b_when_audited_out.txt", "w").write("\n".join(L) + "\n")
