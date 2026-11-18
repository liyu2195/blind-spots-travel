"""(verify_what) Item 1: does RL change WHAT the model says on NEW questions, or only WHEN it answers?
Runs: every qwen7 / llama8 FK20 run with IDK credit 0.5 (tag FK20, idkc05), final greedy ck120. New questions = not in <set>_h1.jsonl;
main analysis drops the 32 TriviaQA new questions that duplicate a training question (same qid), a sensitivity row keeps them.
(A) Of the key-right final answers on new questions (key = st_common.tq_correct, loose alias match, as the claims use), the share that
    equal the untrained greedy answer / equal one of the 32 untrained samples (but not the greedy) / neither, under three notions of 'equal':
      exact   = identical normalised string (parse_tq);
      contain = identical, or one string contains the other as whole words (>= 3 characters), my own implementation;
      key     = the untrained greedy (resp. any untrained sample) was itself key-right on that question (semantic: 'already known').
    Reference row: the same shares for the untrained model re-decoded inside each run (ck0) instead of the trained model = decode noise.
(B) Decomposition per run of accuracy among answered on new questions:  acc|ans - base = sel + chg with
      base = untrained greedy key-accuracy over ALL new questions (IDK / malformed = not right; the claim's baseline),
      sel = base|A - base (base|A = untrained greedy key-accuracy on the questions the run answers), chg = acc|ans - base|A.
    Alternative baseline: untrained greedy accuracy among the questions the UNTRAINED model answers (its own acc|ans).
    Runs with 0.05 <= refusal <= 0.95 as in the claim; all runs as a check.
Output v10_knowledge.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import re, pickle, numpy as np, sys
sys.path.insert(0, _RC)
import st_common as C
OUT = (_RA + "/verify_what")
IDK_FULL = re.compile(r"^(i don t know|i do not know|i dont know|idk|i m not sure|not sure|i cannot answer|cannot answer|no idea|i am not sure|unknown)$")
L = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); L.append(s)
def contain(a, b):
    if a == b: return True
    if len(a) >= 3 and re.search(r"\b" + re.escape(a) + r"\b", b): return True
    if len(b) >= 3 and re.search(r"\b" + re.escape(b) + r"\b", a): return True
    return False
def outcome(x, al):
    if x is None or x == "": return 3
    if IDK_FULL.match(x): return 0
    return 1 if C.tq_correct(x, al) else 2
P(__doc__)
TOT = {k: np.zeros(3) for k in ("exact", "contain", "key", "exact_ck0", "contain_ck0", "key_dupincl")}
REF = {k: np.zeros(3) for k in ("exact", "contain")}
rows = []
extra = dict(n_f0=0, n_f0_alias=0, n_right=0, n_alias_new=0)
for st in ("qwen7", "llama8"):
    for ds in ("h1", "pq", "nq"):
        Dd = pickle.load(open(f"{OUT}/cache_{st}_{ds}.pkl", "rb")); Q = Dd["Q"]; B = Dd["B"]; N = len(Q["train"])
        new = [i for i in range(N) if not Q["train"][i] and not Q["dup"][i]]
        new_d = [i for i in range(N) if not Q["train"][i]]
        al = Q["aliases"]
        go = [outcome(B["g"][i], al[i]) for i in range(N)]
        samp = [[y for y in B["s"][i] if y and not IDK_FULL.match(y)] for i in range(N)]
        skey = [any(C.tq_correct(y, al[i]) for y in samp[i]) for i in range(N)]
        base_all = np.mean([go[i] == 1 for i in new])
        unt_ans = [i for i in new if go[i] in (1, 2)]; base_unt_ans = np.mean([go[i] == 1 for i in unt_ans])
        cnt = {k: np.zeros(3) for k in TOT}; nr = 0
        memo = {}
        def cat(i, x, how):
            k = (i, x, how)
            if k in memo: return memo[k]
            gx = B["g"][i]
            if how == "key":
                c = 0 if go[i] == 1 else (1 if skey[i] else 2)
            else:
                eq = (lambda a, b: a == b) if how == "exact" else contain
                c = 0 if (gx and eq(x, gx)) else (1 if any(eq(x, y) for y in samp[i]) else 2)
            memo[k] = c; return c
        for name, r in Dd["R"].items():
            if r["tag"] != "FK20" or r["c"] != "05": continue
            nr += 1
            o = [outcome(r["ck120"][i], al[i]) for i in range(N)]
            for i in new:
                if o[i] != 1: continue
                x = r["ck120"][i]
                for how in ("exact", "contain", "key"): cnt[how][cat(i, x, how)] += 1
                # equality against the run's own untrained re-decode (ck0) instead of the base greedy
                x0 = r["ck0"][i]
                for how, eq in (("exact_ck0", lambda a, b: a == b), ("contain_ck0", contain)):
                    cnt[how][0 if (x0 and eq(x, x0)) else (1 if any(eq(x, y) for y in samp[i]) else 2)] += 1
                if cat(i, x, "contain") == 2:
                    extra["n_f0"] += 1; extra["n_f0_alias"] += Q["alias_shared"][i]
                extra["n_right"] += 1; extra["n_alias_new"] += Q["alias_shared"][i]
            for i in new_d:
                if o[i] == 1: cnt["key_dupincl"][cat(i, r["ck120"][i], "key")] += 1
            # decode-noise reference: the untrained model re-decoded in this run (ck0), its key-right answers vs the base greedy
            for i in new:
                x0 = r["ck0"][i]
                if outcome(x0, al[i]) != 1: continue
                for how in ("exact", "contain"):
                    eq = (lambda a, b: a == b) if how == "exact" else contain
                    gx = B["g"][i]
                    REF[how][0 if (gx and eq(x0, gx)) else (1 if any(eq(x0, y) for y in samp[i]) else 2)] += 1
            A = [i for i in new if o[i] in (1, 2)]
            a = np.mean([o[i] == 0 for i in new])
            if not A: continue
            acc = np.mean([o[i] == 1 for i in A]); bA = np.mean([go[i] == 1 for i in A])
            rows.append(dict(st=st, ds=ds, run=name, a=a, acc=acc, base=base_all, base_unt=base_unt_ans, bA=bA, sel=bA - base_all, chg=acc - bA,
                             sel_alt=bA - base_unt_ans, n_new=len(new)))
        for k in TOT: TOT[k] += cnt[k]
        c = cnt
        P(f"{st} {ds}: {nr} runs, new questions {len(new)} (untrained greedy key-acc {base_all:.3f}, among its answered {base_unt_ans:.3f}); "
          f"key-right final answers {int(c['exact'].sum())}")
        for how in ("exact", "contain", "key", "contain_ck0"):
            v = c[how] / c[how].sum()
            P(f"     {how:12s} same as untrained greedy {v[0]:.3f}  other untrained sample {v[1]:.3f}  neither {v[2]:.4f}")
P("\nALL six pairs (dedup new questions):")
for how in TOT:
    v = TOT[how] / TOT[how].sum()
    P(f"  {how:12s} n {int(TOT[how].sum())}: same as untrained greedy {v[0]:.3f}  other untrained sample {v[1]:.3f}  neither {v[2]:.4f}")
for how in REF:
    v = REF[how] / REF[how].sum()
    P(f"  decode-noise reference (ck0 key-right answers vs base greedy), {how}: n {int(REF[how].sum())} same {v[0]:.3f} other sample {v[1]:.3f} neither {v[2]:.4f}")
P(f"  never-sampled (contain) right answers: {extra['n_f0']} ({extra['n_f0']/extra['n_right']:.4f}); on new questions sharing an exact alias set with a training "
  f"question: {extra['n_f0_alias']} ({extra['n_f0_alias']/max(1,extra['n_f0']):.3f}) vs {extra['n_alias_new']/extra['n_right']:.3f} of all right answers")
import pandas as pd
T = pd.DataFrame(rows); T.to_csv(f"{OUT}/v10_rows.csv", index=False)
for lab, M in (("0.05 <= a <= 0.95", (T.a >= 0.05) & (T.a <= 0.95)), ("all runs", T.a >= 0)):
    X = T[M]
    P(f"\n(B) {lab}: {len(X)} runs; mean acc|ans - base {(X.acc - X.base).mean():+.4f} = sel {X.sel.mean():+.4f} + chg {X.chg.mean():+.4f}; "
      f"selection share {X.sel.sum() / (X.sel.sum() + X.chg.sum()):.3f}; runs with |chg| < 0.03 {np.mean(np.abs(X.chg) < 0.03):.2f}; chg < 0 in {np.mean(X.chg < 0):.2f}")
    P(f"      alternative baseline (untrained acc among ITS answered): gain {(X.acc - X.base_unt).mean():+.4f} = sel' {X.sel_alt.mean():+.4f} + chg {X.chg.mean():+.4f}; "
      f"selection share {X.sel_alt.sum() / (X.sel_alt.sum() + X.chg.sum()):.3f}")
    for (st, ds), y in X.groupby(["st", "ds"]):
        P(f"      {st:6s} {ds}: runs {len(y):3d} sel {y.sel.mean():+.3f} chg {y.chg.mean():+.4f} [run 5-95%: {np.percentile(y.chg, 5):+.3f}, {np.percentile(y.chg, 95):+.3f}] sel' {y.sel_alt.mean():+.3f}")
open(f"{OUT}/v10_knowledge.txt", "w").write("\n".join(L) + "\n")
