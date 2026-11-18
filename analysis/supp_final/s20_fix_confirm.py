"""(supp_final) s20_fix_confirm: FROZEN scorer for E14: the student's own vote at ONE fixed setting in all six pairs.
Arm frzVOTE20: the key of a training question is the student's majority answer among its 32 untrained samples if at least 20 samples give it, else
only "I don't know" is paid. No labels, no judge. Confirmation runs: seeds 0-2 in five pairs; seeds 3-5 for Llama on PopQA (seeds 0-2 there existed
before the plan and are shown separately). Answer key = FK20 sim00fn00, credit 0.5, all seeds (three new seeds for Qwen on NQ).
Reward 1 / 0 / 0.5 on held-out questions; truth = loose key or audit level >= 4 (aud.py). The lines are read only when no key-wrong answer of the
own-vote and answer-key runs lacks an audit verdict (the count is printed). Runs that answer IDK on >= 95% of held-out questions are 'collapsed':
dropped from the main comparison (the paper's convention), counted for P5 and included in the 'all runs' view.
Lines:
  P1  pooled over the six pairs (pairs weighted equally), own vote minus answer key: lower end of the 95% interval above -0.015
      (questions and runs resampled, 4,000 draws, seed 2040).
  P2  every pair's difference >= -0.025, and at least five of six >= -0.015.
  P3  PopQA capitals probe: mean share of known former-state capitals refused by the three confirmation runs <= 0.20, for each student.
  P4  fresh PopQA questions (key truth): own vote minus answer key >= -0.05 and own vote minus the Qwen judge at its default cut >= +0.08, each student.
  P5  none of the 18 confirmation runs collapsed.
Reported without a line: differences to the Qwen judge (default, 50%, 35% cut), Sonnet and the judge shown the reference; abstention; wrong answers;
reward minus the untrained student's own-vote rule at the same abstention; the pooled difference with collapsed runs included; Llama PopQA with all
six seeds. MOCK=1 (test before the runs exist): uses the existing own-vote arm nearest k = 20 in each pair with all its seeds.
Usage: python3 -I -B s20_fix_confirm.py   Output: s20_fix_confirm_out.txt, s20_fix_confirm.json"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, json, pickle, re, glob, numpy as np
from collections import defaultdict
sys.path.insert(0, (_RA + "/stats")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s00_common import *
import aud as audlib
MOCK = os.environ.get("MOCK") == "1"; ARM = "frzVOTE20"; NB = 4000; rng = np.random.default_rng(2040); out = [__doc__]; JS = dict(mock=MOCK, cells={})
if MOCK: out.append("\n*** MOCK RUN: existing arms stand in for frzVOTE20; nothing below is a result of E14 ***")
P2D = (_RR + "/run1/adeval/probe2"); FR = (_RR + "/run1/adeval/fresh")
CONF = lambda st, ds: (3, 4, 5) if (st, ds) == ("llama8", "pq") else (0, 1, 2)
POOL = []; POOLALL = []; ncol = nconf = 0; miss_tot = 0; cellok = []
for st in ("qwen7", "llama8"):
    for ds, dsn in (("pq", "pqa"), ("h1", "tqa"), ("nq", "nqo")):
        Bp = pickle.load(open(f"{CACHE}/base_{st}_{ds}.pkl", "rb")); R = pickle.load(open(f"{CACHE}/runs_{st}_{ds}.pkl", "rb")); LV = audlib.levels(dsn)
        new = np.asarray(Bp["new"], bool); ii = np.where(new)[0]; n = len(ii); g0 = lambda a: np.asarray(a)[:, 0]
        ok = g0(Bp["valid"]).astype(bool) & ~g0(Bp["idk"]).astype(bool); right = ok & (g0(Bp["kl"]).astype(bool) | (g0(Bp["son"]) >= 4))
        cnt = np.rint(np.asarray(Bp["conf"], float) * 32).astype(int); score = np.where(ok, cnt, -1)[ii]; rt = right[ii]; vals = np.unique(score); nonans = float((score == -1).mean())
        def vote_at(a):
            rem = a * n; tot = 0.0
            for v in vals:
                m = score == v; sz = m.sum()
                if rem >= sz: rem -= sz; continue
                tot += (1 - rem / sz) * rt[m].sum(); rem = 0
            return tot / n + 0.5 * a
        arm = ARM; seeds = CONF(st, ds)
        if MOCK:
            cand = sorted({r["g"] for r in R.values() if r["tag"] == "FK20" and r["c"] == "05" and re.fullmatch(r"frzVOTE\d+", r["g"])}, key=lambda g: abs(int(g[7:]) - 20))
            arm = cand[0]; seeds = tuple(range(10))
        G = defaultdict(list); COL = defaultdict(int); MISS = defaultdict(int); EXTRA = []
        for name, r in sorted(R.items()):
            if r["tag"] != "FK20" or r["c"] != "05": continue
            g = r["g"]; o = np.asarray(r["o"]); idk = (o == 0); collapsed = idk[new].mean() >= 0.95
            rr = np.array([(o[i] == 1) or (o[i] == 2 and LV.get((i, r["f"][i]), 0) >= 4) for i in ii]); vec = rr.astype(float) + 0.5 * idk[ii]
            rec = dict(name=name, vec=vec, idk=idk[ii], wrong=float(((o[ii] == 2) | (o[ii] == 3)).mean() - (rr & (o[ii] == 2)).mean()), collapsed=bool(collapsed), seed=r["seed"])
            if g == arm and r["seed"] not in seeds: EXTRA.append(rec); continue
            if g in (arm, "sim00fn00"): MISS[g] += sum(1 for i in ii if o[i] == 2 and (i, r["f"][i]) not in LV)
            if collapsed: COL[g] += 1
            G[g].append(rec)
        live = lambda g: [x for x in G[g] if not x["collapsed"]]; cell = f"{st} {dsn}"; J = JS["cells"][cell] = dict(arm=arm)
        V = live(arm); K = live("sim00fn00"); nconf += len(G[arm]); ncol += COL[arm]; miss_tot += MISS[arm] + MISS["sim00fn00"]
        out.append(f"\n==================== {cell}: held-out {n}; own-vote arm {arm}, confirmation seeds {seeds if not MOCK else 'all'}: runs {len(G[arm])} (collapsed {COL[arm]}); "
                   f"answer key runs {len(G['sim00fn00'])} (collapsed {COL['sim00fn00']}); key-wrong answers without an audit verdict: own vote {MISS[arm]}, answer key {MISS['sim00fn00']}")
        if not V or not K: out.append("   runs missing"); cellok.append(False); continue
        def boot(A, Bm):
            A = np.array([x["vec"] for x in A]); Bm = np.array([x["vec"] for x in Bm]); d = np.empty(NB)
            for t in range(NB):
                q = rng.integers(0, n, n); d[t] = A[rng.integers(0, len(A), len(A))][:, q].mean() - Bm[rng.integers(0, len(Bm), len(Bm))][:, q].mean()
            return float(A.mean() - Bm.mean()), d
        m, d = boot(V, K); lo, hi = np.percentile(d, [2.5, 97.5]); POOL.append((cell, m, d)); cellok.append(True)
        ma, da = boot(G[arm], G["sim00fn00"]); POOLALL.append((cell, ma, da))
        sel = [x["vec"].mean() - vote_at(x["idk"].mean()) for x in V if x["idk"].mean() >= nonans - 1e-9]
        J.update(vote=float(np.mean([x["vec"].mean() for x in V])), key=float(np.mean([x["vec"].mean() for x in K])), diff=m, lo=float(lo), hi=float(hi), runs_vote=len(V), runs_key=len(K),
                 abst=float(np.mean([x["idk"].mean() for x in V])), wrong=float(np.mean([x["wrong"] for x in V])), diff_all=ma, collapsed_vote=COL[arm], collapsed_key=COL["sim00fn00"],
                 missing_audit=(MISS[arm], MISS["sim00fn00"]), sel=float(np.mean(sel)) if sel else None)
        out.append(f"   own vote {J['vote']:.3f} (runs " + ", ".join(f"s{x['seed']} {x['vec'].mean():.3f}" for x in V) + f"; abstention {J['abst']:.3f}, wrong answers {J['wrong']:.3f}) | answer key {J['key']:.3f} (runs "
                   + ", ".join(f"s{x['seed']} {x['vec'].mean():.3f}" for x in K) + f"; abstention {np.mean([x['idk'].mean() for x in K]):.3f}, wrong {np.mean([x['wrong'] for x in K]):.3f})")
        out.append(f"   own vote minus answer key: {m:+.4f} [{lo:+.4f}, {hi:+.4f}] | with collapsed runs included: {ma:+.4f} | reward minus the own-vote rule at the same abstention: "
                   + (f"{np.mean(sel):+.4f}" if sel else "not defined"))
        row = []
        for b in ("judgeQt50", "judgeQa50", "judgeQa35", "apiS55", "judgeQRt50"):
            if live(b): mb, db = boot(V, live(b)); row.append(f"{b} {mb:+.3f} [{np.percentile(db, 2.5):+.3f}, {np.percentile(db, 97.5):+.3f}] ({len(live(b))})"); J[f"minus_{b}"] = (mb, float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5)))
        out.append("   own vote minus: " + "; ".join(row))
        if EXTRA:
            E = [x for x in EXTRA if not x["collapsed"]]
            if E:
                me, de = boot(E, K); J["extra"] = (me, len(E)); out.append(f"   {arm} runs outside the confirmation seeds ({', '.join('s' + str(x['seed']) for x in E)}): {np.mean([x['vec'].mean() for x in E]):.3f}; minus answer key {me:+.4f} "
                                                                           f"[{np.percentile(de, 2.5):+.4f}, {np.percentile(de, 97.5):+.4f}]; all seeds together minus answer key {boot(V + E, K)[0]:+.4f}")
        if ds == "pq":
            probe = {p["idx"]: p for p in map(json.loads, open(f"{DATA}/pqq_1500.jsonl"))}; ids = sorted(probe); pc = np.array([probe[i]["cls"] for i in ids]); gk = {}
            for r in map(json.loads, open(f"{BASE}/{st}_pqq_1500_n32_s1.jsonl")):
                if r["s"] < 0: x = ans(r); gk[r["idx"]] = bool(x) and not IDKF.match(x) and bool(C.tq_correct(x, set(probe[r["idx"]]["aliases"])))
            kn = np.array([gk[i] for i in ids]); mF = kn & (pc == "F"); mP = kn & (pc == "P")
            def probe_of(name):
                f = f"{P2D}/{name}.jsonl"
                if not os.path.exists(f): return None
                o = {r["idx"]: bool(ans(r)) and bool(IDKF.match(ans(r))) for r in map(json.loads, open(f)) if r["s"] < 0}; y = np.array([o[i] for i in ids], float); return float(y[mF].mean()), float(y[mP].mean())
            pv = [(x["seed"], probe_of(x["name"])) for x in V]; have = [p for _, p in pv if p is not None]
            J["P3"] = dict(runs=len(have), of=len(V), F=float(np.mean([p[0] for p in have])) if have else None, P=float(np.mean([p[1] for p in have])) if have else None, each=[(s, p) for s, p in pv])
            out.append(f"   P3 capitals probe (known former-state {int(mF.sum())}, present-day {int(mP.sum())}): confirmation runs with probe answers {len(have)} of {len(V)}; former-state refused "
                       + (f"{np.mean([p[0] for p in have]):.3f} (" + ", ".join(f"s{s} {p[0]:.2f}" for s, p in pv if p) + f"); present-day {np.mean([p[1] for p in have]):.3f}" if have else "MISSING"))
            ex = [(x["seed"], probe_of(x["name"])) for x in EXTRA if not x["collapsed"]]; ex = [(s, p) for s, p in ex if p]
            if ex: out.append(f"      runs outside the confirmation seeds: former-state refused {np.mean([p[0] for _, p in ex]):.3f} (" + ", ".join(f"s{s} {p[0]:.2f}" for s, p in ex) + ")"
                              + (f"; all seeds together {np.mean([p[0] for p in have] + [p[0] for _, p in ex]):.3f}" if have else ""))
            for b in ("sim00fn00", "judgeQt50", "judgeQacc", "judgeQa50", "judgeQa35", "apiS55"):
                pb = [probe_of(x["name"]) for x in live(b)]; pb = [p for p in pb if p]
                if pb: out.append(f"      {b:10s}: former-state refused {np.mean([p[0] for p in pb]):.3f} ({len(pb)} runs)")
            fresh = [json.loads(l) for l in open(f"{DATA}/pqx_fresh.jsonl")]; NF = len(fresh)
            def fresh_of(name):
                f = f"{FR}/{name}.jsonl"
                if not os.path.exists(f): return None
                v = np.zeros(NF)
                for q in map(json.loads, open(f)):
                    if q["s"] < 0: x = ans(q); v[q["idx"]] = 0.5 if (x and IDKF.match(x)) else float(bool(x) and bool(C.tq_correct(x, set(fresh[q["idx"]]["aliases"]))))
                return v
            fv = [v for v in (fresh_of(x["name"]) for x in V) if v is not None]; fk = [v for v in (fresh_of(x["name"]) for x in K) if v is not None]; fj = [v for v in (fresh_of(x["name"]) for x in live("judgeQt50")) if v is not None]
            def fboot(A, Bm):
                A = np.array(A); Bm = np.array(Bm); d = np.empty(NB)
                for t in range(NB):
                    q = rng.integers(0, NF, NF); d[t] = A[rng.integers(0, len(A), len(A))][:, q].mean() - Bm[rng.integers(0, len(Bm), len(Bm))][:, q].mean()
                return float(A.mean() - Bm.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
            if fv and fk and fj:
                a1 = fboot(fv, fk); a2 = fboot(fv, fj); J["P4"] = dict(runs=len(fv), of=len(V), vs_key=a1, vs_judge=a2, vote=float(np.mean([v.mean() for v in fv])))
                out.append(f"   P4 fresh questions ({NF}, key truth): confirmation runs with fresh answers {len(fv)} of {len(V)}: own vote {np.mean([v.mean() for v in fv]):.3f}; minus answer key {a1[0]:+.4f} [{a1[1]:+.4f}, {a1[2]:+.4f}] "
                           f"({len(fk)} key runs); minus Qwen judge default {a2[0]:+.4f} [{a2[1]:+.4f}, {a2[2]:+.4f}] ({len(fj)})")
            else: J["P4"] = None; out.append("   P4 fresh questions: MISSING answers")
out.append("\n==================== LINES")
if len(POOL) == 6:
    d = np.mean([x[2] for x in POOL], axis=0); m = float(np.mean([x[1] for x in POOL])); lo, hi = np.percentile(d, [2.5, 97.5]); cd = [x[1] for x in POOL]
    da = np.mean([x[2] for x in POOLALL], axis=0); ma = float(np.mean([x[1] for x in POOLALL]))
    p1 = lo > -0.015; p2 = min(cd) >= -0.025 and sum(c >= -0.015 for c in cd) >= 5
    out.append(f"   P1 pooled own vote minus answer key {m:+.4f} [{lo:+.4f}, {hi:+.4f}] (pairs: " + "; ".join(f"{c} {x:+.4f}" for c, x, _ in POOL) + f"): lower end above -0.015: {'INSIDE' if p1 else 'OUTSIDE'}")
    out.append(f"      with collapsed runs included: {ma:+.4f} [{np.percentile(da, 2.5):+.4f}, {np.percentile(da, 97.5):+.4f}]")
    out.append(f"   P2 lowest pair {min(cd):+.4f} (>= -0.025), pairs at or above -0.015: {sum(c >= -0.015 for c in cd)} of 6 (>= 5): {'INSIDE' if p2 else 'OUTSIDE'}")
    JS["P1"] = dict(diff=m, lo=float(lo), hi=float(hi), inside=bool(p1), all_runs=ma); JS["P2"] = dict(min=float(min(cd)), n_ok=int(sum(c >= -0.015 for c in cd)), inside=bool(p2))
else: out.append(f"   P1 / P2: only {len(POOL)} of 6 pairs have runs: NOT READ")
p3 = [JS["cells"][c].get("P3") for c in ("qwen7 pqa", "llama8 pqa") if c in JS["cells"]]
if len(p3) == 2 and all(p and p["runs"] == p["of"] and p["runs"] > 0 for p in p3):
    ok3 = all(p["F"] <= 0.20 for p in p3); JS["P3"] = dict(F=[p["F"] for p in p3], inside=bool(ok3)); out.append(f"   P3 former-state capitals refused: Qwen {p3[0]['F']:.3f}, Llama {p3[1]['F']:.3f} (each <= 0.20): {'INSIDE' if ok3 else 'OUTSIDE'}")
else: out.append("   P3: probe answers missing for some confirmation runs: NOT READ")
p4 = [JS["cells"][c].get("P4") for c in ("qwen7 pqa", "llama8 pqa") if c in JS["cells"]]
if len(p4) == 2 and all(p and p["runs"] == p["of"] for p in p4):
    ok4 = all(p["vs_key"][0] >= -0.05 and p["vs_judge"][0] >= 0.08 for p in p4); JS["P4"] = dict(vs_key=[p["vs_key"][0] for p in p4], vs_judge=[p["vs_judge"][0] for p in p4], inside=bool(ok4))
    out.append(f"   P4 fresh questions, own vote minus key: Qwen {p4[0]['vs_key'][0]:+.4f}, Llama {p4[1]['vs_key'][0]:+.4f} (each >= -0.05); minus default judge: {p4[0]['vs_judge'][0]:+.4f}, {p4[1]['vs_judge'][0]:+.4f} (each >= +0.08): {'INSIDE' if ok4 else 'OUTSIDE'}")
else: out.append("   P4: fresh answers missing for some confirmation runs: NOT READ")
out.append(f"   P5 collapsed confirmation runs: {ncol} of {nconf} (expected 18 runs): {'INSIDE' if (ncol == 0 and nconf == 18) else ('OUTSIDE' if ncol else 'NOT READ (runs missing)')}")
JS["P5"] = dict(collapsed=ncol, runs=nconf)
out.append(f"   audit: key-wrong answers without a verdict in the own-vote and answer-key runs: {miss_tot}" + ("" if miss_tot == 0 else "  -> AUDIT INCOMPLETE: run the audit, then read P1 / P2 again"))
JS["missing_audit"] = miss_tot
H_ = os.path.dirname(os.path.abspath(__file__)); sfx = "_mock" if MOCK else ""
open(os.path.join(H_, f"s20_fix_confirm{sfx}_out.txt"), "w").write("\n".join(out) + "\n"); json.dump(JS, open(os.path.join(H_, f"s20_fix_confirm{sfx}.json"), "w"), indent=1, default=str)
print("\n".join(out[1:]))
