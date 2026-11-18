"""(explorer 'general') Lookup of judge verdicts for small-student answers, keyed by (idx, first line of the answer text) --
exactly what st_judge.py shows the judge -- one table per (judge, dataset). Only untrained (ck0) and final-checkpoint rows are
kept. Sources:
  (1) score_rl_<J>.jsonl in small-student run folders (J = Q Qwen3.5-9B, G Gemma-4-12B; T = the run's own training judge,
      Qwen3.5-2B / 4B), joined to the scored eval rows;
  (2) st/res/run1/filter/<jn>__<set>_1500.jsonl (reference-free q9b, g12b, q4b, q2b on listed runs' ck0 / ck120 greedy answers);
  (3) st/res/run1/keyfp/score_<j>_<key|free>.jsonl (REFERENCE-GUIDED q9 / g12 / q4 / q2 and reference-free q2 / q4 on the
      untrained plain-prompt greedy TriviaQA answers of <st>_goldstrict_h1 ck0).
Judge names: Q9, G12, Q4, Q2 (reference-free); Q9ref, G12ref, Q4ref, Q2ref (reference-guided).
Output: g02_lookup_<judge>_<ds>.pkl {(idx, line): mean P(Yes)/(P(Yes)+P(No))}, g02_scores.txt (row counts; agreement of
repeated scores of the same key)."""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import json, os, sys, pickle, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gcommon import RL, HERE, C, pyes

INV = {r["run"]: r for r in json.load(open(f"{HERE}/g00_inventory.json"))}
FILT = (_RR + "/run1/filter"); KF = (_RR + "/run1/keyfp")
log = open(f"{HERE}/g02_scores.txt", "w")


def eval_texts(run, ckn):
    p = f"{RL}/{run}/{ckn}.jsonl"; d = {}
    if os.path.exists(p):
        for l in open(p):
            e = json.loads(l); d[(e["idx"], e["s"])] = e["text"]
    return d


def keep(run, ckn):
    return ckn == "eval_rl_ck0" or ckn == f"eval_rl_ck{INV[run]['final']}"


def build(J, ds):
    S, N, MN, MX = {}, {}, {}, {}
    def add(k, p):
        if k in S:
            S[k] += p; N[k] += 1; MN[k] = min(MN[k], p); MX[k] = max(MX[k], p)
        else:
            S[k] = p; N[k] = 1; MN[k] = p; MX[k] = p
    if J in ("Q9", "G12", "Q2", "Q4"):
        for run, r in sorted(INV.items()):
            if r["ds"] != ds: continue
            for f in r["scores"]:
                Jn = {"Q": "Q9", "G": "G12"}.get(f) or (r.get("jm") if f == "T" else None)
                if Jn != J: continue
                byck = {}
                for l in open(f"{RL}/{run}/score_rl_{f}.jsonl"):
                    x = json.loads(l)
                    if keep(run, x["st"]): byck.setdefault(x["st"], []).append(x)
                n = 0
                for ckn, rows in byck.items():
                    T = eval_texts(run, ckn)
                    for x in rows:
                        t = T.get((x["idx"], x["s"]))
                        if t is not None: add((x["idx"], C.tq_line(t)), pyes(x)); n += 1
                print(f"(1) {J} {ds} {run} score_rl_{f}: kept {n} rows ({sorted(byck)})", file=log)
        jn = {"Q9": "q9b", "G12": "g12b", "Q4": "q4b", "Q2": "q2b"}[J]
        byst = {}
        for l in open(f"{FILT}/{jn}__{ds}_1500.jsonl"):
            x = json.loads(l); byst.setdefault(x["st"], []).append(x)
        for stn, rows in byst.items():
            run, ck = stn.split("__"); T = eval_texts(run, "eval_rl_" + ck); n = 0
            for x in rows:
                t = T.get((x["idx"], x["s"]))
                if t is not None: add((x["idx"], C.tq_line(t)), pyes(x)); n += 1
            print(f"(2) {J} {ds} filter {stn}: {n} rows", file=log)
    fk = {"Q9ref": "q9_key", "G12ref": "g12_key", "Q2ref": "q2_key", "Q4ref": "q4_key", "Q2": "q2_free", "Q4": "q4_free"}.get(J)
    if fk and ds == "tqa":
        rows = [json.loads(l) for l in open(f"{KF}/score_{fk}.jsonl")]; n = 0
        for stn in sorted({x["st"] for x in rows}):
            T = eval_texts(f"{stn}_goldstrict_h1", "eval_rl_ck0")
            for x in rows:
                if x["st"] != stn: continue
                t = T.get((x["idx"], x["s"]))
                if t is not None: add((x["idx"], C.tq_line(t)), pyes(x)); n += 1
        print(f"(3) {J} {ds} keyfp {fk}: {n} rows", file=log)
    if not S: return
    out = {k: S[k] / N[k] for k in S}
    rep = [k for k in S if N[k] > 1]; spread = sorted(MX[k] - MN[k] for k in rep)
    flips = sum((MN[k] > 0.5) != (MX[k] > 0.5) for k in rep)
    print(f"JUDGE {J:6s} {ds}: {len(out)} distinct keys; repeated {len(rep)}; median spread {spread[len(spread) // 2] if spread else 0:.4f}; "
          f"95th pct spread {spread[int(0.95 * len(spread))] if spread else 0:.4f}; keys whose repeats straddle 0.5: {flips / max(1, len(rep)):.4f}", file=log)
    pickle.dump(out, open(f"{HERE}/g02_lookup_{J}_{ds}.pkl", "wb")); log.flush()


for J in ("Q9", "G12", "Q4", "Q2", "Q9ref", "G12ref", "Q4ref", "Q2ref"):
    for ds in ("tqa", "pqa"):
        build(J, ds); gc.collect()
log.close()
print("".join(l for l in open(f"{HERE}/g02_scores.txt") if l.startswith("JUDGE")))
