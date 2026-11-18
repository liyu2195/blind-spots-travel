"""Loaders for the three SMALL students (Phi-4-mini, SmolLM3-3B, Granite) on TriviaQA held-out questions.
Untrained greedy answer = ck0 greedy row of a strict-format run of the same student (the answers the 'filter' judge files scored);
confidence = share of the 32 untrained samples (base file, IDK prompt) agreeing with that answer (rc.same);
judge verdicts on the untrained greedy answer: P(Yes) of Qwen3.5-9B / 4B / 2B, Gemma-12B, Phi-4, SmolLM3 as judges (st/res/run1/filter),
Sonnet / Haiku / Opus levels (frontier agents; pairs_v1.json + out_<model>_NNN.tsv; YES = level >= 4).
Run outcome codes as bm.py: 0 IDK, 1 right (loose key), 2 wrong, 3 malformed (strict reading rc.sa)."""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, json, re, glob, pickle, numpy as np
from collections import defaultdict
sys.path.insert(0, (_RX + "/common"))
from rc import setup, sa, IDK, load, same, C, RL, BASE
HERE = (_RX + "/story/B_refusal_map"); FR = (_RX + "/frontier"); FILT = (_RR + "/run1/filter")
CK0 = {"phi4m": "phi4m_Fsim00fn00_idkc05_s0_h1", "smol3": "smol3_Fsim00fn00_idkc05_s0_h1"}
def frontier_levels():
    pairs = json.load(open(f"{FR}/pairs_v1.json")); byid = {p["id"]: (p["idx"], p["norm"]) for p in pairs}; out = {}
    for model in ("sonnet", "haiku", "opus"):
        d = {}
        for f in sorted(glob.glob(f"{FR}/out_{model}_*.tsv")):
            for l in open(f):
                p = l.rstrip("\n").split("\t")
                if len(p) >= 2 and p[0].strip().isdigit() and p[1].strip().isdigit() and int(p[0]) in byid: d[byid[int(p[0])]] = int(p[1])
        out[model] = d
    return out
def build_small(st, force=False):
    path = f"{HERE}/cache_small_{st}_h1.pkl"
    if os.path.exists(path) and not force: return pickle.load(open(path, "rb"))
    probs, H1, HELD = setup("tqa"); hs = set(HELD); n = len(HELD)
    g0, _ = load(CK0[st], "ck0"); gr = {i: sa(g0[i]) for i in HELD}; gr = {i: (x if (x and not IDK.match(x)) else "") for i, x in gr.items()}
    samp = defaultdict(list)
    for r in map(json.loads, open(f"{BASE}/{st}_tqa_1500_n32_s1.jsonl")):
        if r["idx"] in hs and r["s"] >= 0:
            x = sa(r); samp[r["idx"]].append(x if (x and not IDK.match(x)) else "")
    Q = dict(idx=np.array(HELD), n=n)
    Q["conf"] = np.array([(sum(same(gr[i], y) for y in samp[i] if y) / 32) if gr[i] else 0.0 for i in HELD])
    Q["gval"] = np.array([bool(gr[i]) for i in HELD]); Q["gans"] = [gr[i] for i in HELD]; Q["question"] = [probs[i]["question"] for i in HELD]
    Q["key"] = np.array([bool(gr[i]) and bool(C.tq_correct(gr[i], set(probs[i]["aliases"]))) for i in HELD])
    Q["kstrict"] = np.array([bool(gr[i]) and bool(C.tq_strict(gr[i], set(probs[i]["aliases"]))) for i in HELD])
    J = {}
    for jn in ("q9b", "q4b", "q2b", "g12b", "phi4m", "smol3"):
        f = f"{FILT}/{jn}__tqa_1500.jsonl"; d = {}
        for r in map(json.loads, open(f)):
            if r["st"] == f"{CK0[st]}__ck0" and r["s"] < 0: d[r["idx"]] = r["py"] / (r["py"] + r["pn"]) if r["py"] + r["pn"] > 1e-3 else 0.5
        J[jn] = np.array([d.get(i, np.nan) for i in HELD])
    FL = frontier_levels()
    for model, d in FL.items(): J[model] = np.array([d.get((i, gr[i]), -1) if gr[i] else -1 for i in HELD], float)
    Q["J"] = J
    son = J["sonnet"]; Q["aud"] = Q["key"] | (son >= 4)
    R = {}
    for d in sorted(glob.glob(f"{RL}/{st}_*_h1")):
        name = os.path.basename(d)
        if not os.path.exists(f"{d}/eval_rl_ck120.jsonl"): continue
        try: gt, _ = load(name, "ck120")
        except Exception: continue
        if not all(i in gt for i in HELD): continue
        o = np.zeros(n, np.int8)
        for k, i in enumerate(HELD):
            x = sa(gt[i])
            if not x: o[k] = 3
            elif IDK.match(x): o[k] = 0
            elif C.tq_correct(x, set(probs[i]["aliases"])): o[k] = 1
            else: o[k] = 2
        R[name] = dict(o=o, a=float((o == 0).mean()), bad=float((o == 3).mean()))
    P = dict(st=st, Q=Q, R=R); pickle.dump(P, open(path, "wb")); return P
if __name__ == "__main__":
    for st in ("phi4m", "smol3"):
        P = build_small(st, force=len(sys.argv) > 1)
        Q = P["Q"]; print(st, "held-out", Q["n"], "valid untrained answers", Q["gval"].sum(), "runs", len(P["R"]))
        for k, v in Q["J"].items(): print("   judge", k, "coverage", np.mean(~np.isnan(v) & (v >= 0)) if k in ("sonnet", "haiku", "opus") else np.mean(~np.isnan(v)))
