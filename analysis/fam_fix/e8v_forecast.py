"""(fam_fix) e8v_forecast (offline, before any E8 run): the own-vote rule applied to the never-trained capitals probe. For each probe question the size n
of the largest cluster of the student's 32 untrained samples (rc.same); a model that learned 'abstain where fewer than k of my samples agree' would
refuse known answers with n < k. Share of known former-state (F) and present-day (P) capitals with n < k, for k = 20 (new), and for the k of the
existing own-vote runs with their observed refusal. Output: e8v_forecast_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, json, numpy as np
from collections import defaultdict
sys.path.insert(0, (_RX + "/common"))
from rc import *
out = [__doc__]
probs = {p["idx"]: p for p in map(json.loads, open(f"{DATA}/pqq_1500.jsonl"))}; ids = sorted(probs)
OBS = {("llama8", 29): "observed R_F 0.35 / 0.33 at abstention 0.45 / 0.37", ("llama8", 32): "observed R_F 0.58 / 0.60 at abstention 0.50 / 0.50",
       ("qwen7", 21): "observed R_F 0.13 at abstention 0.31", ("qwen7", 32): "observed R_F 0.11 / 0.37 at abstention 0.52 / 0.58"}
for st in ("qwen7", "llama8"):
    rows = defaultdict(list); g = {}
    for r in map(json.loads, open(f"{BASE}/{st}_pqq_1500_n32_s1.jsonl")):
        x = sa(r)
        if r["s"] < 0: g[r["idx"]] = bool(x) and not IDK.match(x) and bool(C.tq_correct(x, set(probs[r["idx"]]["aliases"])))
        elif x and not IDK.match(x): rows[r["idx"]].append(x)
    nv = {}
    for i in ids:
        cl = []
        for x in rows.get(i, []):
            for c in cl:
                if same(x, c[0]): c[1].append(x); break
            else: cl.append([x, [x]])
        nv[i] = max((len(c[1]) for c in cl), default=0)
    F = [i for i in ids if probs[i]["cls"] == "F" and g[i]]; P = [i for i in ids if probs[i]["cls"] == "P" and g[i]]
    for k in (16, 20, 21, 24, 29, 32):
        f = np.mean([nv[i] < k for i in F]); p = np.mean([nv[i] < k for i in P])
        out.append(f"  {st} own-vote k {k:2d}: known F with n < k {f:.3f}, known P {p:.3f}, gap {f - p:+.3f}   {OBS.get((st, k), '')}")
open("e8v_forecast_out.txt", "w").write("\n".join(out) + "\n"); print("\n".join(out[1:]))
