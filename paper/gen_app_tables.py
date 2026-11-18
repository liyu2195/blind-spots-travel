"""Appendix tables new in v13: tabs/vote.tex (reward minus the untrained student's own-vote rule at the same abstention, by grader and pair; s18 JSON)
and tabs/own.tex (own vote at one cut against the answer key; s20 JSON). Usage: python3 -I -B gen_app_tables.py"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import json, re, numpy as np
S = (_RA + "/supp_v11"); S18 = json.load(open(f"{S}/s18_fix_ceiling.json")); S20 = json.load(open(f"{S}/s20_fix_confirm.json")); S21 = json.load(open(f"{S}/s21_cut_and_calibration.json"))
CELLS = ["qwen7 pqa", "qwen7 tqa", "qwen7 nqo", "llama8 pqa", "llama8 tqa", "llama8 nqo"]
ROWS = [("Answer key", r"sim00fn00"), ("Sonnet as judge", r"apiS55"), ("Judge shown the reference", r"judgeQRt50"), ("Student's own vote (all cuts)", r"frzVOTE\d+"),
        ("Judges, default threshold", r"judge[QMGO]t50"), ("Judges, 50\\% or key-matched cut", r"judge[QMGO](a50|acc)"), ("Judges, 35\\% cut", r"judge[QMGO]a35"),
        ("Key with random rejections", r"sim00fn[1-9]\d")]
L = []
for lab, pat in ROWS:
    vals = []; n = 0; allv = []
    for c in CELLS:
        v = [(x[0], x[1]) for g, x in S18[c]["C"].items() if re.fullmatch(pat, g)]
        if not v: vals.append("--"); continue
        m = sum(a * b for a, b in v) / sum(b for _, b in v); n += sum(b for _, b in v); vals.append(f"${100 * m:+.1f}$".replace("-", "-")); allv.append(m)
    L.append(f"{lab} & {n} & " + " & ".join(vals) + f" & ${100 * np.mean(allv):+.1f}$ \\\\")
open("tabs/vote.tex", "w").write("\n".join(L) + "\n"); print("\n".join(L))
L = []
NAME = {"qwen7 pqa": "Qwen, PopQA", "qwen7 tqa": "Qwen, TriviaQA", "qwen7 nqo": "Qwen, NQ-open", "llama8 pqa": "Llama, PopQA", "llama8 tqa": "Llama, TriviaQA", "llama8 nqo": "Llama, NQ-open"}
KS = {}
for a in S21["arms"]:
    if a["k"] == 20: KS[a["cell"]] = 20 - a["dist"]
for c in CELLS:
    v = S20["cells"][c]
    L.append(f"{NAME[c]} & {100 * v['vote']:.1f} & {100 * v['key']:.1f} ({v['runs_key']}) & ${100 * v['diff']:+.1f}$ [${100 * v['lo']:+.1f}$, ${100 * v['hi']:+.1f}$] & {round(100 * v['abst']):d}\\% & {100 * v['wrong']:.0f} & {KS[c]} \\\\")
p = S20["P1"]; L.append("\\midrule"); L.append(f"Pooled & & & ${100 * p['diff']:+.1f}$ [${100 * p['lo']:+.1f}$, ${100 * p['hi']:+.1f}$] & & & \\\\")
open("tabs/own.tex", "w").write("\n".join(L) + "\n"); print("\n".join(L))
