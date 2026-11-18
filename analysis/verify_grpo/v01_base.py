"""(verify_grpo) Per-question table of the UNTRAINED student's 32 sampled replies (T 0.8, top-p 0.95, the training sampler) and its greedy reply,
re-implementing the trainer's reward-side rules (st_grpo_b.py and st_grpo_c.py, st_common):
  strict format (fmt 1): valid iff the reply ended with EOS ('fin' == 'stop'), no newline in text.strip(), no 'think>'; answer x = parse_tq(text) if valid else '';
                         IDK iff x and IDK_FULL.fullmatch(x)
  old format  (fmt 0): every reply valid, x = parse_tq(text), IDK iff C.is_idk(x)
  key-right: C.tq_strict(x, aliases of the trainer's data file) for valid non-IDK replies.
Also: judge P(Yes) = py / (py + pn) (0.5 if py + pn <= 1e-3, as the trainer), word counts, the modal non-IDK answer count.
home = ids in <set>_h1.jsonl (the training file); new = the other ids, minus questions whose lower-cased text equals a training question's text.
Usage: nice -n 10 python3 -I v01_base.py <st> <ds>   (ds in h1 pq nq) -> cache/vb_<st>_<ds>.npz, cache/vb_<st>_<ds>_x.json"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, re, json, numpy as np
from collections import Counter
sys.path.insert(0, _RC)
import st_common as C
st, ds = sys.argv[1], sys.argv[2]
SET = {"h1": "tqa", "pq": "pqa", "nq": "nqo"}[ds]
DATA = _RD; BASE = (_RR + "/run1/base"); HERE = (_RA + "/verify_grpo")
IDK_FULL = re.compile(r"^(i don t know|i do not know|i dont know|idk|i m not sure|not sure|i cannot answer|cannot answer|no idea|i am not sure|unknown)$")
rd = lambda p: [json.loads(l) for l in open(p)]
Q = rd(f"{DATA}/{SET}_1500.jsonl"); ids = [q["idx"] for q in Q]; pos = {i: k for k, i in enumerate(ids)}; n = len(ids)
TR = {q["idx"]: q for q in rd(f"{DATA}/{SET}_h1.jsonl")}
# aliases the trainer saw (training file) vs the 1500 file
mism = sum(1 for q in Q if q["idx"] in TR and set(TR[q["idx"]]["aliases"]) != set(q["aliases"]))
home = np.array([i in TR for i in ids]); trtext = {TR[i]["question"].strip().lower() for i in TR}
new = np.array([(i not in TR) and (q["question"].strip().lower() not in trtext) for i, q in zip(ids, Q)])
alias = [set(TR[i]["aliases"]) if i in TR else set(q["aliases"]) for i, q in zip(ids, Q)]
f = f"{BASE}/{st}_{SET}_1500_n32_s1.jsonl"
if not os.path.exists(f): f = f"{BASE}/{st}_{SET}_1500_idk_n32_s1.jsonl"
K = 32
x1 = [[""] * K for _ in range(n)]; x0 = [[""] * K for _ in range(n)]
v1 = np.zeros((n, K), bool); i1 = np.zeros((n, K), bool); r1 = np.zeros((n, K), bool); w1 = np.zeros((n, K), np.int16)
i0 = np.zeros((n, K), bool); r0 = np.zeros((n, K), bool); seen = np.zeros((n, K), bool)
gi1 = np.zeros(n, bool); gi0 = np.zeros(n, bool)
for r in rd(f):
    if r["idx"] not in pos: continue
    k = pos[r["idx"]]; t = r.get("text", "")
    ok = r.get("fin") == "stop" and "\n" not in t.strip() and "think>" not in t
    a1 = C.parse_tq(t) if ok else ""; a0 = C.parse_tq(t)
    idk1 = bool(a1) and bool(IDK_FULL.match(a1)); idk0 = C.is_idk(a0)
    if r["s"] < 0:
        gi1[k] = idk1; gi0[k] = idk0; continue
    j = r["s"]; seen[k, j] = True
    x1[k][j] = a1; x0[k][j] = a0; v1[k, j] = bool(a1); i1[k, j] = idk1; i0[k, j] = idk0
    r1[k, j] = bool(a1) and not idk1 and C.tq_strict(a1, alias[k]); r0[k, j] = bool(a0) and not idk0 and C.tq_strict(a0, alias[k])
    w1[k, j] = len(a1.split())
assert seen.all(), "missing samples"
def modal(X, I):
    out = np.zeros(n, np.int16)
    for k in range(n):
        c = Counter(X[k][j] for j in range(K) if not I[k, j])
        out[k] = c.most_common(1)[0][1] if c else 0
    return out
m1 = modal(x1, i1); m0 = modal(x0, i0)
PJ = {}
for J in ("Q", "M", "G", "O", "R", "QS"):
    sf = f"{BASE}/score_{J}_{st}_{SET}_1500_n32_s1.jsonl"
    if not os.path.exists(sf): continue
    a = np.full((n, K), np.nan, np.float32)
    for r in rd(sf):
        if r["idx"] not in pos or r["s"] < 0: continue
        a[pos[r["idx"]], r["s"]] = r["py"] / (r["py"] + r["pn"]) if r["py"] + r["pn"] > 1e-3 else 0.5
    PJ["PJ_" + J] = a
np.savez_compressed(f"{HERE}/cache/vb_{st}_{ds}.npz", ids=np.array(ids), home=home, new=new, v1=v1, i1=i1, r1=r1, w1=w1, i0=i0, r0=r0, m1=m1, m0=m0,
                    gi1=gi1, gi0=gi0, **PJ)
json.dump({"x1": x1, "x0": x0}, open(f"{HERE}/cache/vb_{st}_{ds}_x.json", "w"))
print(f"{st} {ds}: n {n} home {home.sum()} new {new.sum()} | alias mismatch train-file vs 1500-file {mism} | sampled IDK {i1.mean():.3f} (old {i0.mean():.3f}) "
      f"invalid {(~v1).mean():.3f} key-right {r1.mean():.3f} | greedy IDK {gi1.mean():.3f} | judges {sorted(PJ)} | {os.path.basename(f)}")
