"""(grpo) Per-question table of the UNTRAINED student's 32 samples (+ greedy) for one (student, dataset), built from raw files only.
Trainer rules re-implemented (st_grpo_v12.py, --strict_format 1): a reply is valid iff it ended (fin == 'stop') and has no newline and no 'think>';
an invalid reply gets reward 0; IDK = IDK_FULL full match of parse_tq(text) (strict format) -- also stored: the prefix regex C.is_idk (old trainer);
right = C.tq_strict with the question's aliases (the trainer's reward key; h1 aliases equal the 1500-file aliases, checked).
Arrays are (n_questions, 32) in the order of the 1500-question file. Judge P(Yes) = py / (py + pn) (0.5 if both tiny), as in the trainer.
home = ids in <set>_h1.jsonl; new = other ids minus questions whose lower-cased text or qid equals a training question's.
Usage: nice -n 10 python3 -I g01_base.py <st> <ds>    (ds in h1 pq nq) -> cache/base_<st>_<ds>.npz + .json"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, re, json, numpy as np
sys.path.insert(0, _RC)
import st_common as C
st, ds = sys.argv[1], sys.argv[2]
DSN = {"h1": "tqa", "pq": "pqa", "nq": "nqo"}[ds]
DATA = _RD; BASE = (_RR + "/run1/base"); HERE = (_RA + "/grpo")
IDK_FULL = re.compile(r"^(i don t know|i do not know|i dont know|idk|i m not sure|not sure|i cannot answer|cannot answer|no idea|i am not sure|unknown)$")
jl = lambda p: [json.loads(l) for l in open(p)]
allq = jl(f"{DATA}/{DSN}_1500.jsonl"); ids = [p["idx"] for p in allq]; pos = {i: k for k, i in enumerate(ids)}; Q = len(ids)
H = {p["idx"] for p in jl(f"{DATA}/{DSN}_h1.jsonl")}
qt = {p["question"].strip().lower() for p in allq if p["idx"] in H}; qid = {p.get("qid") for p in allq if p["idx"] in H} - {None}
home = np.array([i in H for i in ids])
new = np.array([(i not in H) and (p["question"].strip().lower() not in qt) and not (p.get("qid") is not None and p.get("qid") in qid) for i, p in zip(ids, allq)])
al = [set(p["aliases"]) for p in allq]
f = f"{BASE}/{st}_{DSN}_1500_n32_s1.jsonl"
if not os.path.exists(f): f = f"{BASE}/{st}_{DSN}_1500_idk_n32_s1.jsonl"
S = 32
X = np.full((Q, S), "", dtype=object); GX = np.full(Q, "", dtype=object)
valid = np.zeros((Q, S), bool); idk = np.zeros((Q, S), bool); idk_old = np.zeros((Q, S), bool); right = np.zeros((Q, S), bool); loose = np.zeros((Q, S), bool)
words = np.zeros((Q, S), np.int16); g_valid = np.zeros(Q, bool); g_idk = np.zeros(Q, bool); g_right = np.zeros(Q, bool); g_loose = np.zeros(Q, bool)
have = np.zeros((Q, S), bool)
for r in jl(f):
    k = pos[r["idx"]]; t = r.get("text", "")
    ok = r.get("fin") == "stop" and "\n" not in t.strip() and "think>" not in t
    x = C.parse_tq(t) if ok else ""
    xo = C.parse_tq(t)                      # old trainer: no format rule
    if r["s"] < 0:
        GX[k] = x; g_valid[k] = bool(x); g_idk[k] = bool(x) and bool(IDK_FULL.match(x))
        g_right[k] = bool(x) and C.tq_strict(x, al[k]); g_loose[k] = bool(x) and C.tq_correct(x, al[k]); continue
    j = r["s"]; have[k, j] = True; X[k, j] = x; valid[k, j] = bool(x)
    idk[k, j] = bool(x) and bool(IDK_FULL.match(x)); idk_old[k, j] = C.is_idk(xo)
    right[k, j] = bool(x) and not idk[k, j] and C.tq_strict(x, al[k]); loose[k, j] = bool(x) and not idk[k, j] and C.tq_correct(x, al[k])
    words[k, j] = len(x.split())
assert have.all(), "missing samples"
PJ = {}
for J in ("Q", "M", "G", "O", "R", "QS"):
    sf = f"{BASE}/score_{J}_{st}_{DSN}_1500_n32_s1.jsonl"
    if not os.path.exists(sf): continue
    a = np.full((Q, S), np.nan, np.float32); g = np.full(Q, np.nan, np.float32)
    for r in jl(sf):
        v = r["py"] / (r["py"] + r["pn"]) if r["py"] + r["pn"] > 1e-3 else 0.5
        if r["idx"] not in pos: continue
        if r["s"] < 0: g[pos[r["idx"]]] = v
        else: a[pos[r["idx"]], r["s"]] = v
    PJ[J] = a; PJ[J + "_g"] = g
os.makedirs(f"{HERE}/cache", exist_ok=True)
np.savez_compressed(f"{HERE}/cache/base_{st}_{ds}.npz", ids=np.array(ids), home=home, new=new, valid=valid, idk=idk, idk_old=idk_old, right=right, loose=loose,
                    words=words, g_valid=g_valid, g_idk=g_idk, g_right=g_right, g_loose=g_loose, **{"PJ_" + k: v for k, v in PJ.items()})
json.dump(dict(X=X.tolist(), GX=GX.tolist()), open(f"{HERE}/cache/base_{st}_{ds}_str.json", "w"))
nonidk = (~idk).sum(1)
print(f"{st} {ds}: Q {Q} home {home.sum()} new {new.sum()} | IDK share {idk.mean():.3f} (old regex {idk_old.mean():.3f}) invalid {(~valid).mean():.3f} "
      f"right {right.mean():.3f} | greedy IDK {g_idk.mean():.3f} | judges {sorted(k for k in PJ if not k.endswith('_g'))} | file {os.path.basename(f)}")
