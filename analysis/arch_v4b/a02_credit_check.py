"""(arch_v4b) Independent recount of the main-student IDK-credit runs: new-question (held-out, duplicate-free) abstention of every
qwen7 FK20 judgeQt50 run at credit 0.5 / 0.7 / 0.85, from eval_rl_ck120.jsonl (greedy rows, s = -1), with the project's answer parser
(rc.sa) and IDK regex. Also the small-student FK20 / FK20D (Dr. GRPO) gold-key credit sweeps, abstention on held-out TriviaQA.
Output: a02_credit_check.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, re, glob, json, pickle, numpy as np
sys.path.insert(0, (_RX + "/common"))
from rc import setup, sa, IDK, RL
OUT = open((_RA + "/arch_v4b/a02_credit_check.txt"), "w")
def pr(*a):
    s = " ".join(str(x) for x in a); print(s); OUT.write(s + "\n")
HELD = {}
for ds, dsn in (("h1", "tqa"), ("pq", "pqa")):
    _, _, H = setup(dsn); HELD[ds] = set(H)
def abst(run):
    f = f"{RL}/{run}/eval_rl_ck120.jsonl"
    if not os.path.exists(f): return None
    ds = run.split("_")[-1]; H = HELD.get(ds)
    if H is None: return None
    n = k = 0
    for r in map(json.loads, open(f)):
        if r["s"] >= 0 or r["idx"] not in H: continue
        x = sa(r); n += 1; k += bool(x and IDK.match(x))
    return k / n if n else None
pr("== qwen7 FK20 judgeQt50: new-question abstention by credit")
for ds in ("h1", "pq"):
    for c in ("idkc05", "idkc07", "idkc085"):
        runs = sorted(glob.glob(f"{RL}/qwen7_FK20judgeQt50_{c}_s*_{ds}"))
        vals = [(os.path.basename(r), abst(os.path.basename(r))) for r in runs]
        pr(f"  {ds} {c}: " + "  ".join(f"{n.split('_')[-2]} {v:.3f}" for n, v in vals if v is not None))
pr("\n== small students, TriviaQA, gold key (sim00fn00), KL 0.2: FK20 (std) vs FK20D (Dr. GRPO), new-question abstention by credit")
for st in ("phi4m", "smol3"):
    for tag in ("FK20", "FK20D"):
        runs = sorted(glob.glob(f"{RL}/{st}_{tag}sim00fn00_idkc*_s*_h1"))
        by = {}
        for r in runs:
            m = re.search(r"_idkc(\d+)_s(\d+)_h1$", os.path.basename(r))
            if not m: continue
            v = abst(os.path.basename(r))
            if v is not None: by.setdefault(m.group(1), []).append(v)
        if by: pr(f"  {st} {tag:5s}: " + "  ".join(f"c0.{k[1:] if k.startswith('0') else k}: " + ",".join(f"{x:.2f}" for x in v) for k, v in sorted(by.items())))
