"""(supp_final) s20b_check_raw (Nov 18): independent check of s20_fix_confirm.py from the raw evaluation files (not the cache): mean held-out reward of the
own-vote k = 20 confirmation runs and of the answer-key runs that did not collapse, per pair and pooled; truth = loose key or pooled audit level >= 4.
Usage: python3 -I -B s20b_check_raw.py > s20b_check_raw_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, json, glob, numpy as np
sys.path.insert(0, (_RA + "/stats")); sys.path.insert(0, (_RA + "/supp_final"))
from s00_common import *
import aud as audlib
diffs = []
for st in ("qwen7", "llama8"):
    for ds, dsn in (("pq", "pqa"), ("h1", "tqa"), ("nq", "nqo")):
        probs, Hm, NEW = questions(dsn); LV = audlib.levels(dsn); seeds = (3, 4, 5) if (st, ds) == ("llama8", "pq") else (0, 1, 2)
        def run(name):
            f = f"{RL}/{name}/eval_rl_ck120.jsonl"
            if not os.path.exists(f): return None
            E = {r["idx"]: ans(r) for r in map(json.loads, open(f)) if r["s"] < 0}; rew = []; idk = []; miss = 0
            for i in NEW:
                x = E[i]; k = bool(x) and bool(IDKF.match(x)); idk.append(k)
                if k: rew.append(0.5); continue
                okk = bool(x) and (bool(C.tq_correct(x, set(probs[i]["aliases"]))) or LV.get((i, x), 0) >= 4); rew.append(1.0 if okk else 0.0)
                if x and not C.tq_correct(x, set(probs[i]["aliases"])) and (i, x) not in LV: miss += 1
            return float(np.mean(rew)), float(np.mean(idk)), miss
        V = [run(f"{st}_FK20frzVOTE20_idkc05_s{s}_{ds}") for s in seeds]; K = [run(os.path.basename(d)) for d in sorted(glob.glob(f"{RL}/{st}_FK20sim00fn00_idkc05_s*_{ds}"))]
        K = [k for k in K if k]; Kl = [k for k in K if k[1] < 0.95]; d = np.mean([v[0] for v in V]) - np.mean([k[0] for k in Kl]); diffs.append(d)
        print(f"{st} {dsn}: own vote {np.mean([v[0] for v in V]):.4f} (" + ", ".join(f"{v[0]:.3f}" for v in V) + f"; abstention {np.mean([v[1] for v in V]):.3f}) | answer key {np.mean([k[0] for k in Kl]):.4f} "
              f"({len(Kl)} of {len(K)} runs not collapsed) | difference {d:+.4f} | answers without a verdict {sum(v[2] for v in V) + sum(k[2] for k in K)}")
print(f"pooled difference {np.mean(diffs):+.4f}; lowest pair {min(diffs):+.4f}")
