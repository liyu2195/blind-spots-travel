"""write the run configs of a new judge family into work/q_p7<tag lower>/ (not uploaded here). Usage: python3 build_nf_queue.py <tag>"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, json, os
TAG = sys.argv[1]; T = json.load(open(f"{_RX}/final/thresholds_{TAG}.json")); out = f"{_RT}/queues/p7{TAG.lower()}"; os.makedirs(out, exist_ok=True)
S = {"qwen7": "STUDENT=Qwen/Qwen2.5-7B-Instruct TAG=qwen7", "llama8": "STUDENT=gs://YOUR_BUCKET/acl_st/models/meta-llama/Llama-3.1-8B-Instruct TAG=llama8"}
DSF = {"pq": ("pqa", "TRAINF=pqa_h1.jsonl EVALF=pqa_1500.jsonl "), "nq": ("nqo", "TRAINF=nqo_h1.jsonl EVALF=nqo_1500.jsonl "), "h1": ("tqa", "")}
rows = [(st, ds, s, "a35") for s in (0, 1) for ds in ("pq", "nq") for st in ("qwen7", "llama8")] + [(st, "h1", 0, "t50") for st in ("qwen7", "llama8")] + [(st, "h1", s, "a35") for s in (0, 1) for st in ("qwen7", "llama8")]
for n, (st, ds, s, kind) in enumerate(rows):
    jt = 0.5 if kind == "t50" else T[f"{st}_{DSF[ds][0]}"]["t_a35"]; name = f"{st}_FK20judge{TAG}{kind}_idkc05_s{s}_{ds}"
    open(f"{out}/000c{n:02d}_{name}", "w").write(f"{S[st]} ADV=std IDK=0.5 FPMODE=random FMT=1 KLB=0.2 SEED={s} MICRO=16 GENCHUNK=32 CAP=21600 REWARD=judge JT={jt} {DSF[ds][1]}OUTNAME={name}\n")
print(len(rows), "configs in", out)
