"""(grpo) Run configurations from the raw queue files (training/queues/* and training/queue_*): every non-comment line with
KEY=VALUE pairs that names OUTNAME. The last line wins; conflicting duplicates are counted. Worker defaults (acl_st_87_worker15.sh line 147):
FPK=0 FPU=0 FN=0 ADV=std IDK=0.5 FPMODE=random FORCE=0 STEPS=120 FMT=1 TEMP=0.8 TOPP=0.95 LRATE=5e-5 KLB=0 PROMPT=idk.
Older runs (no queue line) get a config from the run name, flagged _from_name.
Usage: python3 -I g00_configs.py  -> cache/cfg.json"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import os, re, glob, json
from collections import Counter, defaultdict
HERE = (_RA + "/grpo"); RL = (_RR + "/run1/rl")
roots = sorted(glob.glob((_RT + "/queues/*"))) + sorted(glob.glob((_RT + "/queue_*")))
seen = defaultdict(list)
for r in roots:
    files = [r] if os.path.isfile(r) else [os.path.join(dp, f) for dp, _, fs in os.walk(r) for f in sorted(fs)]
    for p in files:
        try: txt = open(p, errors="replace").read()
        except Exception: continue
        for line in txt.splitlines():
            if line.lstrip().startswith("#"): continue
            kv = dict(re.findall(r"\b([A-Z][A-Z_0-9]*)=(\S+)", line))
            if "OUTNAME" in kv: seen[kv["OUTNAME"]].append(kv)
CFG = {}; nconf = 0
for n, lst in seen.items():
    CFG[n] = dict(lst[-1]); nconf += len({json.dumps(k, sort_keys=True) for k in lst}) > 1
runs = sorted(os.listdir(RL))
print("queue lines name", len(CFG), "runs; runs on disk", len(runs), "; with a queue line", sum(r in CFG for r in runs), "; conflicting duplicates", nconf)
print("without a queue line, by student:", Counter(r.split("_")[0] for r in runs if r not in CFG))
print("ADV:", Counter(CFG[r].get("ADV", "std(default)") for r in runs if r in CFG))
print("KLB:", Counter(CFG[r].get("KLB", "0(default)") for r in runs if r in CFG))
print("IDK:", Counter(CFG[r].get("IDK", "0.5(default)") for r in runs if r in CFG))
print("FMT:", Counter(CFG[r].get("FMT", "1(default)") for r in runs if r in CFG))
print("STEPS:", Counter(CFG[r].get("STEPS", "120(default)") for r in runs if r in CFG))
print("REWARD:", Counter(CFG[r].get("REWARD", "synth(default)") for r in runs if r in CFG))
print("other keys:", Counter(k for r in runs if r in CFG for k in CFG[r]).most_common(60))
json.dump(CFG, open(f"{HERE}/cache/cfg.json", "w"))
