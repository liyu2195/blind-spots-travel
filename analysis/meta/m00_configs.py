"""(meta) Collect every queue config line (KEY=VALUE ...) that names an OUTNAME, from work/q_* and bundle/st/queue_* folders.
Writes cache/configs.json: run name -> dict of config keys (last one wins if repeated; also counts duplicates)."""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import os, re, glob, json
from collections import defaultdict
RL = (_RR + "/run1/rl")
roots = sorted(glob.glob((_RT + "/queues/*"))) + sorted(glob.glob((_RT + "/queue_*")))
cfg = {}; dup = defaultdict(int); src = {}
for r in roots:
    if not os.path.isdir(r): continue
    for f in sorted(os.listdir(r)):
        p = os.path.join(r, f)
        if not os.path.isfile(p): continue
        try: txt = open(p).read()
        except Exception: continue
        for line in txt.splitlines():
            kv = dict(re.findall(r"([A-Z_0-9]+)=(\S+)", line))
            if "OUTNAME" in kv:
                n = kv["OUTNAME"]; dup[n] += 1; cfg[n] = kv; src[n] = p
runs = sorted(os.listdir(RL))
have = [r for r in runs if r in cfg]; miss = [r for r in runs if r not in cfg]
print("configs", len(cfg), "runs", len(runs), "with config", len(have), "without", len(miss))
from collections import Counter
print(Counter(r.split("_")[0] for r in miss))
print([m for m in miss if m.startswith(("qwen7", "llama8"))][:60])
json.dump(dict(cfg=cfg, src=src, dup=dup), open((_RA + "/meta/cache/configs.json"), "w"))
