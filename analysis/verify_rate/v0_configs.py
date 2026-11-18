"""(verify_rate) Own parser of the raw queue configs: every line with KEY=VALUE pairs naming OUTNAME, from training/queues/* and
training/queue_*. Records conflicts (same OUTNAME, different values). Writes cfg.json (run -> dict) and prints a summary.
Usage: python3 -I v0_configs.py"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import os, re, glob, json
from collections import defaultdict, Counter
HERE = (_RA + "/verify_rate"); RL = (_RR + "/run1/rl")
roots = sorted(glob.glob((_RT + "/queues/*"))) + sorted(glob.glob((_RT + "/queue_*")))
seen = defaultdict(list)
for r in roots:
    files = [r] if os.path.isfile(r) else [os.path.join(r, f) for f in sorted(os.listdir(r))]
    for p in files:
        if not os.path.isfile(p): continue
        try: txt = open(p, errors="replace").read()
        except Exception: continue
        for line in txt.splitlines():
            if line.lstrip().startswith("#"): continue
            kv = dict(re.findall(r"\b([A-Z][A-Z_0-9]*)=(\S+)", line))
            if "OUTNAME" in kv: seen[kv["OUTNAME"]].append((p, kv))
cfg, conflicts = {}, {}
for n, lst in seen.items():
    cfg[n] = lst[-1][1]
    vals = {json.dumps(k, sort_keys=True) for _, k in lst}
    if len(vals) > 1:
        diff = sorted({k for _, a in lst for k in a if len({b.get(k) for _, b in lst}) > 1})
        conflicts[n] = diff
runs = sorted(os.listdir(RL))
print("config lines for", len(cfg), "names; runs on disk", len(runs), "; runs with a config", sum(r in cfg for r in runs))
print("runs without config by student:", Counter(r.split("_")[0] for r in runs if r not in cfg))
print("names with conflicting duplicate lines:", len(conflicts), "; keys that differ:", Counter(k for v in conflicts.values() for k in v).most_common(12))
print("ADV values:", Counter(c.get("ADV") for c in cfg.values()), "| IDK values:", Counter(c.get("IDK") for c in cfg.values()))
json.dump(dict(cfg=cfg, conflicts=conflicts), open(f"{HERE}/cfg.json", "w"))
