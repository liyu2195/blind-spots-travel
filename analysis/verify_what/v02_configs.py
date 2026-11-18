"""(verify_what) Index the worker queue config files (training/queues/*/*) by OUTNAME; check that each run directory of the large
students has a config and that duplicate configs agree on the fields that matter here (TRAINF, SEED, FPMODE, FPK, FPU, JT, KLB, STEPS,
REWARD). Writes v02_configs.pkl {run: dict of fields} and v02_configs.txt."""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import os, re, glob, pickle, collections, shlex
OUT = (_RA + "/verify_what"); RL = (_RR + "/run1/rl")
KEYS = ("TRAINF", "SEED", "FPMODE", "FPK", "FPU", "FN", "JT", "KLB", "STEPS", "REWARD", "IDK", "JUDGE", "JMODEL", "ADV")
cfg = collections.defaultdict(list)
for f in glob.glob((_RT + "/queues/*/*")):
    if not os.path.isfile(f): continue
    try: txt = open(f).read()
    except Exception: continue
    if "OUTNAME=" not in txt: continue
    d = {}
    try:
        for tok in shlex.split(txt.replace("\n", " ")):
            if "=" in tok: k, v = tok.split("=", 1); d[k] = v
    except ValueError: continue
    if "OUTNAME" in d: cfg[d["OUTNAME"]].append((f, d))
L = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); L.append(s)
runs = sorted(os.path.basename(d) for d in glob.glob(f"{RL}/*") if re.match(r"^(qwen7|llama8)_", os.path.basename(d)))
have = [r for r in runs if r in cfg]; miss = [r for r in runs if r not in cfg]
P(f"large-student run dirs {len(runs)}; with a config {len(have)}; without {len(miss)}; e.g. {miss[:5]}")
incons = []
C = {}
for r in have:
    vals = [tuple(d.get(k) for k in KEYS) for f, d in cfg[r]]
    if len(set(vals)) > 1: incons.append(r)
    C[r] = cfg[r][-1][1]
P(f"runs with disagreeing duplicate configs: {len(incons)} {incons[:5]}")
# defaults of the worker (acl_st_89_worker16.sh line 172): STEPS=120 TRAINF=tqa_h1.jsonl KLB=0 FPMODE=random SEED=0
for r in C:
    for k, v in (("STEPS", "120"), ("TRAINF", "tqa_h1.jsonl"), ("KLB", "0"), ("FPMODE", "random"), ("SEED", "0")):
        C[r].setdefault(k, v)
pickle.dump(C, open(f"{OUT}/v02_configs.pkl", "wb"))
# summaries that matter: TRAINF by (tag, ds); seed in name == SEED; hash runs' FPMODE and FPK
by = collections.Counter()
bad_seed = []
for r, d in C.items():
    m = re.match(r"^(qwen7|llama8)_([A-Z0-9a-z]+?K\d+|FK\d+)(.+)_idkc(\d+)_s(\d+)_(h1|pq|nq)$", r)
    if not m: continue
    tag, ds, s = m.group(2), m.group(6), m.group(5)
    by[(tag, ds, d["TRAINF"])] += 1
    if d["SEED"] != s: bad_seed.append((r, d["SEED"]))
P("runs by (tag, dataset, TRAINF):")
for k, v in sorted(by.items()): P("  ", k, v)
P(f"runs whose SEED differs from the name: {len(bad_seed)} {bad_seed[:5]}")
for r in sorted(C):
    if "hash35" in r or "B10vK20judgeQt50" in r:
        d = C[r]; P(f"  {r}: TRAINF {d['TRAINF']} SEED {d['SEED']} FPMODE {d['FPMODE']} FPK {d.get('FPK')} FPU {d.get('FPU')} KLB {d['KLB']} STEPS {d['STEPS']} JT {d.get('JT')}")
open(f"{OUT}/v02_configs.txt", "w").write("\n".join(L) + "\n")
