"""(verify_what) Item 2, part 1: rebuild the trainer's batch schedule from st_grpo.py / st_grpo_b.py and validate it.
Trainer: random.seed(seed); order = list(range(len(train_file))); random.shuffle(order); 32 rows per step taken in order; when the
order is used up (also mid-batch) it is reshuffled. Visit number of a row = how many times it has been drawn so far.
Validation, all qwen7 / llama8 runs with a config TRAINF of the form <set>_h1.jsonl (no blind / frz files, no forced-IDK suffix):
 (V1) step-1 strict correctness predicted from the 32 untrained samples of the 32 scheduled questions vs the logged correct_strict,
      true seed vs 200 wrong seeds (placebo), within each (student, dataset);
 (V2) the step-1 log is a fingerprint of the batch: runs with the same (student, dataset, seed) and different graders should log the
      same step-1 correct_strict / idk / distinct_per_group;
 (V3) hash35 runs: the paid share of wrong samples per first-pass step (1-23) predicted from the untrained samples, the schedule and my
      re-implementation of the trainer's hash (sha256 of 'fp|seed|idx|answer' < 0.35), vs the logged wrong_paid; placebos: wrong
      schedule seed, wrong hash seed.
Writes logs.pkl (all large-student train logs) and v20_sched.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import os, re, json, glob, random, pickle, hashlib, sys, numpy as np
sys.path.insert(0, _RC)
import st_common as C
OUT = (_RA + "/verify_what"); RL = (_RR + "/run1/rl"); DATA = _RD
IDK_FULL = re.compile(r"^(i don t know|i do not know|i dont know|idk|i m not sure|not sure|i cannot answer|cannot answer|no idea|i am not sure|unknown)$")
DSN = {"h1": "tqa", "pq": "pqa", "nq": "nqo"}
L = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); L.append(s)

def schedule(seed, n, steps=120, batch=32):
    rnd = random.Random(seed); order = list(range(n)); rnd.shuffle(order); ptr = 0; seen = [0] * n; out = []
    for _ in range(steps):
        b = []
        for _ in range(batch):
            if ptr == n: rnd.shuffle(order); ptr = 0
            j = order[ptr]; ptr += 1; seen[j] += 1; b.append((j, seen[j]))
        out.append(b)
    return out

def hunit(*parts):
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12], 16) / float(16 ** 12)

if __name__ == "__main__":
    P(__doc__)
    CFG = pickle.load(open(f"{OUT}/v02_configs.pkl", "rb"))
    # logs
    LOG = {}
    for d in sorted(glob.glob(f"{RL}/*")):
        name = os.path.basename(d)
        if not re.match(r"^(qwen7|llama8)_", name) or not os.path.exists(f"{d}/train_log.jsonl"): continue
        rows = [json.loads(l) for l in open(f"{d}/train_log.jsonl")]
        LOG[name] = {k: np.array([np.nan if r.get(k) is None else float(r[k]) for r in rows]) for k in
                     ("step", "reward", "correct", "correct_strict", "idk", "wrong_paid", "distinct_per_group", "resp_tokens", "invalid", "judge", "kl")}
    pickle.dump(LOG, open(f"{OUT}/logs.pkl", "wb"))
    P(f"logs: {len(LOG)} runs")
    rng = np.random.default_rng(0)
    for ds in ("h1", "pq", "nq"):
        tf = f"{DSN[ds]}_h1.jsonl"; ids = [json.loads(l)["idx"] for l in open(f"{DATA}/{tf}")]
        for st in ("qwen7", "llama8"):
            Dd = pickle.load(open(f"{OUT}/cache_{st}_{ds}.pkl", "rb")); Q = Dd["Q"]; B = Dd["B"]
            al = Q["aliases"]
            qs = np.array([np.mean([bool(x) and C.tq_strict(x, al[i]) for x in B["s"][i]]) for i in ids])   # per training row
            runs = [r for r in LOG if r.startswith(st + "_") and r.endswith("_" + ds) and r in CFG and CFG[r]["TRAINF"] == tf
                    and not re.search(r"(f1|q3|q5)_idkc", r) and len(LOG[r]["step"]) >= 1 and LOG[r]["step"][0] == 1]
            seeds = sorted({int(CFG[r]["SEED"]) for r in runs})
            # V2: fingerprint
            spread = []
            for sd in seeds:
                v = np.array([LOG[r]["correct_strict"][0] for r in runs if int(CFG[r]["SEED"]) == sd])
                if len(v) > 1: spread.append((sd, len(v), np.ptp(v), np.mean(np.abs(v - np.median(v)) < 1e-9)))
            # V1: true vs wrong seeds
            err_true, err_wrong = [], []
            for sd in seeds:
                obs = np.median([LOG[r]["correct_strict"][0] for r in runs if int(CFG[r]["SEED"]) == sd])
                pred = qs[[j for j, v in schedule(sd, len(ids), steps=1)[0]]].mean()
                err_true.append(abs(pred - obs))
                for w in rng.choice(np.arange(1000, 100000), 200, replace=False):
                    pw = qs[[j for j, v in schedule(int(w), len(ids), steps=1)[0]]].mean(); err_wrong.append(abs(pw - obs))
            ew = np.array(err_wrong)
            P(f"{st} {ds}: runs {len(runs)} seeds {seeds}: step-1 |pred - obs| true seed mean {np.mean(err_true):.4f} max {np.max(err_true):.4f}; "
              f"wrong seeds mean {ew.mean():.4f}, share of wrong-seed errors below the true-seed mean {np.mean(ew < np.mean(err_true)):.3f}; "
              f"fingerprint: seeds with >1 run {len(spread)}, max range of step-1 correct_strict across graders {max([s[2] for s in spread], default=np.nan):.4f}, "
              f"share identical {np.mean([s[3] for s in spread]):.3f}")
            # V3: hash runs, per-step paid share during the first pass
            for r in sorted(x for x in runs if "hash35" in x and x.startswith(st + "_FK20")):
                sd = int(CFG[r]["SEED"]); obs = LOG[r]["wrong_paid"]
                def pred_paid(sched_seed, hash_seed):
                    S = schedule(sched_seed, len(ids), steps=23); out = []
                    for b in S:
                        num = den = 0.0
                        for j, v in b:
                            i = ids[j]
                            for x in B["s"][i]:
                                if not x or IDK_FULL.match(x) or C.tq_strict(x, al[i]): continue
                                den += 1; num += hunit("fp", hash_seed, i, x) < 0.35
                        out.append(num / den if den else np.nan)
                    return np.array(out)
                p = pred_paid(sd, sd); o = obs[:23]
                ok = np.isfinite(p) & np.isfinite(o)
                rt = np.corrcoef(p[ok], o[ok])[0, 1]
                rws = [np.corrcoef(pred_paid(sd + 7 + k, sd)[ok], o[ok])[0, 1] for k in range(3)]
                rwh = [np.corrcoef(pred_paid(sd, sd + 7 + k)[ok], o[ok])[0, 1] for k in range(3)]
                P(f"   V3 {r}: steps 1-23 corr(pred, logged wrong_paid) true {rt:+.3f} | wrong schedule seed {np.mean(rws):+.3f} | wrong hash seed {np.mean(rwh):+.3f}; "
                  f"step 1 pred {p[0]:.3f} obs {o[0]:.3f}; mean |pred - obs| steps 1-23 {np.nanmean(np.abs(p - o)):.3f}")
    open(f"{OUT}/v20_sched.txt", "w").write("\n".join(L) + "\n")
