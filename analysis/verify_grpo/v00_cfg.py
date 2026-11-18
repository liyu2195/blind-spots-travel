"""(verify_grpo) Independent run-configuration table.
Sources: every KEY=VALUE line naming OUTNAME in training/queues/*, training/queue_*, AND training/queues/** (the grpo explorer
did not read ops/q). Worker defaults: old workers (acl_st_61/63/64, queues sim / hyst) ran st_grpo.py without --strict_format and without --kl_beta
(fmt 0, KL 0); workers v5+ (acl_st_67 ... 89) default FMT=1, KLB=0, ADV=std, IDK=0.5, TEMP=0.8, TOPP=0.95, LRATE=5e-5, STEPS=120.
Every run is then checked against its own train_log.jsonl: 'invalid' logged and > 0 at some step => strict format on; 'kl' logged and > 0 => KL on;
number of logged steps; for key-graded runs (FP = FN = 0) the IDK credit implied by reward = c * idk + correct_strict.
Output: cache/cfg_v.json (one record per run on disk) and a comparison with the grpo explorer's eval table.
Usage: nice -n 10 python3 -I v00_cfg.py > out_v00_cfg.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import os, re, json, glob
from collections import Counter, defaultdict
HERE = (_RA + "/verify_grpo"); RL = (_RR + "/run1/rl")
roots = sorted(glob.glob((_RT + "/queues/*"))) + sorted(glob.glob((_RT + "/queue_*"))) + [(_RT + "/queues")]
lines = defaultdict(list)
for r in roots:
    files = [r] if os.path.isfile(r) else [os.path.join(dp, f) for dp, _, fs in os.walk(r) for f in sorted(fs)]
    for p in files:
        try: txt = open(p, errors="replace").read()
        except Exception: continue
        for ln in txt.splitlines():
            if ln.lstrip().startswith("#") or "OUTNAME=" not in ln: continue
            kv = dict(re.findall(r"\b([A-Z][A-Z_0-9]*)=(\S+)", ln))
            if "OUTNAME" in kv: lines[kv["OUTNAME"]].append((p, kv))
def old_queue(p):
    return bool(re.search(r"/(queue_sim2?|queue_hyst)(/|$)|/ops/q/(sim|hyst|csim|csim_now|chyst)(/|$)|/ops/q/p\d/(sim|hyst)(/|$)", p))
runs = sorted(d for d in os.listdir(RL) if os.path.isdir(f"{RL}/{d}"))
CFG = {}; conflicts = []
for run in runs:
    L = lines.get(run, [])
    distinct = {json.dumps(kv, sort_keys=True) for _, kv in L}
    if len(distinct) > 1: conflicts.append((run, [p for p, _ in L]))
    rec = {}
    if L:
        p, kv = L[-1]; rec.update(kv); rec["_src"] = p; rec["_old"] = old_queue(p)
        if rec["_old"]:
            rec.setdefault("FMT", "0"); rec.setdefault("KLB", "0")
        rec.setdefault("FMT", "1"); rec.setdefault("KLB", "0"); rec.setdefault("ADV", "std"); rec.setdefault("IDK", "0.5")
        rec.setdefault("TEMP", "0.8"); rec.setdefault("LRATE", "5e-5"); rec.setdefault("STEPS", "120"); rec.setdefault("FPMODE", "random")
        rec.setdefault("FORCE", "0"); rec.setdefault("SWITCH", "0"); rec.setdefault("PROMPT", "idk"); rec.setdefault("REWARD", "synth")
    else:
        rec["_src"] = None
    # train-log facts
    tl = f"{RL}/{run}/train_log.jsonl"; Lg = []
    if os.path.exists(tl):
        for l in open(tl):
            try: Lg.append(json.loads(l))
            except Exception: pass
    rec["_log_steps"] = len(Lg)
    rec["_log_has_invalid"] = bool(Lg) and all("invalid" in x for x in Lg)
    rec["_log_max_invalid"] = max([x.get("invalid") or 0 for x in Lg], default=None)
    rec["_log_has_kl"] = bool(Lg) and all("kl" in x for x in Lg)
    rec["_log_mean_kl"] = (sum((x.get("kl") or 0) for x in Lg) / len(Lg)) if Lg else None
    cs = [(x["reward"] - x["correct_strict"]) / x["idk"] for x in Lg if x.get("idk", 0) > 0.05 and x.get("correct_strict") is not None]
    rec["_log_c_implied_median"] = sorted(cs)[len(cs) // 2] if cs else None
    cks = sorted(int(m.group(1)) for m in (re.search(r"eval_rl_ck(\d+)\.jsonl$", f) for f in os.listdir(f"{RL}/{run}")) if m)
    rec["_cks"] = cks
    CFG[run] = rec
json.dump(CFG, open(f"{HERE}/cache/cfg_v.json", "w"))
print(f"queue lines name {len(lines)} runs; runs on disk {len(runs)}; with a queue line {sum(1 for r in runs if CFG[r]['_src'])}; conflicting duplicates {len(conflicts)}")
print("runs without a queue line, by student:", Counter(r.split('_')[0] for r in runs if not CFG[r]['_src']))
ops = [r for r in runs if CFG[r]['_src'] and '/ops/q' in CFG[r]['_src']]
print(f"runs whose (last) queue line is in {_RT}/queues (not read by the grpo explorer): {len(ops)}; by student {Counter(r.split('_')[0] for r in ops)}")
for r, ps in conflicts[:8]: print("   conflict:", r, ps)
# consistency of stated FMT / KL with the train log
bad_fmt = [r for r in runs if CFG[r]['_src'] and CFG[r]['_log_steps'] and ((CFG[r]['FMT'] == '1') != bool(CFG[r]['_log_has_invalid'] and (CFG[r]['_log_max_invalid'] or 0) > 0))]
bad_kl = [r for r in runs if CFG[r]['_src'] and CFG[r]['_log_steps'] and ((float(CFG[r]['KLB']) > 0) != bool(CFG[r]['_log_has_kl'] and (CFG[r]['_log_mean_kl'] or 0) > 0))]
print(f"stated FMT vs log (invalid > 0 at some step): {len(bad_fmt)} disagreements, e.g. {[(r, CFG[r]['FMT'], CFG[r]['_log_has_invalid'], CFG[r]['_log_max_invalid']) for r in bad_fmt[:6]]}")
print(f"stated KLB vs log (mean kl > 0): {len(bad_kl)} disagreements, e.g. {[(r, CFG[r]['KLB'], CFG[r]['_log_mean_kl']) for r in bad_kl[:6]]}")
gold = [r for r in runs if CFG[r]['_src'] and CFG[r].get('REWARD', 'synth') == 'synth' and float(CFG[r].get('FPK', 0)) == 0 and float(CFG[r].get('FPU', 0)) == 0
        and float(CFG[r].get('FN', 0) or 0) == 0 and not CFG[r].get('FNK') and not CFG[r].get('TRAINF', '').count('_') > 1 and CFG[r]['_log_c_implied_median'] is not None]
badc = [r for r in gold if abs(CFG[r]['_log_c_implied_median'] - float(CFG[r]['IDK'])) > 0.02]
print(f"key-graded runs: implied credit from the log vs stated IDK: {len(gold)} runs checked, {len(badc)} off by > 0.02, e.g. {[(r, CFG[r]['IDK'], round(CFG[r]['_log_c_implied_median'], 3)) for r in badc[:6]]}")
# compare with the grpo explorer's run table
try:
    import pandas as pd
    T = pd.read_csv((_RA + "/grpo/eval_table_k1.0.csv"))
    diffs = Counter(); ex = defaultdict(list)
    for t in T.itertuples():
        c = CFG.get(t.run)
        if not c or not c['_src']: diffs['no queue line (mine)'] += 1; ex['no queue line (mine)'].append(t.run); continue
        for k, mine, theirs in (("kl", float(c['KLB']), float(t.kl)), ("c", float(c['IDK']), float(t.c)), ("adv", c['ADV'], t.adv), ("fmt", str(c['FMT']), str(t.fmt)),
                                ("steps", int(c['STEPS']), int(t.steps)), ("temp", float(c['TEMP']), float(t.temp)), ("lr", float(c['LRATE']), float(t.lr))):
            if (mine != theirs) if isinstance(mine, str) else abs(mine - theirs) > 1e-9:
                diffs[k] += 1; ex[k].append((t.run, mine, theirs))
    print(f"\ngrpo explorer's eval table: {len(T)} runs; config disagreements with mine:", dict(diffs))
    for k, v in ex.items(): print(f"   {k}: {v[:10]}")
except Exception as e:
    print("comparison failed:", e)
