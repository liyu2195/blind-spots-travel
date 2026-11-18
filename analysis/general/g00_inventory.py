"""(explorer 'general') Inventory of the small-student runs: grader, recipe and seed from the run name, checked against every
queue / control config that names the run, and the train log. Output: g00_inventory.json, g00_inventory.txt."""
import json, os, sys
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gcommon import load_configs, parse_run, final_ck, RL, SMALL, HERE

cfgs = load_configs()
runs = sorted(r for r in os.listdir(RL) if r.split("_")[0] in SMALL)
rows, bad = [], []
for run in runs:
    p = parse_run(run, cfgs)
    if p is None:
        bad.append(run); continue
    p["final"] = final_ck(run)
    p["has_ck0"] = os.path.exists(f"{RL}/{run}/eval_rl_ck0.jsonl")
    lg = f"{RL}/{run}/train_log.jsonl"
    if os.path.exists(lg):
        L = [json.loads(l) for l in open(lg)]
        p["log_steps"] = len(L)
        p["log_has_kl"] = any(x.get("kl") not in (None, 0.0) for x in L)
        p["kl_meas_last20"] = (sum((x.get("kl") or 0) for x in L[-20:]) / max(1, len(L[-20:])))
        p["idk_last20"] = sum((x.get("idk") or 0) for x in L[-20:]) / max(1, len(L[-20:]))
        wp = [x.get("wrong_paid") for x in L]
        p["wp_first23"] = (lambda v: sum(v) / len(v) if v else None)([w for w in wp[:23] if w is not None])
        p["wp_last20"] = (lambda v: sum(v) / len(v) if v else None)([w for w in wp[-20:] if w is not None])
        p["judge_last20"] = (lambda v: sum(v) / len(v) if v else None)([x.get("judge") for x in L[-20:] if x.get("judge") is not None])
    else:
        p["log_steps"] = 0
    p["scores"] = sorted(f[9:-6] for f in os.listdir(f"{RL}/{run}") if f.startswith("score_rl_"))
    # consistency checks between name-derived recipe and configs
    p["conflicts"] = []
    if "kl_cfg" in p and abs(p["kl_cfg"] - p["kl"]) > 1e-9: p["conflicts"].append(f"kl name {p['kl']} cfg {p['kl_cfg']}")
    if "adv_cfg" in p and p["adv_cfg"] != p["adv"]: p["conflicts"].append(f"adv name {p['adv']} cfg {p['adv_cfg']}")
    if "credit_cfg" in p and p["credit"] is not None and abs(p["credit_cfg"] - p["credit"]) > 1e-9: p["conflicts"].append(f"credit name {p['credit']} cfg {p['credit_cfg']}")
    if "fmt_cfg" in p and p["fmt_cfg"] != p["fmt"]: p["conflicts"].append(f"fmt name {p['fmt']} cfg {p['fmt_cfg']}")
    if "jmode_cfg" in p and p["jmode"] and p["jmode_cfg"] != p["jmode"]: p["conflicts"].append(f"jmode name {p['jmode']} cfg {p['jmode_cfg']}")
    if "thr_cfg" in p and p["thr"] is not None and abs(p["thr_cfg"] - p["thr"]) > 1e-9: p["conflicts"].append(f"thr name {p['thr']} cfg {p['thr_cfg']}")
    if p["log_steps"] and p.get("log_has_kl") != (p["kl"] > 0): p["conflicts"].append(f"log kl present {p.get('log_has_kl')} vs name kl {p['kl']}")
    p.pop("cfg", None)
    rows.append(p)
json.dump(rows, open(f"{HERE}/g00_inventory.json", "w"), indent=0)
out = open(f"{HERE}/g00_inventory.txt", "w")
def P(*a): print(*a, file=out)
P(f"small-student run folders: {len(runs)} (unparsed: {bad})")
P("by student:", dict(Counter(r["st"] for r in rows)))
P("with a config file:", sum("cfg_src" in r for r in rows), "| with final eval:", sum(r["final"] is not None for r in rows))
P("final checkpoint:", dict(Counter(r["final"] for r in rows)))
P("\nrecipe cells (prefix, KL, adv, format, temp, steps) x grader family -> runs")
cell = defaultdict(Counter)
for r in rows:
    key = (r["pre"] or "old", r["kl"], r["adv"], r["fmt"], r["temp"], r["steps"], r["outlet"])
    cell[key][r["fam"]] += 1
for k in sorted(cell, key=str):
    P(f"  prefix {k[0]:5s} KL {k[1]:<5} adv {k[2]:4s} fmt {k[3]} T {k[4]} steps {k[5]} outlet {k[6]:5s}: {dict(cell[k])}")
P("\ncredits:", dict(Counter((r["credit"]) for r in rows)))
P("seeds:", dict(Counter(r["seed"] for r in rows)))
P("datasets:", dict(Counter((r["st"], r["ds"]) for r in rows)))
P("\njudge runs (judge model, mode, threshold, outlet, recipe prefix):")
for r in rows:
    if r["fam"] == "judge":
        P(f"  {r['run']:40s} {r['jm']} {r['jmode']:4s} thr {r['thr']} outlet {r['outlet']} credit {r['credit']} pre {r['pre'] or 'old'} KL {r['kl']} final ck{r['final']} scores {r['scores']} cfg {r.get('cfg_src','-')[:60]}")
P("\nconflicts between name and config / log:")
for r in rows:
    if r["conflicts"]:
        P(f"  {r['run']}: {r['conflicts']}")
P("\nruns without a config file:", " ".join(r["run"] for r in rows if "cfg_src" not in r))
out.close()
print(open(f"{HERE}/g00_inventory.txt").read()[:12000])
