"""(fix_ledger) L01: collapse / all-IDK / degenerate-output inventory of EVERY finished run, from the raw eval files.
Read-only on project data; writes only to analysis/fix_ledger/.
Conventions (same as stats/s00_common.py, stated there before any estimate):
  answer = '' (malformed) if fin != 'stop', or a newline in the stripped text, or 'think>' in it; else st_common.parse_tq(text)
  IDK = full match of the project IDK list on the parsed answer
  new = ids not in <dsn>_h1.jsonl, minus questions whose lower-cased text or qid equals a training question's (rc.setup HELD)
  home = ids in <dsn>_h1.jsonl
Per run (final greedy answers, s = -1): share IDK / malformed on new and home questions, at ck120 and at the last checkpoint on disk.
Flags: COLLAPSED = new-question IDK share >= 0.95 at ck120 (the exclusion rule used by stats c5/c6/c78, verify_map, dose, arch_v4a);
       NEAR = 0.90 <= IDK < 0.95; DEGENERATE = malformed share >= 0.50 on new questions (no all-IDK, but no usable answers);
       LOWCOV = answers (not IDK, not malformed) on < 5% of new questions (selfcons s2 drop rule).
Metadata: main students from meta/cache/runs_table.csv; small students from general/g00_inventory.json; smoke runs skipped.
Output: L01_runs.csv (one row per run), L01_collapse_out.txt (tables by student x recipe x grader family).
Usage: nice -n 10 timeout 280 python3 -I L01_collapse.py"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, re, json, glob, csv
from collections import defaultdict, Counter
sys.path.insert(0, _RC)
import st_common as C

RL = (_RR + "/run1/rl"); DATA = _RD
HERE = (_RA + "/fix_ledger")
DSN = {"h1": "tqa", "pq": "pqa", "nq": "nqo"}
IDKF = re.compile(r"^(i don t know|i do not know|i dont know|idk|i m not sure|not sure|i cannot answer|cannot answer|no idea|i am not sure|unknown)$")

def jl(p): return [json.loads(l) for l in open(p)]

def setup(dsn):
    allq = jl(f"{DATA}/{dsn}_1500.jsonl"); H1 = {p["idx"] for p in jl(f"{DATA}/{dsn}_h1.jsonl")}
    trq = {p["question"].strip().lower() for p in allq if p["idx"] in H1}
    trid = {p.get("qid") for p in allq if p["idx"] in H1}
    held = {p["idx"] for p in allq if p["idx"] not in H1 and not (p["question"].strip().lower() in trq or (p.get("qid") is not None and p.get("qid") in trid))}
    return H1, held

SPLIT = {ds: setup(dsn) for ds, dsn in DSN.items()}

def ans(r):
    t = r.get("text", "")
    if r.get("fin") != "stop" or "\n" in t.strip() or "think>" in t: return ""
    return C.parse_tq(t)

def outcome_shares(path, ds):
    H1, held = SPLIT[ds]; c = {"new": Counter(), "home": Counter()}
    for line in open(path):
        r = json.loads(line)
        if r.get("s", -1) != -1: continue
        part = "home" if r["idx"] in H1 else ("new" if r["idx"] in held else None)
        if part is None: continue
        a = ans(r); k = "mal" if not a else ("idk" if IDKF.match(a) else "ans")
        c[part][k] += 1
    out = {}
    for part in ("new", "home"):
        n = sum(c[part].values()); out[f"n_{part}"] = n
        for k in ("idk", "mal", "ans"): out[f"{k}_{part}"] = c[part][k] / n if n else float("nan")
    return out

# ---------------------------------------------------------------- metadata
META = {}
for r in csv.DictReader(open((_RA + "/meta/cache/runs_table.csv"))):
    if r["st"] in ("qwen7", "llama8"):
        META[r["run"]] = dict(st=r["st"], ds=r["ds"], fam=r["fam"], sub=r["sub"], tag=r["tag"], kl=float(r["kl"]), c=float(r["c"]) if r["c"] else None,
                              adv=r["adv"], steps=int(float(r["steps"])), force=int(float(r["force"])), outlet="idk", fmt=int(float(r["fmt"])))
for r in json.load(open((_RA + "/general/g00_inventory.json"))):
    META[r["run"]] = dict(st=r["st"], ds=r["dtag"], fam=r["fam"], sub=r["grader"], tag=r.get("pre") or "old", kl=r["kl"], c=r["credit"], adv=r["adv"],
                          steps=r["steps"], force=0, outlet=r["outlet"], fmt=r["fmt"])

def family(m):
    """plain grader family used in the report tables."""
    f, s = m["fam"], m["sub"]
    if f == "judge":
        mm = re.match(r"judge([A-Z]+?)(a35|a50|t50)", s)
        return f"LLM judge {mm.group(2)}" if mm else "LLM judge (small-student threshold)"
    if f in ("gold", "key") and s in ("gold", "sim00fn00"): return "answer key"
    if f == "keyFN": return "key + random false rejects"
    if f == "keyFP": return "key + random false accepts"
    if f == "mixU": return "lenQ (random, judge's rates)" if s == "lenQ" else "key + random FA and FR"
    if f == "hash": return "hash (fixed false accepts)"
    if f == "api": return "Sonnet as reward"
    if f in ("frz", "selfvote", "quorum"): return "own-vote / frozen keys" if f != "quorum" else "LLM judge quorum"
    if f in ("plFN", "typeFN", "hidFN"): return "planted false rejects"
    if f in ("typeFP", "popFP"): return "planted false accepts"
    if f == "long": return "long-answer grader"
    if f == "blind": return "blinded key"
    return f"other ({f})"

def recipe(m):
    if m["st"] in ("qwen7", "llama8"):
        if m["tag"] == "FK20" and m["c"] == 0.5 and m["force"] == 0: return "main (KL 0.2, credit 0.5)"
        if m["tag"] == "FK20" and m["force"] == 1: return "forced IDK"
        if m["tag"] == "FK20": return f"credit {m['c']}"
        return {"B1": "1 visit", "B2": "2 visits (B2)", "B5": "5 visits", "B10": "10 visits", "FK00": "no KL", "FK50": "KL 0.5", "FK100": "KL 1.0",
                "LK20": "360 steps KL 0.2", "LK50": "360 steps KL 0.5"}.get(m["tag"], m["tag"])
    return f"{m['tag']} KL{m['kl']} {m['adv']} c{m['c']} {m['outlet']}"

rows = []
for d in sorted(glob.glob(f"{RL}/*")):
    run = os.path.basename(d)
    if "SMOKE" in run or run.startswith("zz_smoke"): continue
    m = META.get(run)
    if m is None: print("no metadata", run); continue
    cks = sorted(int(re.search(r"ck(\d+)", f).group(1)) for f in glob.glob(f"{d}/eval_rl_ck*.jsonl"))
    cks = [k for k in cks if k > 0]
    if not cks: print("no final eval", run); continue
    ck = 120 if 120 in cks else cks[-1]
    o = outcome_shares(f"{d}/eval_rl_ck{ck}.jsonl", m["ds"])
    last = cks[-1]; ol = o if last == ck else outcome_shares(f"{d}/eval_rl_ck{last}.jsonl", m["ds"])
    row = dict(run=run, **m, family=family(m), recipe=recipe(m), ck=ck, last_ck=last, **o, idk_new_last=ol["idk_new"], mal_new_last=ol["mal_new"])
    row["collapsed"] = int(o["idk_new"] >= 0.95); row["near"] = int(0.90 <= o["idk_new"] < 0.95)
    row["degenerate"] = int(o["mal_new"] >= 0.50); row["lowcov"] = int(o["ans_new"] < 0.05)
    rows.append(row)

keys = list(rows[0].keys())
with open(f"{HERE}/L01_runs.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]

out = open(f"{HERE}/L01_collapse_out.txt", "w")
def P(s=""): print(s); out.write(s + "\n")
P(__doc__)
P(f"runs with metadata and a final eval: {len(rows)} (smoke runs skipped)")
by_st = Counter(r["st"] for r in rows); P("by student: " + ", ".join(f"{k} {v}" for k, v in sorted(by_st.items())))

def table(sel, title, key=lambda r: r["family"]):
    P(f"\n== {title}")
    G = defaultdict(list)
    for r in rows:
        if sel(r): G[key(r)].append(r)
    P(f"{'group':44s} {'runs':>5s} {'collapsed':>9s} {'rate':>6s} {'near':>5s} {'degen':>6s} {'lowcov':>6s} | new IDK median (min-max)")
    tot = Counter()
    for g in sorted(G):
        R = G[g]; n = len(R); c = sum(r["collapsed"] for r in R); nr = sum(r["near"] for r in R); dg = sum(r["degenerate"] for r in R)
        lc = sum(r["lowcov"] for r in R); idk = sorted(r["idk_new"] for r in R)
        tot.update(dict(n=n, c=c, nr=nr, dg=dg, lc=lc))
        P(f"{g:44s} {n:5d} {c:9d} {c / n:6.2f} {nr:5d} {dg:6d} {lc:6d} | {idk[len(idk) // 2]:.2f} ({idk[0]:.2f}-{idk[-1]:.2f})")
    if tot["n"]: P(f"{'ALL':44s} {tot['n']:5d} {tot['c']:9d} {tot['c'] / tot['n']:6.2f} {tot['nr']:5d} {tot['dg']:6d} {tot['lc']:6d}")

MAIN = lambda r: r["st"] in ("qwen7", "llama8")
for st in ("qwen7", "llama8"):
    table(lambda r, st=st: r["st"] == st and r["recipe"].startswith("main"), f"{st}: main recipe (FK20, KL 0.2, credit 0.5, 120 steps), by grader family")
table(lambda r: MAIN(r) and r["recipe"].startswith("main"), "both main students: main recipe, by grader family")
table(lambda r: MAIN(r) and r["recipe"].startswith("main"), "both main students: main recipe, by student x dataset", key=lambda r: f"{r['st']} {r['ds']}")
table(lambda r: MAIN(r) and r["recipe"].startswith("main") and r["family"].startswith("LLM judge"), "LLM judges (main recipe), by judge code",
      key=lambda r: r["sub"])
table(lambda r: MAIN(r) and not r["recipe"].startswith("main"), "both main students: other recipes, by recipe", key=lambda r: r["recipe"])
table(lambda r: not MAIN(r), "small students (phi4m, smol3, granite), by student x recipe prefix x outlet",
      key=lambda r: f"{r['st']} {r['tag']} KL{r['kl']} {r['adv']} {r['outlet']}")
table(lambda r: not MAIN(r) and r["outlet"] == "idk", "small students with an IDK outlet, by grader family")

P("\n== collapsed runs (new-question IDK >= 0.95 at ck120), main students, with recipe; home IDK in brackets")
for r in rows:
    if MAIN(r) and r["collapsed"]:
        P(f"  {r['run']:46s} {r['recipe']:26s} {r['family']:30s} new IDK {r['idk_new']:.3f} (home {r['idk_home']:.3f}); last ck{r['last_ck']} {r['idk_new_last']:.3f}")
P("\n== degenerate runs (malformed >= 0.50 on new questions), all students")
for r in rows:
    if r["degenerate"]:
        P(f"  {r['run']:46s} {r['recipe']:30s} malformed {r['mal_new']:.3f} IDK {r['idk_new']:.3f}")
P("\n== near-collapsed (0.90 <= new IDK < 0.95), main students")
for r in rows:
    if MAIN(r) and r["near"]:
        P(f"  {r['run']:46s} {r['recipe']:26s} new IDK {r['idk_new']:.3f}")
out.close()
