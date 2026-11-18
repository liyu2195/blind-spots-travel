"""(fix_ledger) L03: seeds per grader configuration (student x dataset x grader code) for the run sets behind each headline claim,
counted from L01_runs.csv (raw eval files; collapsed = new-question IDK >= 0.95 at ck120). Output: L03_seeds_out.txt
Usage: python3 -I L03_seeds.py"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import csv, re
from collections import defaultdict
HERE = (_RA + "/fix_ledger")
rows = list(csv.DictReader(open(f"{HERE}/L01_runs.csv")))
out = open(f"{HERE}/L03_seeds_out.txt", "w")
def P(s=""): print(s); out.write(s + "\n")
P(__doc__)
def seed(r): m = re.search(r"_s(\d+)_", r["run"]); return int(m.group(1)) if m else 0
def show(title, sel, alive_only=True):
    G = defaultdict(list)
    for r in rows:
        if r["st"] in ("qwen7", "llama8") and sel(r) and (not alive_only or r["collapsed"] == "0"):
            G[(r["st"], r["ds"], r["sub"])].append(seed(r))
    P(f"\n== {title}  ({'collapsed runs excluded' if alive_only else 'all runs'})")
    for st in ("qwen7", "llama8"):
        cells = {k: v for k, v in G.items() if k[0] == st}
        if not cells: continue
        n = [len(v) for v in cells.values()]
        P(f"  {st}: runs {sum(n)} in {len(cells)} configurations; seeds per configuration min {min(n)} median {sorted(n)[len(n) // 2]} max {max(n)}")
        P("     " + "; ".join(f"{k[1]} {k[2]} {len(v)}" for k, v in sorted(cells.items())))
main = lambda r: r["recipe"].startswith("main")
show("strict LLM judges (35% cut): crossover, map figure, dose-response, readable split",
     lambda r: main(r) and re.fullmatch(r"judge[QMGOR]a35", r["sub"]) is not None)
show("Qwen judge at three operating points (strictness dose-response)", lambda r: main(r) and re.fullmatch(r"judgeQ(a35|a50|t50)", r["sub"]) is not None)
show("other judges at P(Yes) > 0.5", lambda r: main(r) and re.fullmatch(r"judge[MGO]t50", r["sub"]) is not None)
show("answer key (gold) runs", lambda r: main(r) and r["sub"] == "gold")
show("answer key, all runs incl. collapsed", lambda r: main(r) and r["sub"] == "gold", alive_only=False)
show("key + random false rejects (no-location placebo pool)", lambda r: main(r) and r["fam"] == "keyFN")
show("key + random false rejects, all runs incl. collapsed", lambda r: main(r) and r["fam"] == "keyFN", alive_only=False)
show("lenQ (random grader at the default Qwen judge's error rates)", lambda r: main(r) and r["sub"] == "lenQ")
show("Sonnet as reward", lambda r: main(r) and r["fam"] == "api")
show("hash (fixed false accepts)", lambda r: r["fam"] == "hash")
show("causal arms (PopQA): copies / scrambled / type-wide copies / planted sets", lambda r: main(r) and r["fam"] in ("plFN", "typeFN", "hidFN"))
show("causal arms incl. collapsed", lambda r: main(r) and r["fam"] in ("plFN", "typeFN", "hidFN"), alive_only=False)
show("visit regimes (TriviaQA, Qwen judge P(Yes) > 0.5 and others)", lambda r: r["recipe"] in ("1 visit", "2 visits (B2)", "5 visits", "10 visits"))
show("uniform random-error graders (leniency-curve fit / test set)", lambda r: main(r) and r["fam"] in ("gold", "keyFN", "keyFP", "mixU"))
show("off-recipe runs of the main students", lambda r: not main(r) and r["recipe"] not in ("1 visit", "2 visits (B2)", "5 visits", "10 visits"),
     alive_only=False)
out.close()
