"""(question, answer) pairs of the UNTRAINED students for key-free judging by Claude agents.
Key = (question idx, normalised answer). Shown to the judge: the question and the most common raw reply for that answer."""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, json, random
sys.path.insert(0, (_RX + "/common"))
from rc import *
from collections import Counter
probs, H1, HELD = setup("tqa"); TR = sorted(H1)
P = {}   # (idx, norm) -> Counter(raw)
use = defaultdict(set)
for st, src in (("phi4m", "phi4m_judgeQ_idk_h1"), ("smol3", "smol3_goldstrict_idk_h1")):
    g, s = load(src, "ck0")
    for i in TR:
        for r in s[i]:
            x = sa(r)
            if x and not IDK.match(x): P.setdefault((i, x), Counter())[C.tq_line(r["text"]).strip()] += 1; use[(i, x)].add(f"{st}_trainS")
    for i in HELD:
        x = sa(g[i])
        if x and not IDK.match(x): P.setdefault((i, x), Counter())[C.tq_line(g[i]["text"]).strip()] += 1; use[(i, x)].add(f"{st}_heldG")
keys = sorted(P); random.Random(7).shuffle(keys)
rows = [dict(id=k, idx=i, norm=x, q=probs[i]["question"].strip(), a=P[(i, x)].most_common(1)[0][0], use=sorted(use[(i, x)])) for k, (i, x) in enumerate(keys)]
json.dump(rows, open("pairs_v1.json", "w"))
print(len(rows), "pairs;", sum(any(u.endswith("trainS") for u in r["use"]) for r in rows), "from trained-question samples;", sum(any(u.endswith("heldG") for u in r["use"]) for r in rows), "from held-out greedy answers")
print("mean chars per line:", sum(len(r["q"]) + len(r["a"]) for r in rows) / len(rows))
B = 250
for b in range(0, len(rows), B):
    with open(f"in_{b//B:03d}.tsv", "w") as f:
        for r in rows[b:b + B]: f.write(f"{r['id']}\t{r['q'].replace(chr(9), ' ').replace(chr(10), ' ')}\t{r['a'].replace(chr(9), ' ').replace(chr(10), ' ')}\n")
print((len(rows) + B - 1) // B, "batch files of", B)
