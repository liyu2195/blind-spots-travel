"""(rev_R5) Check: is 'content stays home' tied to the KL anchor? Compare no-KL (FK00) with KL 0.2 (FK20) runs that share grader
and seeds (TriviaQA; judgeQt50 and sim35fn00, seeds 0-1). Per run, on HOME (trained) and NEW (duplicate-free held-out) questions:
abstention, key-right, key-wrong (loose key, final greedy), and among answered questions the share whose final answer differs from the
untrained greedy answer (rc.same: equal or whole-word containment either way), split by key-right / key-wrong final answers.
Helpers: analysis/extra/common/rc.py (read-only import). Output: r5_kl.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, numpy as np
sys.path.insert(0, (_RX + "/common"))
import rc
probs, H1, HELD = rc.setup("tqa")
HOME = sorted(H1)
def base_greedy(st):
    g = {}
    import json
    for r in map(json.loads, open(f"{rc.BASE}/{st}_tqa_1500_n32_s1.jsonl")):
        if r["s"] < 0: g[r["idx"]] = rc.sa(r)
    return g
out = []
for st in ("qwen7", "llama8"):
    BG = base_greedy(st)
    for g in ("judgeQt50", "sim35fn00"):
        for tag in ("FK20", "FK00"):
            for s in (0, 1):
                run = f"{st}_{tag}{g}_idkc05_s{s}_h1"
                if not os.path.isdir(os.path.join(rc.RL, run)): continue
                G, _ = rc.load(run, "ck120")
                res = []
                for part, items in (("home", HOME), ("new", HELD)):
                    n = len(items); idk = right = wrong = 0; ch_r = ch_w = nr = nw = 0
                    for i in items:
                        x = rc.sa(G[i])
                        if not x: continue
                        if rc.IDK.match(x): idk += 1; continue
                        ok = rc.C.tq_correct(x, set(probs[i]["aliases"]))
                        b = BG.get(i, "") or ""
                        changed = not (b and rc.same(x, b))
                        if ok: right += 1; nr += 1; ch_r += changed
                        else: wrong += 1; nw += 1; ch_w += changed
                    res.append(f"{part}: idk {idk/n:.3f} right {right/n:.3f} wrong {wrong/n:.3f} | changed among right {ch_r/max(1,nr):.3f} among wrong {ch_w/max(1,nw):.3f}")
                out.append(f"{run:38s} " + " || ".join(res))
open((_RA + "/rev_R5/r5_kl.txt"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
