"""(verify_what) Item 2(b) check: does the '22/22 multi-visit judgeQt50 runs jump at the first revisit' count depend on the window?
jump(w) = mean(first w steps of visit 2) - mean(last w steps of visit 1), w = 3, 5, 10; placebo = the same steps in the 4 B1 runs of the
student (no revisit). Also a slope-adjusted jump: the visit-2 window minus the extrapolation of a line fitted to the visit-1 steps.
Output v22b_jumpwin.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import re, pickle, sys, numpy as np
sys.path.insert(0, (_RA + "/verify_what"))
from v20_sched import schedule
OUT = (_RA + "/verify_what"); DATA = _RD
LOG = pickle.load(open(f"{OUT}/logs.pkl", "rb")); CFG = pickle.load(open(f"{OUT}/v02_configs.pkl", "rb"))
NQ = {f + ".jsonl": sum(1 for _ in open(f"{DATA}/{f}.jsonl")) for f in ("tqa_b1_3840", "tqa_b1_1920", "tqa_b1_768", "tqa_b1_384", "tqa_h1")}
def runs_of(g, st):
    out = []
    for r in LOG:
        m = re.match(rf"^{st}_(B1K20|B2vK20|B5vK20|B10vK20|FK20){g}_idkc05_s(\d+)_h1$", r)
        if m and r in CFG: out.append((r, m.group(1), int(m.group(2))))
    return sorted(out)
def visits(r):
    S = schedule(int(CFG[r]["SEED"]), NQ[CFG[r]["TRAINF"]], steps=len(LOG[r]["step"]))
    return np.array([b[0][1] if len({v for j, v in b}) == 1 else 0 for b in S])
L = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); L.append(s)
P(__doc__)
for st in ("qwen7", "llama8"):
    RS = runs_of("judgeQt50", st); V = {r: visits(r) for r, _, _ in RS}
    B1 = [r for r, reg, _ in RS if reg == "B1K20"]
    for w in (3, 5, 10):
        pos = above = adjpos = 0; n = 0; js = []; pls = []
        for r, reg, s in RS:
            if reg == "B1K20": continue
            v = V[r]; i1 = np.where(v == 1)[0]; i2 = np.where(v == 2)[0]; y = LOG[r]["wrong_paid"]
            a, b = i2[:w], i1[-w:]
            j = np.nanmean(y[a]) - np.nanmean(y[b]); pl = [np.nanmean(LOG[q]["wrong_paid"][a]) - np.nanmean(LOG[q]["wrong_paid"][b]) for q in B1]
            c = np.polyfit(i1, y[i1], 1); adj = np.nanmean(y[a]) - np.mean(np.polyval(c, a))
            n += 1; pos += j > 0; above += j > max(pl); adjpos += adj > 0; js.append(j); pls.append(np.mean(pl))
        P(f"{st} window {w:2d}: jump > 0 in {pos}/{n}; above every B1 placebo {above}/{n}; slope-adjusted jump > 0 {adjpos}/{n}; mean jump {np.mean(js):+.3f}; mean B1 placebo {np.mean(pls):+.3f}")
open((_RA + "/verify_what/v22b_jumpwin.txt"), "w").write("\n".join(L) + "\n")
