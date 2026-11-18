"""Phase 4: planted false rejects by question set (PopQA), written BEFORE the runs. Read-only.
T(run) = [abstention - confidence rule] on new group-B questions minus the same on new group-A questions (as phase3/measures.py, E3).
E(cell) = mean T of the cell minus the uniform reference at the cell's mean abstention (straight lines through the (mean abstention, mean T) points of the
uniform cells; flat beyond the ends). Here the uniform cells are the uniform false-reject cells FN 0 / 0.20 / 0.40 / 0.50 (all seeds that exist).
Hidden arm: HOME = abstention on trained questions of the high-reject half minus the other half; NEW = the same contrast between the pseudo-halves of new questions.
Usage: python3 fn_score.py [draws, default 1000] [selftest]"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, json, numpy as np
sys.path.insert(0, (_RX + "/common"))
from rc import *
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 1000; SELF = len(sys.argv) > 2
SP = json.load(open((_RX + "/supp2/split.json"))); HID = {int(k): v for k, v in json.load(open((_RX + "/final/hid_halves.json"))).items()}
rng = np.random.default_rng(20261114)
if SELF:   # reproduce the earlier E3 numbers with this code (FP dose cells against the uniform-FP cells, seeds 0-4)
    UNI = ["sim00fn20", "sim10fn20", "sim20fn20", "sim30fn20", "sim40fn20"]; CELLS = [("typeB40fn20", range(5)), ("typeB60fn20", range(5)), ("typeA40fn20", range(5))]; USEEDS = range(5)
else:
    UNI = ["sim00fn00", "sim00fn20", "sim00fn40", "sim00fn50"]; CELLS = [("fnB80", range(3)), ("fnB50", range(2)), ("fnA80", range(3)), ("fnH80", range(3))]; USEEDS = range(6)
res = {}
for st in ("qwen7", "llama8"):
    probs, H1, HELD = setup("pqa"); TR = sorted(H1); n = len(HELD); F = base32(st, probs, "pqa")
    conf = np.array([F[i]["gagree"] for i in HELD]); order = np.argsort(conf, kind="stable")
    grpB = SP["groupB"] if "groupB" in SP else None
    knowsB = {json.loads(l)["idx"]: json.loads(l)["knows"] for l in open(f"{DATA}/pqa_knows_type.jsonl")}
    Btypes = {probs[i]["prop"] for i in TR if knowsB[i]}; ht = np.array([probs[i]["prop"] in Btypes for i in HELD])
    def rule(a):
        x = np.zeros(n, bool); x[order[:round(a * n)]] = True; return x
    def absn(run, items):
        o = outcome(run, probs, items); return np.array([o[i] == "idk" for i in items])
    def cellruns(g, seeds):
        return [f"{st}_FK20{g}_idkc05_s{s}_pq" for s in seeds if os.path.exists(f"{RL}/{st}_FK20{g}_idkc05_s{s}_pq/eval_rl_ck120.jsonl")]
    def Tv(x, idx=None):
        r = rule(float(x.mean() if idx is None else x[idx].mean())); h = ht
        if idx is not None: x, r, h = x[idx], r[idx], ht[idx]
        return (x[h].mean() - r[h].mean()) - (x[~h].mean() - r[~h].mean())
    uni = {g: [x for x in (absn(r, HELD) for r in cellruns(g, USEEDS)) if x.mean() < 0.95] for g in UNI}
    def ref(a, idx=None):
        pts = sorted((float(np.mean([(x if idx is None else x[idx]).mean() for x in uni[g]])), float(np.mean([Tv(x, idx) for x in uni[g]]))) for g in UNI if uni[g])
        return float(np.interp(a, [p[0] for p in pts], [p[1] for p in pts])) if pts else float("nan")
    def Ev(rr, idx=None):
        return float(np.mean([Tv(x, idx) for x in rr]) - ref(float(np.mean([(x if idx is None else x[idx]).mean() for x in rr])), idx))
    print(f"{st}: uniform reference cells: " + "; ".join(f"{g} n {len(uni[g])} abstention {np.mean([x.mean() for x in uni[g]]):.3f} T {np.mean([Tv(x) for x in uni[g]]):+.3f}" for g in UNI if uni[g]))
    for g, seeds in CELLS:
        runs = cellruns(g, seeds); xs = [absn(r, HELD) for r in runs]
        if not xs: print(f"  {st} {g}: no finished runs"); continue
        bs = []
        for _ in range(NB):
            idx = rng.integers(0, n, n); bs.append(Ev([xs[k] for k in rng.integers(0, len(xs), len(xs))], idx))
        lo, hi = np.percentile(bs, [5, 95]); e = Ev(xs); res[(st, g)] = (e, lo, hi, len(xs))
        line = f"  {st} {g}: n {len(xs)}; new-question abstention {np.mean([x.mean() for x in xs]):.3f} (group B {np.mean([x[ht].mean() for x in xs]):.3f}, group A {np.mean([x[~ht].mean() for x in xs]):.3f}); T " + ", ".join(f"{Tv(x):+.3f}" for x in xs) + f"; E = {e:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        tb = np.array([knowsB[i] for i in TR]); xt = [absn(r, TR) for r in runs]
        line += f" | trained: group B {np.mean([x[tb].mean() for x in xt]):.3f}, group A {np.mean([x[~tb].mean() for x in xt]):.3f}"
        if g.startswith("fnH"):
            hh = np.array([HID[i] for i in TR]); hn = np.array([HID[i] for i in HELD])
            home = [float(x[hh].mean() - x[~hh].mean()) for x in xt]; new = [float(x[hn].mean() - x[~hn].mean()) for x in xs]
            res[(st, g, "home")] = float(np.mean(home)); res[(st, g, "new")] = float(np.mean(new))
            line += f" | HIDDEN halves: HOME " + ", ".join(f"{v:+.3f}" for v in home) + f" (mean {np.mean(home):+.3f}); NEW " + ", ".join(f"{v:+.3f}" for v in new) + f" (mean {np.mean(new):+.3f})"
        print(line)
if not SELF:
    S = ("qwen7", "llama8"); g = lambda k: res.get(k)
    f1 = [g((s, "fnB80")) is not None and g((s, "fnA80")) is not None and g((s, "fnB80"))[1] > 0 and g((s, "fnA80"))[2] < 0 for s in S]
    f2 = [g((s, "fnB80")) is not None and g((s, "fnB50")) is not None and g((s, "fnB80"))[0] > g((s, "fnB50"))[0] for s in S]
    f3 = [g((s, "fnH80", "home")) is not None and g((s, "fnH80", "home")) >= 0.15 for s in S]; f4 = [g((s, "fnH80", "new")) is not None and abs(g((s, "fnH80", "new"))) <= 0.05 for s in S]
    print(f"\nLINES: F1 visible (E(B80) interval above 0 and E(A80) interval below 0): {sum(f1)} of 2 students; F2 dose (E(B80) > E(B50)): {sum(f2)} of 2; F3 hidden at home (>= +0.15): {sum(f3)} of 2; F4 hidden on new questions (|contrast| <= 0.05): {sum(f4)} of 2 (each line needs 2 of 2)")
