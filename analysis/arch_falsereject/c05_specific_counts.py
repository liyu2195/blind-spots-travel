"""(arch_falsereject) Judge-SPECIFIC hacks and refusals of known answers, per 100 questions, home vs new, by threshold level. Within-run contrast against
each other judge J' at the same level (removes answering-rate / leniency confounds), minus the same contrast in key-trained (gold) runs (placebo).
  hack_spec(J,J')  = #changed key-wrong final answers paid by J and not J'  -  #paid by J' and not J
  refK_spec(J,J')  = #untrained key-right greedy answers rejected by J, accepted by J', refused at the end  -  #rejected by J', accepted by J, refused
Per run: mean over J' in {Q,M,G,O} minus J, then minus the gold-run mean of the same quantity (same J, level, pair). Units: per 100 questions of the part.
Discovery split = qwen7, confirmation = llama8 (reported separately). Inputs: answers/ caches. Usage: per dataset 'nice -n 10 timeout 280 python3 c05_specific_counts.py <tqa|pqa|nqo>' (writes c05_rows_<dsn>.pkl), then 'python3 c05_specific_counts.py sum > c05_specific_counts_out.txt' to pool the three rows files and print"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, numpy as np, pandas as pd
sys.path.insert(0, (_RA + "/answers"))
from au import load, jc, judge_of, code, thr, runs, STS, SETS
rng = np.random.default_rng(9); JS = ("Q", "M", "G", "O"); rows = []
SETS_RUN = [sys.argv[1]] if (len(sys.argv) > 1 and sys.argv[1] in SETS) else []
for dsn in SETS_RUN:
    JC = {J: jc(dsn, J) for J in JS}
    for st in STS:
        P = load(st, dsn); Q = P["Q"]; al = Q["aliases"]; parts = {"home": sorted(Q["H1"]), "new": Q["HELD"]}
        T = {(J, l): thr(st, dsn, J, l) for J in JS for l in ("t50", "a50", "a35")}
        def acc(J, l, i, x):
            v = JC[J].get((i, x)); return None if v is None else v > T[(J, l)]
        def spec(r, J, l):
            out = {}
            for part, items in parts.items():
                hs = []; rs = []
                for Jp in JS:
                    if Jp == J: continue
                    h = 0; rr = 0
                    for i in items:
                        a0 = r["a0"].get(i); a1 = r["a120"].get(i); c0 = code(a0, al[i]); c1 = code(a1, al[i])
                        if c1 == 2 and a1 != a0:
                            u, w = acc(J, l, i, a1), acc(Jp, l, i, a1)
                            if u is not None and w is not None: h += (u and not w) - (w and not u)
                        if c0 == 1 and c1 == 0:
                            u, w = acc(J, l, i, a0), acc(Jp, l, i, a0)
                            if u is not None and w is not None: rr += ((not u) and w) - ((not w) and u)
                    hs.append(100 * h / len(items)); rs.append(100 * rr / len(items))
                out[part] = (np.mean(hs), np.mean(rs))
            return out
        golds = [P["R"][g] for g in runs(P, fams={"gold"})]; GC = {}
        for name in runs(P, fams={"judge"}):
            r = P["R"][name]; jo = judge_of(r["g"])
            if jo is None or jo[0] not in JS or jo[2]: continue
            J, l = jo[0], jo[1]
            if (J, l) not in GC: GC[(J, l)] = [spec(g, J, l) for g in golds]
            s = spec(r, J, l)
            for part in parts:
                gh = np.mean([g[part][0] for g in GC[(J, l)]]); gr = np.mean([g[part][1] for g in GC[(J, l)]])
                rows.append(dict(st=st, dsn=dsn, J=J, lvl=l, part=part, run=name, hack=s[part][0] - gh, ref=s[part][1] - gr, hack_raw=s[part][0], ref_raw=s[part][1], hack_g=gh, ref_g=gr))
D = pd.DataFrame(rows)
if SETS_RUN: D.to_pickle(f"c05_rows_{sys.argv[1]}.pkl"); sys.exit(0)
D = pd.concat([pd.read_pickle(f"c05_rows_{d}.pkl") for d in SETS]); D.to_pickle("c05_rows.pkl")
def ci(v):
    v = np.asarray(v); bs = [rng.choice(v, len(v)).mean() for _ in range(2000)]; return f"{v.mean():+5.2f} [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}]"
for who, E0 in (("both", D), ("qwen7 (discovery)", D[D.st == "qwen7"]), ("llama8 (confirmation)", D[D.st == "llama8"])):
    print(f"== {who}")
    for l in ("t50", "a50", "a35"):
        for part in ("home", "new"):
            E = E0[(E0.lvl == l) & (E0.part == part)]
            if len(E): print(f"  {l} {part:4s} runs {len(E):3d} | judge-specific hacks/100q {ci(E.hack)} | judge-specific refusals of known answers/100q {ci(E.ref)} | gold placebo hacks {E.hack_g.mean():+.2f} refusals {E.ref_g.mean():+.2f}")
print("\nper dataset, a35 and t50, both students:")
for dsn in SETS:
    for l in ("t50", "a35"):
        s = []
        for part in ("home", "new"):
            E = D[(D.dsn == dsn) & (D.lvl == l) & (D.part == part)]
            if len(E): s.append(f"{part} hacks {E.hack.mean():+.2f} refusals {E.ref.mean():+.2f} (runs {len(E)})")
        print(f"  {dsn} {l}: " + " | ".join(s))
