"""(answers) Do different seeds of the same grader converge on the same final answers / refusals, question by question? Compared with
pairs of runs that differ in grader. FK20 idkc05 runs. For each pair of runs (same student x dataset):
  agree   = share of questions with the identical final normalized string;
  kIDK    = Cohen's kappa of the IDK indicator;
  sameNew = among questions where BOTH runs moved away from their untrained greedy to a non-IDK answer, share with the identical string;
  sameNewW= the same restricted to both-final-wrong (key) answers;
  coll    = chance expectation for sameNew: collision probability of the untrained samples' non-greedy, non-IDK strings
            (sum p^2 over strings, renormalised), averaged over the same questions.
Output a4_out.txt, a4_pairs.pkl."""
import itertools
from au import *


def pairstats(r1, r2, items, al, B):
    a = [r1["a120"][i] for i in items]; b = [r2["a120"][i] for i in items]
    agree = np.mean([x == y for x, y in zip(a, b)])
    k = kappa([is_idk(x) for x in a], [is_idk(y) for y in b])
    sn, snw, coll = [], [], []
    for i, x, y in zip(items, a, b):
        if x and y and not is_idk(x) and not is_idk(y) and x != r1["a0"][i] and y != r2["a0"][i]:
            sn.append(x == y)
            if code(x, al[i]) == 2 and code(y, al[i]) == 2: snw.append(x == y)
            g = B["g"][i]; smp = [z for z in B["s"][i] if z and not is_idk(z) and z != g]
            if smp:
                c = Counter(smp); n = sum(c.values()); coll.append(sum((v / n) ** 2 for v in c.values()))
    return dict(agree=agree, kIDK=k, sameNew=np.mean(sn) if sn else np.nan, nNew=len(sn), sameNewW=np.mean(snw) if snw else np.nan,
                nNewW=len(snw), coll=np.mean(coll) if coll else np.nan, idk1=np.mean([is_idk(x) for x in a]), idk2=np.mean([is_idk(y) for y in b]))

out = []
for st in STS:
    for dsn in SETS:
        P = load(st, dsn); Q = P["Q"]; al = Q["aliases"]; B = P["B"]
        parts = {"train": sorted(Q["H1"]), "new": Q["HELD"]}
        names = runs(P)
        byg = defaultdict(list)
        for n in names: byg[P["R"][n]["g"]].append(n)
        # same grader, different seeds
        for g, ns in byg.items():
            for n1, n2 in itertools.combinations(sorted(ns), 2):
                for part, items in parts.items():
                    out.append(dict(st=st, dsn=dsn, kind="seed", g1=g, g2=g, f1=fam(g), f2=fam(g), part=part, **pairstats(P["R"][n1], P["R"][n2], items, al, B)))
        # different graders: one pair per (g1, g2) using the lowest seeds (s0 vs s0 when available, else lowest two)
        gs = sorted(byg)
        for g1, g2 in itertools.combinations(gs, 2):
            f1, f2 = fam(g1), fam(g2)
            keep = {"gold", "judge", "hash", "keyFP", "sonnet", "keyFN", "lenQ", "mix"}
            if f1 not in keep or f2 not in keep: continue
            n1 = sorted(byg[g1], key=lambda n: P["R"][n]["s"])[0]; n2 = sorted(byg[g2], key=lambda n: P["R"][n]["s"])[0]
            if P["R"][n1]["s"] == P["R"][n2]["s"] and len(byg[g2]) > 1:   # avoid sharing the seed (question order) when possible
                n2 = sorted(byg[g2], key=lambda n: P["R"][n]["s"])[1]
            for part, items in parts.items():
                out.append(dict(st=st, dsn=dsn, kind="cross", g1=g1, g2=g2, f1=f1, f2=f2, part=part, **pairstats(P["R"][n1], P["R"][n2], items, al, B)))
pickle.dump(out, open(f"{HERE}/a4_pairs.pkl", "wb"))
with open(f"{HERE}/a4_out.txt", "w") as fo:
    def pr(*a):
        print(*a); print(*a, file=fo)
    pr("Pairwise agreement of final greedy answers. Means over pairs (pooled over students x datasets). n = pairs.")
    pr("%-34s %-5s %4s %6s %6s %7s %6s %7s %6s %6s" % ("pair type", "part", "n", "agree", "kIDK", "sameNew", "nNew", "sameNwW", "coll", "idk"))
    def show(label, sel):
        for part in ("train", "new"):
            s = [x for x in sel if x["part"] == part]
            if not s: continue
            m = lambda k: np.nanmean([x[k] for x in s])
            pr("%-34s %-5s %4d %6.3f %6.3f %7.3f %6.1f %7.3f %6.3f %6.3f" % (label, part, len(s), m("agree"), m("kIDK"), m("sameNew"), m("nNew"), m("sameNewW"), m("coll"), (m("idk1") + m("idk2")) / 2))
    for f in ("gold", "keyFN", "keyFP", "mix", "hash", "judge", "sonnet", "lenQ"):
        show(f"seed pairs: {f}", [x for x in out if x["kind"] == "seed" and x["f1"] == f])
    # judge seeds split by judge and level
    for J in "QMGO":
        for lv in ("a35", "a50", "t50"):
            sel = [x for x in out if x["kind"] == "seed" and x["g1"] == f"judge{J}{lv}"]
            if sel: show(f"seed pairs: judge{J}{lv}", sel)
    show("cross: judge vs other judge (any lvl)", [x for x in out if x["kind"] == "cross" and x["f1"] == "judge" and x["f2"] == "judge" and judge_of(x["g1"])[0] != judge_of(x["g2"])[0]])
    show("cross: judge a35 vs other judge a35", [x for x in out if x["kind"] == "cross" and x["f1"] == "judge" and x["f2"] == "judge" and judge_of(x["g1"])[0] != judge_of(x["g2"])[0] and judge_of(x["g1"])[1] == judge_of(x["g2"])[1] == "a35" and not judge_of(x["g1"])[2] and not judge_of(x["g2"])[2]])
    show("cross: same judge, a35 vs a50", [x for x in out if x["kind"] == "cross" and x["f1"] == "judge" and x["f2"] == "judge" and judge_of(x["g1"])[0] == judge_of(x["g2"])[0] and {judge_of(x["g1"])[1], judge_of(x["g2"])[1]} == {"a35", "a50"}])
    show("cross: judge vs gold", [x for x in out if x["kind"] == "cross" and {x["f1"], x["f2"]} == {"judge", "gold"}])
    show("cross: judge vs keyFP", [x for x in out if x["kind"] == "cross" and {x["f1"], x["f2"]} == {"judge", "keyFP"}])
    show("cross: judge vs hash", [x for x in out if x["kind"] == "cross" and {x["f1"], x["f2"]} == {"judge", "hash"}])
    show("cross: judge vs sonnet", [x for x in out if x["kind"] == "cross" and {x["f1"], x["f2"]} == {"judge", "sonnet"}])
    show("cross: hash vs keyFP", [x for x in out if x["kind"] == "cross" and {x["f1"], x["f2"]} == {"hash", "keyFP"}])
    show("cross: gold vs keyFP", [x for x in out if x["kind"] == "cross" and {x["f1"], x["f2"]} == {"gold", "keyFP"}])
    show("cross: gold vs keyFN", [x for x in out if x["kind"] == "cross" and {x["f1"], x["f2"]} == {"gold", "keyFN"}])
    show("cross: sonnet vs gold", [x for x in out if x["kind"] == "cross" and {x["f1"], x["f2"]} == {"sonnet", "gold"}])
