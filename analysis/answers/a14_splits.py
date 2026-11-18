"""(answers) Discovery (qwen7) / confirmation (llama8) splits with bootstrap CIs over runs or run pairs, for:
 (a) provenance and payment of CHANGED key-wrong final answers (from a3_rows.pkl);
 (b) seed convergence on the identical installed wrong string, same grader vs hash / random graders (from a4_pairs.pkl; the pairs
     there are same student x dataset, so the student split is clean);
 (c) per dataset (both students) for (b).
Output a14_out.txt."""
from au import *
A3 = pickle.load(open(f"{HERE}/a3_rows.pkl", "rb")); A4 = pickle.load(open(f"{HERE}/a4_pairs.pkl", "rb"))
with open(f"{HERE}/a14_out.txt", "w") as fo:
    def pr(*a):
        print(*a); print(*a, file=fo)
    pr("(a) CHANGED key-wrong final answers (x != run's ck0): per-run means [95% bootstrap CI over runs]")
    for f in ("hash", "judge", "sonnet", "keyFP", "gold"):
        for st in STS:
            for part in ("train", "new"):
                sel = [x for x in A3 if x["fam"] == f and x["st"] == st and x["part"] == part and not x["g0"]]
                if not sel: continue
                byrun = defaultdict(list)
                for x in sel: byrun[x["run"]].append(x)
                def per(fn):
                    v = [fn(u) for u in byrun.values()]; v = [a for a in v if a is not None and not np.isnan(a)]
                    return (np.mean(v), *boot_ci(v)) if v else (np.nan, np.nan, np.nan)
                pd = per(lambda u: np.mean([x["paid"] for x in u if x["paid"] is not None]) if any(x["paid"] is not None for x in u) else None)
                bp = per(lambda u: np.mean([x["base_paid"] for x in u if x["base_paid"] is not None]) if any(x["base_paid"] is not None for x in u) else None)
                i32 = per(lambda u: np.mean([x["in32"] for x in u])); nov = per(lambda u: np.mean([not x["inC"] for x in u]))
                pr("  %-6s %-6s %-5s runs %3d n/run %5.1f | paid %.3f [%.3f,%.3f] base %.3f | in32 %.3f [%.3f,%.3f] | novel %.3f [%.3f,%.3f]" % (f, st, part, len(byrun),
                    np.mean([len(u) for u in byrun.values()]), *pd, bp[0], *i32, *nov))
    pr("\n(b) Same-student run pairs: P(identical string | both runs installed a NEW key-wrong answer) [CI over pairs]; coll = chance level")
    for f in ("judge", "sonnet", "gold", "keyFP", "lenQ", "hash"):
        for st in STS:
            for part in ("train", "new"):
                s = [x for x in A4 if x["kind"] == "seed" and x["f1"] == f and x["st"] == st and x["part"] == part and not np.isnan(x["sameNewW"])]
                if not s: continue
                v = [x["sameNewW"] for x in s]; lo, hi = boot_ci(v)
                pr("  seed  %-7s %-6s %-5s pairs %3d  sameNewW %.3f [%.3f,%.3f]  nNewW/pair %.1f  coll %.3f  kIDK %.3f" % (f, st, part, len(s), np.mean(v), lo, hi,
                    np.mean([x["nNewW"] for x in s]), np.nanmean([x["coll"] for x in s]), np.nanmean([x["kIDK"] for x in s])))
    for lab, cond in (("cross judge-vs-other-judge", lambda x: x["f1"] == x["f2"] == "judge" and judge_of(x["g1"])[0] != judge_of(x["g2"])[0]),
                      ("cross judge-vs-gold", lambda x: {x["f1"], x["f2"]} == {"judge", "gold"}),
                      ("cross judge-vs-hash", lambda x: {x["f1"], x["f2"]} == {"judge", "hash"})):
        for st in STS:
            for part in ("train", "new"):
                s = [x for x in A4 if x["kind"] == "cross" and cond(x) and x["st"] == st and x["part"] == part and not np.isnan(x["sameNewW"])]
                if not s: continue
                v = [x["sameNewW"] for x in s]; lo, hi = boot_ci(v)
                pr("  %-27s %-6s %-5s pairs %3d  sameNewW %.3f [%.3f,%.3f]  kIDK %.3f" % (lab, st, part, len(s), np.mean(v), lo, hi, np.nanmean([x["kIDK"] for x in s])))
    pr("\n(c) Per dataset, both students: seed sameNewW on TRAINING questions")
    for dsn in SETS:
        line = "  %-4s" % dsn
        for f in ("judge", "sonnet", "keyFP", "lenQ", "hash"):
            s = [x["sameNewW"] for x in A4 if x["kind"] == "seed" and x["f1"] == f and x["dsn"] == dsn and x["part"] == "train" and not np.isnan(x["sameNewW"])]
            if s: line += "  %s %.3f (n=%d)" % (f, np.mean(s), len(s))
        pr(line)
