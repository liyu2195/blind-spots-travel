"""(answers) Per dataset x student: paid/unpaid selection ratio for key-wrong sample strings with count 2-7 of 32 (own grader), from a19_rows.pkl.
Output a19c_out.txt."""
from au import *
rows = pickle.load(open(f"{HERE}/a19_rows.pkl", "rb"))
with open(f"{HERE}/a19c_out.txt", "w") as fo:
    def pr(*a):
        print(*a); print(*a, file=fo)
    pr("Paid/unpaid selection ratio for key-wrong sample strings with count 2-7 of 32 (own grader), pooled runs, per dataset x student")
    for f in ("hash", "judge", "sonnet"):
        for dsn in SETS:
            line = "  %-6s %-4s" % (f, dsn)
            for st in STS:
                for part in ("train", "new"):
                    s = [x for x in rows if x["fam"] == f and x["dsn"] == dsn and x["st"] == st and x["part"] == part]
                    ph = sum(x["loc"].get((k, True), [0, 0])[0] for x in s for k in (1, 2)); pt = sum(x["loc"].get((k, True), [0, 0])[1] for x in s for k in (1, 2))
                    uh = sum(x["loc"].get((k, False), [0, 0])[0] for x in s for k in (1, 2)); ut = sum(x["loc"].get((k, False), [0, 0])[1] for x in s for k in (1, 2))
                    if pt and ut and uh: line += " | %s %s %.1f%%/%.1f%% = %5.1fx" % (st, part, 100 * ph / pt, 100 * uh / ut, (ph / pt) / (uh / ut))
            pr(line)
