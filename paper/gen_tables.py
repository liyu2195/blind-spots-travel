#!/usr/bin/env python3
"""Generate appendix tables from the analysis outputs and released run inventory.
Usage: python3 -I gen_tables.py   -> tabs/*.tex  (and tabs/check.txt, a plain-text copy for checking)
Sources (read only):
  analysis/fam_fix/e9b_forced_out.txt      judges at each cut (PopQA)
  runs/run1/rl/<run>/eval_rl_ck120.jsonl   run inventory (finished runs)
  analysis/fix_ladder/out/ladder_<cut>_final_*.json   judge contrast, costs
  analysis/fam_fix/e15_count_dose.json     rejected training examples and refusals on the probe
  analysis/fam_fix/e16_caricature.json     model against its own judge on the probe
  analysis/fam_fix/e17_mirror_out.txt      shared rejections
  analysis/fix_sc/f3_table.txt             self-consistency
"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import json, os, re, collections
import numpy as np

W = _RA
RL = (_RR + "/run1/rl")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tabs")
os.makedirs(OUT, exist_ok=True)
CHK = []


def d2(x):
    """0.468 -> .47 ; negative with a minus sign"""
    if x is None:
        return "--"
    s = f"{abs(x):.2f}"
    s = s[1:] if s.startswith("0") else s
    return ("$-$" if x < -0.004999 else "") + s


def sg(x, nd=2):
    return ("$+$" if x >= 0 else "$-$") + f"{abs(x):.{nd}f}"


def ci(lo, hi, nd=2):
    f = lambda v: ("$-$" if v < 0 else "") + f"{abs(v):.{nd}f}"
    return f"[{f(lo)}, {f(hi)}]"


def write(name, body):
    open(os.path.join(OUT, name), "w").write(body)
    CHK.append(f"==== {name}\n{body}\n")


JN = {"Q": "Qwen3.5", "M": "Mistral-Nemo", "G": "Gemma-4", "O": "OLMo-2", "R": "Granite-3.3"}
CUTN = {"t50": "default", "key": "key-matched", "acc": "key-matched", "a50": "50\\%", "a35": "35\\%"}

# ---------------------------------------------------------------- 1. judges at each cut (PopQA)
rows = {}
st = None
for line in open(f"{W}/fam_fix/e9b_forced_out.txt"):
    m = re.match(r"== (\w+) pq: pool (\d+) answers; right by strict key ([\d.]+), loose key ([\d.]+), key or audit ([\d.]+)", line)
    if m:
        st = m.group(1)
        rows[("pool", st)] = (int(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5)))
        continue
    if line.startswith("== "):
        st = None      # another dataset: not part of this table
        continue
    m = re.match(r"\s+judge (\w) (\w+) \(P > ([\d.]+)\): accepts ([\d.]+) \| least share of right answers rejected: strict key ([\d.]+), loose key "
                 r"([\d.]+), key or audit ([\d.]+) \| judge rejects ([\d.]+) of audited-right and accepts ([\d.]+) of audited-wrong", line)
    if m and st:
        J, cut = m.group(1), m.group(2)
        rows[(J, cut, st)] = tuple(float(m.group(i)) for i in (3, 4, 7, 8, 9))
b = []
for J in "QMGO":
    for k, cut in enumerate(["t50", "key", "a50", "a35"]):
        q, l = rows[(J, cut, "qwen7")], rows[(J, cut, "llama8")]
        cells = [JN[J] if k == 0 else "", CUTN[cut]]
        for r in (q, l):
            cells += [f"{r[0]:.3f}", d2(r[1]), d2(r[2]), d2(r[3]), d2(r[4])]
        b.append(" & ".join(cells) + " \\\\")
    if J != "O":
        b.append("\\addlinespace")
write("judges.tex", "\n".join(b))
write("judges_pool.txt", json.dumps({k[1]: v for k, v in rows.items() if k[0] == "pool"}))

# ---------------------------------------------------------------- 2. run inventory
FAM = [
    ("Answer key", lambda t, g: t == "FK20" and g == "sim00fn00"),
    ("Answer key with random errors", lambda t, g: t == "FK20" and re.fullmatch(r"sim\d\dfn\d\d", g) and g != "sim00fn00"),
    ("Random grader at the Qwen judge's error rates", lambda t, g: t == "FK20" and g == "lenQ"),
    ("Reference-free judges, default threshold", lambda t, g: t == "FK20" and re.fullmatch(r"judge[QMGO]t50(f1|q3|q5)?", g)),
    ("Reference-free judges, key-matched cut", lambda t, g: t == "FK20" and re.fullmatch(r"judge[QMGO]acc", g)),
    ("Reference-free judges, 50\\% cut", lambda t, g: t == "FK20" and re.fullmatch(r"judge[QMGO]a50", g)),
    ("Reference-free judges, 35\\% cut", lambda t, g: t == "FK20" and re.fullmatch(r"judge[QMGOR]a35", g)),
    ("Reference-guided judge (default; 35\\% cut)", lambda t, g: t == "FK20" and g in ("judgeQRt50", "judgeQRa35")),
    ("Sonnet as judge", lambda t, g: t == "FK20" and g == "apiS55"),
    ("Majority of five judges", lambda t, g: t == "FK20" and g == "frzMAJ35"),
    ("Copies of a judge's verdicts (copy, scrambled, by type)", lambda t, g: t == "FK20" and (g.startswith("pl") or re.fullmatch(r"frz(Q35|Q50k\d+|S55k\d+|PEER1|OR\d+)", g))),
    ("Student's own vote", lambda t, g: t == "FK20" and (g.startswith("frzVOTE") or g.startswith("voteq"))),
    ("Rejections placed on relation types or popularity", lambda t, g: t == "FK20" and re.fullmatch(r"(fn[ABH]\d\d|type[AB]\d\dfn\d\d|pop[HL]\d\dfn\d\d)", g)),
    ("Hash-placed paid strings", lambda t, g: t == "FK20" and g.startswith("hash")),
    ("Long answers paid", lambda t, g: t == "FK20" and g == "long80"),
    ("Any answer paid on a share of questions", lambda t, g: t == "FK20" and g.startswith("blind")),
    ("Visits per question varied (1, 2, 5, 10)", lambda t, g: re.fullmatch(r"B\d+v?K20", t)),
    ("KL weight or run length varied", lambda t, g: t in ("FK00", "FK50", "FK100", "LK20", "LK50")),
    ("Unnormalised advantages, credit varied", lambda t, g: t == "FK20D"),
]
cnt = collections.defaultdict(collections.Counter)
tot = collections.Counter()
small = collections.Counter()
unassigned = collections.Counter()
for r in sorted(os.listdir(RL)):
    m = re.match(r"^(qwen7|llama8|phi4m|smol3|granite)_(.+)_(idkc|flagc)(\d+)_s(\d+)_(\w+)$", r)
    if not m or not os.path.exists(f"{RL}/{r}/eval_rl_ck120.jsonl"):
        continue
    stu, mid, ds = m.group(1), m.group(2), m.group(6)
    if stu not in ("qwen7", "llama8"):
        small[stu] += 1
        continue
    mm = re.match(r"^([A-Z0-9]+?v?K\d+D?)(.+)$", mid)
    if not mm:
        unassigned[mid] += 1
        continue
    tag, g = mm.group(1), mm.group(2)
    tot[stu] += 1
    for name, f in FAM:
        if f(tag, g):
            cnt[name][stu] += 1
            break
    else:
        unassigned[f"{tag}|{g}"] += 1
b = []
for name, _ in FAM:
    b.append(f"{name} & {cnt[name]['qwen7']} & {cnt[name]['llama8']} \\\\")
oth = {s: tot[s] - sum(cnt[n][s] for n, _ in FAM) for s in ("qwen7", "llama8")}
b.append(f"Other graders & {oth['qwen7']} & {oth['llama8']} \\\\")
b.append("\\midrule")
b.append(f"All finished runs & {tot['qwen7']} & {tot['llama8']} \\\\")
write("runs.tex", "\n".join(b))
write("runs_small.txt", json.dumps({"small": dict(small), "unassigned": dict(unassigned), "tot": dict(tot)}))

# ---------------------------------------------------------------- 3. judge contrast by cut
L = {c: json.load(open(f"{W}/fix_ladder/out/ladder_{c}_final_{'1117d' if c == 'acc' else '1117b'}.json"))[c] for c in ("t50", "acc", "a50", "a35")}
b = []
names = [("pooled", "Both students, all datasets"), ("qwen7", "Qwen student"), ("llama8", "Llama student"), ("h1", "TriviaQA"), ("pq", "PopQA"),
         ("nq", "NQ-open"), ("qwen7@pq", "PopQA, Qwen student"), ("llama8@pq", "PopQA, Llama student")]
for key, lab in names:
    cells = [lab]
    for c in ("t50", "a50", "a35"):
        s = L[c]["X_FR"]["summary"].get(key)
        cells.append("--" if s is None else f"{sg(s['X'])} {ci(*s['ci'])}")
    b.append(" & ".join(cells) + " \\\\")
cells = ["Judge pairs with a positive contrast"]
for c in ("t50", "a50", "a35"):
    v = [x[0] for x in L[c]["X_FR"]["cells"].values()]
    cells.append(f"{sum(1 for x in v if x > 0)} of {len(v)}")
b.append("\\midrule")
b.append(" & ".join(cells) + " \\\\")
write("contrast.tex", "\n".join(b))
s = L["acc"]["X_FR"]["summary"]["llama8@pq"]
v = [x[0] for x in L["acc"]["X_FR"]["cells"].values()]
write("contrast_acc.txt", f"key-matched llama pq: {s['X']:.3f} {s['ci']} cells {len(v)} pos {sum(1 for x in v if x > 0)}")

# wrong-answer mirror (NQ) at the 35% cut
s = L["a35"]["X_FA"]["summary"]
v = [x[0] for x in L["a35"]["X_FA"]["cells"].values()]
write("mirror_fa.txt", json.dumps({k: [round(s[k]["X"], 3), [round(a, 3) for a in s[k]["ci"]]] for k in s}) + f" cells {len(v)} neg {sum(1 for x in v if x < 0)}")

# content transfer by cut
b = []
for c in ("t50", "a50", "a35"):
    cp = L[c]["content_panel"]
    cells = [CUTN[c].replace("default", "Default").replace("50\\%", "50\\% cut").replace("35\\%", "35\\% cut")]
    for k in ("pooled",):
        cells += [f"{cp[k]['runs_new']}", f"{cp[k]['home']:.2f}", f"{cp[k]['new']:.2f}",
                  f"{cp[k]['ratio']:.2f} {ci(*cp[k]['ci_ratio'])}"]
    b.append(" & ".join(cells) + " \\\\")
write("content.tex", "\n".join(b))

# costs per judge and cut, PopQA
b = []
for J in "QMGO":
    for k, c in enumerate(("t50", "acc", "a50", "a35")):
        cells = [JN[J] if k == 0 else "", CUTN[c]]
        for stu in ("qwen7", "llama8"):
            v = L[c]["costs"].get(f"{J}|{stu}|pq")
            if v is None:
                cells += ["--", "--", "--"]
            else:
                cells += [f"{v['R1'][0]:.1f}", f"{v['RW'][0]:.1f}", d2(v['inh_share'][0])]
        b.append(" & ".join(cells) + " \\\\")
    if J != "O":
        b.append("\\addlinespace")
write("costs.tex", "\n".join(b))

# ---------------------------------------------------------------- 4. rejected training examples and probe refusals
E = json.load(open(f"{W}/fam_fix/e15_count_dose.json"))
idx = {(e["st"], e["label"]): e for e in E}
b = []
for J in "QMGO":
    for k, c in enumerate(("t50", "acc", "a50", "a35")):
        cells = [JN[J] if k == 0 else "", CUTN[c]]
        for stu in ("qwen7", "llama8"):
            e = idx.get((stu, f"judge {J} {c}"))
            if e is None:
                cells += ["--"] * 5
            else:
                cells += [f"{e['kF']}", f"{e['runs']}", d2(e["abst"]), d2(e["RF"]), d2(e["RP"]), d2(e["cF"])][0:1] + [d2(e["abst"]), d2(e["RF"]), d2(e["RP"]), d2(e["cF"])]
        b.append(" & ".join(cells) + " \\\\")
    b.append("\\addlinespace")
for lab, tex in (("answer key", "Answer key"),):
    cells = [tex, ""]
    for stu in ("qwen7", "llama8"):
        e = idx[(stu, lab)]
        cells += [f"{e['kF']}", d2(e["abst"]), d2(e["RF"]), d2(e["RP"]), d2(e["cF"])]
    b.append(" & ".join(cells) + " \\\\")
for stu_labs, tex in (({"qwen7": "frozen key frzVOTE21", "llama8": "frozen key frzVOTE29"}, "Own vote, 21 / 29 of 32"),
                      ({"qwen7": "frozen key frzVOTE32", "llama8": "frozen key frzVOTE32"}, "Own vote, 32 of 32")):
    cells = ["\\multicolumn{2}{@{}l}{" + tex + "}"]
    for stu in ("qwen7", "llama8"):
        e = idx[(stu, stu_labs[stu])]
        cells += [f"{e['kF']}", d2(e["abst"]), d2(e["RF"]), d2(e["RP"]), d2(e["cF"])]
    b.append(" & ".join(cells) + " \\\\")
# own vote at the cut of 20 (all runs at that cut: three with Qwen, six with Llama; e15 rerun on Nov 18)
cells = ["\\multicolumn{2}{@{}l}{Own vote, 20 of 32}"]
for stu in ("qwen7", "llama8"):
    e = idx[(stu, "frozen key frzVOTE20")]
    cells += [f"{e['kF']}", d2(e["abst"]), d2(e["RF"]), d2(e["RP"]), d2(e["cF"])]
b.append(" & ".join(cells) + " \\\\")
write("dose.tex", "\n".join(b).replace("Answer key & ", "\\multicolumn{2}{@{}l}{Answer key} "))
write("dose_runs.txt", json.dumps({f"{e['st']}|{e['label']}": e["runs"] for e in E}))

# ---------------------------------------------------------------- 5. the model against its own judge
Cc = json.load(open(f"{W}/fam_fix/e16_caricature.json"))
ix = {(e["st"], e["J"], e["cut"]): e for e in Cc}
b = []
for J in "QMGO":
    for k, c in enumerate(("acc", "a50", "a35")):
        cells = [JN[J] if k == 0 else "", CUTN[c]]
        for stu in ("qwen7", "llama8"):
            e = ix.get((stu, J, c))
            if e is None:
                cells += ["--"] * 6
            else:
                cells += [d2(e["jrejF"]), d2(e["mF"]), f"{d2(e['F_acc'][0])} ({e['F_acc'][1]})", d2(e["jrejP"]), d2(e["mP"]),
                          f"{d2(e['P_rej'][0])} ({e['P_rej'][1]})"]
        b.append(" & ".join(cells) + " \\\\")
    if J != "O":
        b.append("\\addlinespace")
write("caric.tex", "\n".join(b))

# ---------------------------------------------------------------- 6. shared rejections
M = {}
cur = None
for line in open(f"{W}/fam_fix/e17_mirror_out.txt"):
    m = re.match(r"== (\w+) (\w+): right untrained greedy answers on training questions (\d+); unanimous \(conf = 1\) ([\d.]+)", line)
    if m:
        cur = (m.group(1), m.group(2))
        M[("n",) + cur] = (int(m.group(3)), float(m.group(4)))
        continue
    m = re.match(r"\s+judge (\w): rejects ([\d.]+) of right answers \| student confidence predicts the verdict AUC ([\d.]+); rejects ([\d.]+) of unanimous "
                 r"and ([\d.]+) of non-unanimous right answers \| another judge's score predicts it AUC ([\d.]+); of its rejections ([\d.]+) are rejected "
                 r"by >= 2 of the other 3 judges \(of its acceptances ([\d.]+)\)", line)
    if m and cur:
        M[(m.group(1),) + cur] = tuple(float(m.group(i)) for i in range(2, 9))
b = []
for stu, sn in (("qwen7", "Qwen"), ("llama8", "Llama")):
    for k, J in enumerate("QMGO"):
        cells = [sn if k == 0 else "", JN[J]]
        for ds in ("pq", "h1", "nq"):
            r = M[(J, stu, ds)]
            cells += [d2(r[0]), d2(r[5]), d2(r[6]), d2(r[4]), d2(r[1])]
        b.append(" & ".join(cells) + " \\\\")
    if stu == "qwen7":
        b.append("\\addlinespace")
write("shared.tex", "\n".join(b))

# ---------------------------------------------------------------- 7. self-consistency
NAMES = {"key": "Answer key", "key+noise (all)": "Answer key with random errors", "own vote": "Student's own vote", "Q t50": "Qwen3.5 judge, default",
         "Q a50": "Qwen3.5 judge, 50\\% cut", "Q a35": "Qwen3.5 judge, 35\\% cut", "M a35": "Mistral-Nemo judge, 35\\% cut",
         "G a35": "Gemma-4 judge, 35\\% cut", "O a35": "OLMo-2 judge, 35\\% cut", "R a35": "Granite-3.3 judge, 35\\% cut", "Sonnet": "Sonnet as judge",
         "planted FR": "Placed rejections of right answers", "planted FA": "Placed paid wrong answers"}
b = []
for line in open(f"{W}/fix_sc/f3_table.txt"):
    if not line.startswith("| ") or line.startswith("| grader") or line.startswith("|---"):
        continue
    p = [x.strip() for x in line.strip().strip("|").split("|")]
    key = p[0]
    if key.startswith("blind"):
        key, nm = "blind", "Any answer paid on 40--60\\% of questions"
    elif key in NAMES:
        nm = NAMES[key]
    else:
        continue
    runs = [x.split("(")[0].strip() for x in p[1].split(";")]
    cov = [x.strip() for x in p[2].split(";")]
    sc1 = [x.strip() for x in p[3].split(";")]
    sc32 = [re.sub(r"†.*$", "", x).strip() for x in p[4].split(";")]
    ms = [x.strip() for x in p[5].split(";")]
    fx = lambda s: s.replace("+", "$+$").replace("-", "$-$").replace(">", "$>$")
    b.append(" & ".join([nm, f"{runs[0]}; {runs[1]}", f"{cov[0]}; {cov[1]}", fx(sc1[0]), fx(sc1[1]), fx(sc32[0]), fx(sc32[1]), fx(ms[0]), fx(ms[1])]) + " \\\\")
# reference-guided judge (default threshold), PopQA + NQ-open: supp_final/s4_sc_reference_out.txt
t4 = open(f"{W}/supp_final/s4_sc_reference_out.txt").read()
qr = {}
for st in ("qwen7", "llama8"):
    mm = re.search(rf"{st}\s+judgeQRt50\s+\[PopQA \+ NQ-open\s*\] runs\s+(\d+)\s+answered ([\d.]+)\s+RL - SC_32 ([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\]", t4)
    qr[st] = mm.groups()
fx = lambda s: s.replace("+", "$+$").replace("-", "$-$")
b.append(" & ".join(["Judge shown the reference, default", f"{qr['qwen7'][0]}; {qr['llama8'][0]}", f"{qr['qwen7'][1]}; {qr['llama8'][1]}", "--", "--",
                     fx(f"{qr['qwen7'][2]} [{qr['qwen7'][3]}, {qr['qwen7'][4]}]"), fx(f"{qr['llama8'][2]} [{qr['llama8'][3]}, {qr['llama8'][4]}]"), "--", "--"]) + " \\\\")
write("sc.tex", "\n".join(b))

# ---------------------------------------------------------------- 8. designed graders on the capitals (supp_final/s3_planted_counts.json, s9 when present)
PL = json.load(open(f"{W}/supp_final/s3_planted_counts.json"))
if os.path.exists(f"{W}/supp_final/s9_dose_planted.json"):
    PL = json.load(open(f"{W}/supp_final/s9_dose_planted.json"))
px = {(r["st"], r["arm"]): r for r in PL}
b = []
ARMS = [("sharpM", "Type-wide copy of Mistral"), ("sharpO", "Type-wide copy of OLMo"), ("scrM", "Scrambled copy of Mistral"), ("realO", "Copy of OLMo"), ("realM", "Copy of Mistral")]
ARMS += [(f"Mk{k}", f"Copy of Mistral, {k} of the twelve rejected") for k in (0, 4, 8, 12) if ("llama8", f"Mk{k}") in px and px[("llama8", f"Mk{k}")].get("runs")]
for arm, lab in ARMS:
    cells = [lab]
    for st in ("qwen7", "llama8"):
        r = px.get((st, arm))
        if not r or not r.get("runs"): cells += ["--"] * 7; continue
        cells += [f"{r['kF']}", d2(r["rO"]), f"{r['runs']}", d2(r["abst"]), d2(r["RF"]), d2(r["RP"]), d2(r["cF"])]
    b.append(" & ".join(cells) + " \\\\")
write("planted.tex", "\n".join(b))

# ---------------------------------------------------------------- 9. return under the training reward (supp_final/s5_return_by_cut.json), PopQA
RT = json.load(open(f"{W}/supp_final/s5_return_by_cut.json"))
rx = {(r["st"], r["g"]): r for r in RT if r["ds"] == "pq"}
b = []
def rrow(lab1, lab2, g):
    cells = [lab1, lab2]
    for st in ("qwen7", "llama8"):
        r = rx.get((st, g))
        cells += (["--"] * 4 if r is None else [d2(r["right"]), d2(r["wrong"]), d2(r["idk"]), f"{r['ret']:.3f}"])
    return " & ".join(cells) + " \\\\"
b.append("\\multicolumn{2}{@{}l}{Untrained model} & " + " & ".join(["--", "--", "--", f"{rx[(st, 'sim00fn00')]['base']:.3f}"][i] for st in ("qwen7", "llama8") for i in range(4)) + " \\\\")
b.append(rrow("\\multicolumn{2}{@{}l}{Answer key}", "", "sim00fn00").replace("} &  & ", "} & ", 1))
b.append("\\addlinespace")
for J in "QMGO":
    for k, c in enumerate(("t50", "acc", "a50", "a35")):
        b.append(rrow(JN[J] if k == 0 else "", CUTN[c], f"judge{J}{c}"))
    b.append("\\addlinespace")
b.append(rrow("\\multicolumn{2}{@{}l}{Majority of five judges, 35\\% cut}", "", "frzMAJ35").replace("} &  & ", "} & ", 1))
b.append(rrow("\\multicolumn{2}{@{}l}{Judge shown the reference, default}", "", "judgeQRt50").replace("} &  & ", "} & ", 1))
b.append(rrow("\\multicolumn{2}{@{}l}{Sonnet as judge}", "", "apiS55").replace("} &  & ", "} & ", 1))
write("returns.tex", "\n".join(b))

# ---------------------------------------------------------------- 10. kinds on fresh questions (supp_final/s8_fresh_kinds.json), 35% cut, judges averaged
K8 = json.load(open(f"{W}/supp_final/s8_fresh_kinds.json"))
agg = collections.defaultdict(list)
for r in K8:
    if r["cut"] == "a35": agg[(r["st"], r["kind"])].append(r)
kinds = sorted({k for (_, k) in agg}, key=lambda k: -np.mean([x["audit"] for x in agg.get(("llama8", k), agg.get(("qwen7", k)))]))
KN = {"country": "country (``In what country is X?'')", "capital": "capital", "father": "father", "screenwriter": "screenwriter", "director": "director",
      "composer": "composer", "genre": "genre", "sport": "sport", "author": "author", "occupation": "occupation"}
b = []
for k in kinds:
    cells = [KN.get(k, k)]
    for st in ("llama8", "qwen7"):
        v = agg.get((st, k))
        if not v: cells += ["--"] * 7; continue
        m = lambda f: float(np.mean([x[f] for x in v]))
        cells += [f"{v[0]['n_fresh']}", d2(m("audit")), d2(m("judge")), d2(m("free")), d2(m("vote")), d2(m("key")), sg(m("judge") - m("free"))]
    b.append(" & ".join(cells) + " \\\\")
write("kinds.tex", "\n".join(b))

open(os.path.join(OUT, "check.txt"), "w").write("\n".join(CHK))
print("wrote", sorted(os.listdir(OUT)))
