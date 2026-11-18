"""(dose) Figure: dose-response of refusal of known answers on NEW questions vs the grader's predicted rejection (from d04_plot.pkl, d05_summary.pkl).
A: adjusted refusal by quintile of the grader-specific predicted rejection (lin16, strict judges), trained vs matched no-location runs (95% CI).
B: grader-specific part (trained minus reference) by operating point. C: partial correlation of refusal with each predictor (strict judges),
new questions (blue) and training questions with out-of-fold predictors (orange). D: capitals probe groups and missing-alias items, observed refusal
vs the refusal predicted by the readable curve (text predictors for capitals, lin16 for aliases). Output: d06_fig.png"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import pickle, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = (_RA + "/dose")
PL = pickle.load(open(f"{HERE}/d04_plot.pkl", "rb")); SM = pickle.load(open(f"{HERE}/d05_summary.pkl", "rb"))
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"; INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.facecolor": SURF, "figure.facecolor": SURF, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True})
fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.6))
x = np.arange(1, 6)
def band(a, c, lab, d, off=0.0, ls="-"):
    v, lo, hi = d; a.errorbar(x + off, v, yerr=[v - lo, hi - v], color=c, lw=2, ls=ls, marker="o", ms=6, capsize=0, elinewidth=1.2, mec=SURF, mew=1.5)
    a.annotate(lab, (x[-1] + off, v[-1]), xytext=(6, 0), textcoords="offset points", va="center", color=INK2, fontsize=8.5)
a = ax[0, 0]; c = PL["curve"]["pooled"]
band(a, BLUE, "judge-trained\n(strict cut)", c["run"], -0.04); band(a, ORANGE, "no-location\nreference", c["ref"], 0.04)
a.set_xticks(x); a.set_xticklabels(["Q1\nlowest", "Q2", "Q3", "Q4", "Q5\nhighest"]); a.set_xlim(0.6, 6.3); a.set_ylim(0, 0.6)
a.set_ylabel("refusal of known answers, new questions\n(adjusted for generic difficulty)"); a.set_xlabel("grader-specific predicted rejection (quintile; linear probe, layer 16)")
a.set_title("A  Refusal rises with the judge's predicted rejection", loc="left", color=INK, fontsize=10)
a = ax[0, 1]
for (lab, d), col, off in zip(PL["ops"].items(), (BLUE, ORANGE, AQUA), (-0.05, 0.0, 0.05)):
    if d is not None: band(a, col, {"strict a35": "strict cut", "middle Q a50": "50% cut (Q)", "default t50": "P(Yes) > 0.5"}[lab], d, off)
a.axhline(0, color=AXIS, lw=1); a.set_xticks(x); a.set_xticklabels(["Q1", "Q2", "Q3", "Q4", "Q5"]); a.set_xlim(0.6, 6.3)
a.set_ylabel("trained minus matched reference"); a.set_xlabel("grader-specific predicted rejection (quintile)")
a.set_title("B  The grader-specific part needs an abstention budget", loc="left", color=INK, fontsize=10)
a = ax[1, 0]
names = {"lin4": "linear, layer 4", "lin8": "linear, layer 8", "lin12": "linear, layer 12", "lin16": "linear, layer 16", "lin20": "linear, layer 20",
         "lin24": "linear, layer 24", "linlast": "linear, last layer", "lin16p64": "linear, layer 16 (64 PCs)", "mlp16": "MLP, layer 16 (64 PCs)",
         "knn16": "kNN, layer 16", "tfidf": "TF-IDF of question", "knnE": "kNN, question embedding", "v_bin": "actual verdict (greedy answer)",
         "v_share": "actual verdict (33 answers)", "v_py": "actual verdict P(Yes)"}
ks = list(names)[::-1]; y = np.arange(len(ks))
for i, k in enumerate(ks):
    r = PL["pcor_new"].get(k)
    if r: a.errorbar(r[0], i + 0.12, xerr=[[r[0] - r[1]], [r[2] - r[0]]], color=BLUE, marker="o", ms=6, lw=0, elinewidth=1.2, mec=SURF, mew=1.2)
    h = PL["pcor_home"].get(k)
    if h: a.errorbar(h[0], i - 0.12, xerr=[[h[0] - h[1]], [h[2] - h[0]]], color=ORANGE, marker="s", ms=5.5, lw=0, elinewidth=1.2, mec=SURF, mew=1.2)
a.set_yticks(y); a.set_yticklabels([names[k] for k in ks]); a.set_xlim(0, 0.72); a.axhspan(2.5, -0.5, color=GRID, alpha=0.35, lw=0)
a.set_xlabel("partial correlation with refusal of known answers (strict judges)")
a.plot([], [], color=BLUE, marker="o", lw=0, label="new questions"); a.plot([], [], color=ORANGE, marker="s", lw=0, label="training questions (out-of-fold)")
a.legend(loc="upper right", frameon=False, fontsize=8.5)
a.set_title("C  Readable prediction vs actual verdict", loc="left", color=INK, fontsize=10)
a = ax[1, 1]
a.plot([0, 1], [0, 1], color=AXIS, lw=1, ls="--"); a.text(0.47, 0.51, "on the curve", color=MUTED, fontsize=8, rotation=38)
lab = {("pqq", "F"): "former-state\n(hand-written)", ("pqq", "P"): "present-day\n(hand-written)", ("pqp", "P"): "present-day\n(PopQA-derived)", ("pqp", "R"): "religion\n(PopQA-derived)"}
for key, r in SM.items():
    if key[0] != "cap": continue
    kind, st, s, g, vs = key
    if vs != "all" or (s, g) not in lab: continue
    mk = "o" if st == "qwen7" else "s"; a.plot(r["pred_read_tfidf"], r["yj"], marker=mk, color=BLUE, ms=7, lw=0, mec=SURF, mew=1.2)
    off = {("pqp", "P"): (-4, -24)}.get((s, g), (7, -3))
    if st == "llama8": a.annotate(lab[(s, g)], (r["pred_read_tfidf"], r["yj"]), xytext=off, textcoords="offset points", fontsize=7.5, color=INK2)
for key, r in SM.items():
    if key[0] != "alias": continue
    kind, st, ds, pt = key
    mk = "o" if st == "qwen7" else "s"; a.plot(r["pred_read"], r["yk_miss"], marker=mk, color=ORANGE if pt == "new" else AQUA, ms=6, lw=0, mec=SURF, mew=1.0)
a.plot([], [], color=BLUE, marker="o", lw=0, label="capitals probes, strict-judge runs (TF-IDF curve)")
a.plot([], [], color=ORANGE, marker="o", lw=0, label="key's missing aliases, new questions (key runs)")
a.plot([], [], color=AQUA, marker="o", lw=0, label="key's missing aliases, training questions")
a.plot([], [], color=MUTED, marker="o", lw=0, label="Qwen"); a.plot([], [], color=MUTED, marker="s", lw=0, label="Llama")
a.legend(loc="lower right", frameon=False, fontsize=7.5); a.set_xlim(0, 1.0); a.set_ylim(0, 1.04)
a.set_xlabel("refusal predicted by the readable curve"); a.set_ylabel("observed refusal")
a.set_title("D  Probe items vs the curve", loc="left", color=INK, fontsize=10)
for a_ in ax.flat:
    for sp in ("top", "right"): a_.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(f"{HERE}/d06_fig.png", dpi=160); print("wrote d06_fig.png")
