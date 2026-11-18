"""(verify_map) Renders the proposed central figure from cache/v08_fig_main.pkl (numbers computed by v08_figure.py main 1000 2000).
Two panels on one shared y scale: new questions, judge-pair contrast; x = quintile of the readable contrast (K's probe minus J's probe, net of
generic difficulty), y = refusal of K-trained minus J-trained runs (percentage points), 95% bootstrap band. Output: v11_figure.png"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import pickle, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = (_RA + "/verify_map")
F = pickle.load(open(f"{HERE}/cache/v08_fig_main.pkl", "rb"))
SURF, INK, INK2, GRID, BLUE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de", "#2a78d6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True, facecolor=SURF)
titles = {"right": "Untrained answer RIGHT\n(judge rejections are false rejections)", "wrong": "Untrained answer WRONG\n(judge acceptances are false accepts)"}
for ax, panel in zip(axes, ("right", "wrong")):
    d = F[("MAIN: new questions", panel)]; x = np.arange(1, 6)
    c, lo, hi = 100 * d["curve"], 100 * d["curve_lo"], 100 * d["curve_hi"]
    ax.set_facecolor(SURF); ax.axhline(0, color=GRID, lw=1)
    ax.fill_between(x, lo, hi, color=BLUE, alpha=0.12, lw=0)
    ax.plot(x, c, color=BLUE, lw=2, solid_capstyle="round", solid_joinstyle="round")
    ax.scatter(x, c, s=42, color=BLUE, edgecolor=SURF, linewidth=2, zorder=3)
    ax.annotate(f"Q5 - Q1 = {c[4] - c[0]:+.0f} pts\nslope {100 * d['slope']:+.1f} pts / SD", xy=(5, c[4]), xytext=(3.05, 13.2), color=INK, fontsize=9)
    ax.set_title(titles[panel], color=INK, fontsize=10, loc="left")
    ax.set_xticks(x); ax.set_xticklabels(["Q1\nprobes: J\nrejects more", "Q2", "Q3", "Q4", "Q5\nprobes: K\nrejects more"], fontsize=8.5)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axes[0].set_ylabel("refusal: K-trained minus J-trained\n(percentage points, centred)")
axes[0].set_ylim(-16, 17)
fig.suptitle("New questions: each judge's model refuses where that judge's readable map predicts rejection - for right and wrong answers", color=INK, fontsize=10.5, x=0.01, ha="left")
fig.text(0.01, -0.06, "x: quintile of [probe_K - probe_J], layer-16 linear probes of each judge's verdict fitted on training questions only, residualised on grader-free "
         "difficulty.\n44 judge pairs (K, J) pooled over 2 students x 3 datasets, strict (a35) runs; band = 95% bootstrap (questions, runs). Placebo (judge labels permuted): flat.",
         color=INK2, fontsize=8)
fig.tight_layout(rect=(0, 0.02, 1, 0.93)); fig.savefig(f"{HERE}/v11_figure_pooled.png", dpi=150, facecolor=SURF, bbox_inches="tight")
print("saved", f"{HERE}/v11_figure_pooled.png")
# ---- main proposed figure: same design, split by the judges' error profile (NQ: judges pay 8-16% of wrong answers; TriviaQA + PopQA: 0-6%)
FB = pickle.load(open(f"{HERE}/cache/v08_fig_byds.pkl", "rb")); ORANGE = "#eb6834"
LAB = {"NQ-open only (judges pay 8-16% of wrong answers at a35)": ("NQ-open", ORANGE), "TriviaQA + PopQA only (judges pay 0-6% of wrong answers at a35)": ("TriviaQA + PopQA", BLUE)}
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True, facecolor=SURF)
for ax, panel in zip(axes, ("right", "wrong")):
    ax.set_facecolor(SURF); ax.axhline(0, color=GRID, lw=1); x = np.arange(1, 6)
    for lab, (name, col) in LAB.items():
        d = FB[(lab, panel)]; c, lo, hi = 100 * d["curve"], 100 * d["curve_lo"], 100 * d["curve_hi"]
        ax.fill_between(x, lo, hi, color=col, alpha=0.10, lw=0)
        ax.plot(x, c, color=col, lw=2, solid_capstyle="round", solid_joinstyle="round", label=f"{name}: slope {100 * d['slope']:+.1f} pts/SD")
        ax.scatter(x, c, s=42, color=col, edgecolor=SURF, linewidth=2, zorder=3)
        ax.annotate(name, xy=(5, c[4]), xytext=(5.12, c[4]), color=INK2, fontsize=8.5, va="center")
    ax.set_title(titles[panel], color=INK, fontsize=10, loc="left")
    ax.set_xticks(x); ax.set_xticklabels(["Q1\nprobes: J\nrejects more", "Q2", "Q3", "Q4", "Q5\nprobes: K\nrejects more"], fontsize=8.5)
    ax.set_xlim(0.7, 6.1); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK2)
axes[0].set_ylabel("refusal: K-trained minus J-trained\n(percentage points, centred)"); axes[0].set_ylim(-17, 20)
fig.suptitle("New questions: each judge's model refuses where its judge's readable map predicts rejection, on right AND wrong answers", color=INK, fontsize=10.5, x=0.01, ha="left")
fig.text(0.01, -0.07, "x: quintile of [probe_K - probe_J] (layer-16 linear probes of each judge's a35 verdict, fitted on training questions only), residualised on grader-free difficulty.\n"
         "Strict (a35) runs, 2 students; NQ 20 judge pairs, TriviaQA + PopQA 24 pairs; bands = 95% bootstrap (questions, runs); judge-label permutation p <= 0.001 in every panel.\n"
         "At a35 the judges reject 31-59% of right answers everywhere but pay 8-16% of wrong answers on NQ-open and 0-6% on TriviaQA / PopQA.", color=INK2, fontsize=8)
fig.tight_layout(rect=(0, 0.02, 1, 0.93)); fig.savefig(f"{HERE}/v11_figure.png", dpi=150, facecolor=SURF, bbox_inches="tight")
print("saved", f"{HERE}/v11_figure.png")
