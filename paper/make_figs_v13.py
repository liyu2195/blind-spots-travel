"""Figures new in v13. Every number is read from v13_numbers.json (caches, complete audit), s18_fix_ceiling.json and s21_cut_and_calibration.json; the
random-rejection row of Figure 1 is copied from the v12 figure (fam_fix/e15_count_dose.json, matched controls). fig_dose / fig_round / fig_ladder are unchanged from v12.
Usage: python3 -I -B make_figs_v13.py"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
BLUE, ORANGE, AQUA, YELLOW, GREY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8985", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "serif", "font.serif": ["TeX Gyre Termes", "Times New Roman", "Nimbus Roman", "DejaVu Serif"], "font.size": 8, "axes.edgecolor": "#b9b8b3",
                     "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "text.color": INK, "pdf.fonttype": 42})
S = (_RA + "/supp_v11"); N = json.load(open("v13_numbers.json")); S18 = json.load(open(f"{S}/s18_fix_ceiling.json")); S21 = json.load(open(f"{S}/s21_cut_and_calibration.json")); S22B = json.load(open(f"{S}/s22b_transition_checks.json"))
CAP = N["cap"]; T = N["T"]; Q, L = "qwen7 pqa", "llama8 pqa"

# ---------------- Figure 1: who refuses the former-state capitals, at every level of abstention (fig1_data.json, written by fig1_data.py)
F1 = json.load(open("fig1_data.json"))
GROUPS = [("Graders that know the answers", [("Answer key", "sim00fn00", "sim00fn00", GREY), ("Sonnet as judge", "apiS55", "apiS55", GREY), ("Judge shown the reference", "judgeQRt50", "judgeQRt50", GREY)]),
          ("No judge", [("Student's own vote", "frzVOTE20", "frzVOTE20", AQUA)]),
          ("Judges at the key-matched cut", [("Gemma judge", "judgeGa50", "judgeGacc", BLUE), ("OLMo judge", "judgeOa50", "judgeOacc", BLUE), ("Mistral judge", "judgeMa50", "judgeMacc", BLUE), ("Qwen judge", "judgeQa50", "judgeQacc", BLUE)]),
          ("Judges at the 35% cut", [("OLMo judge", "judgeOa35", "judgeOa35", BLUE), ("Gemma judge", "judgeGa35", "judgeGa35", BLUE), ("Mistral judge", "judgeMa35", "judgeMa35", BLUE), ("Qwen judge", "judgeQa35", "judgeQa35", BLUE),
                                     ("Jury of the five judges", "frzMAJ35", "frzMAJ35", BLUE)])]
ypos = []; ylab = []; heads = []; y0 = 0.0
for gi, (gname, rows_) in enumerate(GROUPS):
    heads.append((y0, gname)); y0 -= 0.95
    for r in rows_: ypos.append((y0, r)); ylab.append(r[0]); y0 -= 1.0
    y0 -= 0.45
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), sharey=True, gridspec_kw={"wspace": 0.05})
for ax, (ti, st, col) in zip(axes, (("Qwen2.5-7B student", "qwen7", 1), ("Llama-3.1-8B student", "llama8", 2))):
    for y, r in ypos:
        d = F1[st][r[col]]
        ax.barh(y, d["F"], height=0.7, color=r[3], zorder=2); ax.scatter(d["P"], y, s=13, color=ORANGE, zorder=4, edgecolor="white", linewidth=0.5)
        ax.plot([d["ctrlF"], d["ctrlF"]], [y - 0.42, y + 0.42], color=INK, linewidth=1.1, zorder=3, solid_capstyle="butt")
        ax.text(max(d["F"], d["ctrlF"]) + 0.025, y, f"{round(d['F'] * 100):d}%", va="center", ha="left", fontsize=7, color=INK)
        ax.text(1.31, y, f"{round(d['abst'] * 100):d}%", va="center", ha="right", fontsize=7, color=INK2)
    ax.text(1.31, 0.05, "abstains", va="center", ha="right", fontsize=6.8, color=INK2, style="italic")
    ax.set_xlim(0, 1.33); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0", "25", "50", "75", "100%"]); ax.set_title(ti, fontsize=8.5, pad=4)
    ax.set_ylim(y0 + 0.7, 0.6); ax.grid(axis="x", color="#e7e6e1", linewidth=0.5, zorder=0); ax.spines[["top", "right"]].set_visible(False); ax.tick_params(length=0)
    for xg in (1.25,): ax.axvline(1.0, color="#e7e6e1", linewidth=0.5, zorder=0)
axes[0].set_yticks([y for y, _ in ypos]); axes[0].set_yticklabels(ylab, fontsize=7.3)
for y, gname in heads: axes[0].text(-0.62, y, gname, transform=axes[0].get_yaxis_transform(), ha="left", va="center", fontsize=7.3, style="italic", color=INK)
fig.text(0.56, 0.02, "Capitals the student knew that it refuses after training (questions never trained on)", ha="center", fontsize=8)
fig.legend(handles=[Patch(color=BLUE, label="reference-free judge"), Patch(color=AQUA, label="own vote"), Patch(color=GREY, label="grader that knows the answers"),
                    Line2D([], [], marker="|", color=INK, markersize=8, markeredgewidth=1.1, linestyle="None", label="random rejections, same abstention"),
                    Line2D([], [], marker="o", color="w", markerfacecolor=ORANGE, markersize=5, label="present-day capitals")],
           loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=5, frameon=False, fontsize=7.0, handlelength=1.1, columnspacing=1.1, handletextpad=0.4)
fig.savefig("figs/fig_capitals.pdf", bbox_inches="tight"); plt.close(fig)

# ---------------- Figure: does a grader's knowledge reach new questions? reward minus the untrained student's own-vote rule at the same abstention (s18, section C)
cells = ["qwen7 pqa", "qwen7 tqa", "qwen7 nqo", "llama8 pqa", "llama8 tqa", "llama8 nqo"]
G = [("Answer key", "sim00fn00"), ("Sonnet as judge", "apiS55"), ("Judge shown the reference", "judgeQRt50"), ("Student's own vote as grader", "frzVOTE20")]
fig, ax = plt.subplots(figsize=(3.35, 1.6)); yy = np.arange(len(G))[::-1]
ax.axvline(0, color=INK2, linewidth=0.8, zorder=1)
for yi, (lab, g) in zip(yy, G):
    v = [(c, 100 * S18[c]["C"][g][0]) for c in cells if g in S18[c]["C"]]; m = np.mean([x for _, x in v])
    for c, x in v:
        ll = c.startswith("llama8"); ax.scatter(x, yi, s=26, marker="o", facecolor=BLUE if ll else "white", edgecolor=BLUE, linewidth=1.0, zorder=3)
    tv = S22B["audit"]["vs_rule"][g]["total"]; assert abs(tv["mean"] - m) < 0.05, (g, tv["mean"], m)
    ax.plot(tv["t"], [yi - 0.27, yi - 0.27], color=INK, linewidth=0.9, zorder=4, solid_capstyle="butt")
    for e in tv["t"]: ax.plot([e, e], [yi - 0.34, yi - 0.20], color=INK, linewidth=0.9, zorder=4)
    ax.plot([m, m], [yi - 0.36, yi + 0.3], color=INK, linewidth=1.6, zorder=4); ax.text(3.45, yi, f"{m:+.1f}", va="center", ha="right", fontsize=7.5, color=INK)
ax.set_yticks(yy); ax.set_yticklabels([g[0] for g in G], fontsize=7.5); ax.set_xlim(-3.1, 3.5); ax.set_ylim(-0.6, len(G) - 0.4); ax.set_xticks([-3, -2, -1, 0, 1, 2]); ax.set_xticklabels(["-3", "-2", "-1", "0", "+1", "+2"])
ax.set_xlabel("True reward minus the untrained student's own\n32-sample vote at the same abstention (points)", fontsize=7.5)
ax.grid(axis="x", color="#e7e6e1", linewidth=0.5, zorder=0); ax.spines[["top", "right", "left"]].set_visible(False); ax.tick_params(length=0)
ax.legend(handles=[Line2D([], [], marker="o", color="w", markerfacecolor=BLUE, markeredgecolor=BLUE, markersize=5, label="Llama"), Line2D([], [], marker="o", color="w", markerfacecolor="white", markeredgecolor=BLUE, markersize=5, label="Qwen"),
                   Line2D([], [], color=INK, linewidth=1.6, label="mean, 95% interval")], loc="upper center", bbox_to_anchor=(0.4, 1.22), ncol=3, frameon=False, fontsize=7, handletextpad=0.3, columnspacing=1.2)
fig.savefig("figs/fig_vote.pdf", bbox_inches="tight"); plt.close(fig)

# ---------------- Figure: where the cut sits (s21): own vote minus answer key against the distance of the cut from the student's calibration point
fig, ax = plt.subplots(figsize=(3.35, 2.05)); ax.axvspan(-2, 12, color="#f3f2ee", zorder=0); ax.axhline(0, color=INK2, linewidth=0.8, zorder=1)
for a in S21["arms"]:
    ll = a["cell"].startswith("llama8"); x, yv = a["dist"], 100 * a["diff"]
    ax.plot([x, x], [100 * a["lo"], 100 * a["hi"]], color=BLUE, linewidth=0.6, zorder=2)
    ax.scatter(x, yv, s=26, marker="o", facecolor=BLUE if ll else "white", edgecolor=BLUE, linewidth=1.0, zorder=3)
ax.set_xlim(-11.5, 25.5); ax.set_ylim(-13.5, 7.5); ax.set_xticks([-10, -5, 0, 5, 10, 15, 20, 25]); ax.set_yticks([-10, -5, 0, 5])
ax.set_xlabel("Cut minus the student's calibration point (samples of 32)", fontsize=7.5); ax.set_ylabel("Own vote minus answer key\n(true reward, points)", fontsize=7.5)
ax.text(5, 6.4, "matches the key", ha="center", va="center", fontsize=7, color=INK2); ax.text(-6.8, -12.3, "too lenient", ha="center", fontsize=7, color=INK2); ax.text(19, -12.3, "too strict", ha="center", fontsize=7, color=INK2)
ax.spines[["top", "right"]].set_visible(False); ax.tick_params(length=2)
ax.legend(handles=[Line2D([], [], marker="o", color="w", markerfacecolor=BLUE, markeredgecolor=BLUE, markersize=5, label="Llama"), Line2D([], [], marker="o", color="w", markerfacecolor="white", markeredgecolor=BLUE, markersize=5, label="Qwen")],
          loc="upper right", bbox_to_anchor=(1.0, 1.02), ncol=1, frameon=False, fontsize=7, handletextpad=0.3, labelspacing=0.3)
fig.savefig("figs/fig_cut.pdf", bbox_inches="tight"); plt.close(fig)
print("figures written")
