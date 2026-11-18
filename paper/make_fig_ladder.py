"""Draw the appendix threshold-ladder figure using audited wrong-answer counts from s5_return_by_cut.json."""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
BLUE, ORANGE, AQUA, YELLOW, GREY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8985", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "serif", "font.serif": ["TeX Gyre Termes", "Times New Roman", "Nimbus Roman", "DejaVu Serif"], "font.size": 8, "axes.edgecolor": "#b9b8b3",
                     "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "text.color": INK, "pdf.fonttype": 42})
FF = (_RA + "/fam_fix")
# ---------------- Figure 3: the ladder for Llama / PopQA / Qwen judge (fix_ladder ladder_*_final_1117b, ladder_acc_final_1117b; e67; Table 3)
xs = np.arange(4); lab = ["default\nthreshold\n(3%)", "key-\nmatched\n(23%)", "50%\ncut\n(41%)", "35%\ncut\n(56%)"]
# every value read from the analysis outputs (Llama student, Qwen judge, PopQA); wrong answers by audited truth (key or audit), as in Table "What each cut earns"
_W = _RA; _CUTS = ("t50", "acc", "a50", "a35")
_S5 = {(r["st"], r["ds"], r["g"]): r for r in json.load(open(f"{_W}/supp_final/s5_return_by_cut.json"))}
_LD = {c: json.load(open(f"{_W}/fix_ladder/out/ladder_{c}_final_{'1117d' if c == 'acc' else '1117b'}.json"))[c]["costs"]["Q|llama8|pq"] for c in _CUTS}
_E15 = {(e["st"], e["label"]): e for e in json.load(open(f"{FF}/e15_count_dose.json"))}
wrong = [100 * _S5[("llama8", "pq", f"judgeQ{c}")]["wrong"] for c in _CUTS]; key_wrong = 100 * _S5[("llama8", "pq", "sim00fn00")]["wrong"]
withheld = [_LD[c]["RW"][0] for c in _CUTS]; hacks = [_LD[c]["R1"][0] for c in _CUTS]; refF = [100 * _E15[("llama8", f"judge Q {c}")]["RF"] for c in _CUTS]
import re as _re
_t = open(f"{FF}/e9b_forced_out.txt").read(); _blk = _t[_t.index("== llama8 pq"):]
_forced = [100 * float(_re.search(r"judge Q %s \(.*?key or audit ([\d.]+)" % k, _blk).group(1)) for k in ("t50", "key", "a50", "a35")]
lab = [f"{n}\n({v:.0f}%)" for n, v in zip(("default\nthreshold", "key-\nmatched", "50%\ncut", "35%\ncut"), _forced)]
print("Figure 3 values: wrong", [round(x, 1) for x in wrong], "key", round(key_wrong, 1), "withheld", [round(x, 1) for x in withheld], "refF", [round(x, 1) for x in refF], "hacks", [round(x, 1) for x in hacks], "forced", [round(x, 1) for x in _forced])
fig, ax = plt.subplots(figsize=(3.35, 2.5))
for v, c, m, name, dy in ((refF, BLUE, "o", "former-state capitals refused (%)", 4), (withheld, AQUA, "s", "right answers withheld", 4), (wrong, ORANGE, "^", "wrong answers given", 4), (hacks, GREY, "D", "paid wrong answers installed", -8)):
    ax.plot(xs, v, color=c, marker=m, markersize=4.5, linewidth=1.6, zorder=3)
ax.text(3.08, 98, "former-state\ncapitals refused (%)", color=BLUE, fontsize=6.8, va="center"); ax.text(3.08, 41.4, "right answers\nwithheld", color="#12865d", fontsize=6.8, va="center")
ax.text(-0.26, wrong[0] + 11, "wrong answers\ngiven", color="#c2501f", fontsize=6.8, va="bottom", ha="left"); ax.text(3.08, -5, "paid wrong answers\ninstalled (training)", color=INK2, fontsize=6.8, va="center")
ax.axhline(key_wrong, color=ORANGE, linewidth=0.7, linestyle=(0, (3, 3)), zorder=1); ax.text(3.08, key_wrong + 1.0, f"answer key:\n{key_wrong:.0f} wrong", color="#c2501f", fontsize=6.4, va="bottom")
ax.set_xticks(xs); ax.set_xticklabels(lab, fontsize=6.8); ax.set_xlim(-0.3, 4.45); ax.set_ylim(-8, 104); ax.set_yticks([0, 25, 50, 75, 100])
ax.set_ylabel("Per 100 questions"); ax.set_xlabel("Judge's cut (share of right answers it forces out)", fontsize=7.5)
ax.spines[["top", "right"]].set_visible(False); ax.tick_params(length=2); ax.grid(axis="y", color="#e7e6e1", linewidth=0.5, zorder=0)
fig.savefig("figs/fig_ladder.pdf", bbox_inches="tight"); plt.close(fig)
