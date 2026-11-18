"""Figure: judge-paid share of wrong training samples (judgeQt50, TriviaQA) vs step (left) and vs visit number (right), by visit regime."""
import numpy as np, pandas as pd, dyn, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = dyn.derive(dyn.logs()); D = D[(D.ds == "h1") & (D.g == "judgeQt50") & (D.idkc == "idkc05") & D.tag.isin(["B1K20", "B2vK20", "B5vK20", "B10vK20", "FK20"])]
V = pd.read_csv(f"{dyn.HERE}/d09_visit_curves.csv"); V = V[V.g == "judgeQt50"]
col = {"B1K20": "#888888", "B2vK20": "#1f77b4", "B5vK20": "#2ca02c", "FK20": "#9467bd", "B10vK20": "#d62728"}
lab = {"B1K20": "1 visit (3840 q)", "B2vK20": "2 visits (1920 q)", "B5vK20": "5 visits (768 q)", "FK20": "5.1 visits (750 q)", "B10vK20": "10 visits (384 q)"}
fig, ax = plt.subplots(2, 2, figsize=(11, 7))
for r, st in enumerate(("qwen7", "llama8")):
    for tag in col:
        x = D[(D.st == st) & (D.tag == tag)].groupby("step").wrong_paid.mean().rolling(5, min_periods=1, center=True).mean()
        ax[r, 0].plot(x.index, x.values, color=col[tag], label=lab[tag])
        v = V[(V.st == st) & (V.tag == tag)].groupby("visit").wrong_paid.mean()
        ax[r, 1].plot(v.index, v.values, "o-", color=col[tag], label=lab[tag])
    ax[r, 0].set_title(f"{st}: judge-paid share of wrong samples vs step"); ax[r, 1].set_title(f"{st}: same vs visit number")
    ax[r, 0].set_xlabel("step"); ax[r, 1].set_xlabel("visit to the question")
ax[0, 0].legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{dyn.HERE}/d21_fig_visits.png", dpi=80)
