import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.duration.survfunc import SurvfuncRight, survdiff

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200})

labels = {"clinical": "Clinical", "rna": "RNA expression",
          "text": "Pathology reports", "slide": "Slides (attention)",
          "mutations": "Mutations", "concat": "Fusion: concatenation",
          "late_avg": "Fusion: equal-weight average",
          "late_stack": "Fusion: stacking"}

s = pd.read_csv("scores_fusion2.csv")
order = list(labels)[::-1]
y = np.arange(len(order))

fig, ax = plt.subplots(figsize=(7.5, 4.2))
m, sd = s[order].mean(), s[order].std(ddof=0)
ax.axvline(0.5, color=GRID, lw=1)
ax.axvline(m["clinical"], color=MUTED, lw=1, ls="--")
ax.errorbar(m, y, xerr=sd, fmt="o", color=BLUE, ms=7, lw=2, capsize=0)
for i, k in enumerate(order):
    ax.text(0.80, i, "%.3f" % m[k], va="center", color=INK)
ax.set_yticks(y)
ax.set_yticklabels([labels[k] for k in order])
ax.set_xlabel("C-index (mean and spread over 25 test folds)")
ax.set_xlim(0.38, 0.85)
ax.set_ylim(-0.6, len(order) - 0.1)
ax.text(0.5, len(order) - 0.45, "random", color=MUTED, ha="center", fontsize=9)
ax.text(m["clinical"] - 0.005, len(order) - 0.45, "clinical", color=MUTED, ha="center", fontsize=9)
ax.set_title("C-index of each model", loc="left", color=INK, fontweight="bold")
fig.tight_layout()
fig.savefig("fig_models.png")

fus = ["late_stack", "concat", "late_avg"][::-1]
fig, ax = plt.subplots(figsize=(7.5, 2.8))
for i, k in enumerate(fus):
    d = (s[k] - s["clinical"]).values
    se = np.sqrt((1 / len(d) + 0.25) * d.var(ddof=1))
    h = stats.t.ppf(0.975, len(d) - 1) * se
    ax.errorbar(d.mean(), i, xerr=h, fmt="o", color=BLUE, ms=7, lw=2)
    ax.text(d.mean() + h + 0.004, i, "%+.3f" % d.mean(), va="center", color=INK)
ax.axvline(0, color=MUTED, lw=1, ls="--")
ax.set_yticks(range(len(fus)))
ax.set_yticklabels([labels[k] for k in fus])
ax.set_ylim(-0.6, len(fus) - 0.4)
ax.set_xlabel("Gain in C-index over clinical (95% confidence interval)")
ax.set_title("Gain over clinical data alone",
             loc="left", color=INK, fontweight="bold")
fig.tight_layout()
fig.savefig("fig_gain.png")

w = pd.read_csv("weights_fusion2.csv")
names = {"clinical": "Clinical", "text": "Reports", "rna": "RNA",
         "slide": "Slides", "mutations": "Mutations"}
fig, ax = plt.subplots(figsize=(6.5, 3.6))
x = np.arange(len(names))
mw, sw = w[list(names)].mean(), w[list(names)].std(ddof=0)
ax.bar(x, mw, yerr=sw, width=0.55, color=BLUE, ecolor=MUTED, error_kw={"lw": 1.5})
ax.axhline(0, color="#c3c2b7", lw=1)
for i, k in enumerate(names):
    ax.text(i, max(mw[k], 0) + sw[k] + 0.02, "%.2f" % mw[k], ha="center", color=INK)
ax.set_ylim(top=(mw + sw).max() + 0.12)
ax.set_xticks(x)
ax.set_xticklabels(list(names.values()), color=INK)
ax.set_ylabel("Weight in the stacked fusion")
ax.set_title("How much the fusion trusts each data type",
             loc="left", color=INK, fontweight="bold")
fig.tight_layout()
fig.savefig("fig_weights.png")

r = pd.read_csv("risks_fusion2.csv")
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, col, title in zip(axes, ["risk_clinical", "risk_stack"],
                          ["Clinical model", "Stacked fusion"]):
    high = (r[col] > r[col].median()).values
    for g, c, lab in [(~high, BLUE, "Low risk"), (high, ORANGE, "High risk")]:
        sf = SurvfuncRight(r["time"][g] / 365.25, r["event"][g])
        ax.step(np.r_[0, sf.surv_times], np.r_[1, sf.surv_prob], where="post",
                color=c, lw=2, label="%s (n = %d)" % (lab, g.sum()))
    _, p = survdiff(r["time"], r["event"], high)
    ptxt = "log-rank p < 0.0001" if p < 0.0001 else "log-rank p = %.4f" % p
    ax.text(0.03, 0.06, ptxt, transform=ax.transAxes, color=INK)
    ax.set_title(title, loc="left", color=INK, fontweight="bold")
    ax.set_xlabel("Years since diagnosis")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(frameon=False, loc="lower right")
axes[0].set_ylabel("Progression-free probability")
fig.tight_layout()
fig.savefig("fig_km.png")
print("saved fig_models.png fig_gain.png fig_weights.png fig_km.png")