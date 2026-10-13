import numpy as np, pandas as pd
from scipy import stats
from statsmodels.duration.survfunc import survdiff

s = pd.read_csv("scores_fusion2.csv")
print(s.agg(["mean", "std"]).T.round(3))
print()

for k in s.columns:
    if k == "clinical":
        continue
    d = (s[k] - s["clinical"]).values
    se = np.sqrt((1 / len(d) + 0.25) * d.var(ddof=1))
    h = stats.t.ppf(0.975, len(d) - 1) * se
    p = 2 * stats.t.sf(abs(d.mean() / se), len(d) - 1)
    print(f"{k:12s} gain {d.mean():+.3f}  CI {d.mean()-h:+.3f} to {d.mean()+h:+.3f}  p {p:.3f}  better {(d > 0).sum()}/25")
print()

w = pd.read_csv("weights_fusion2.csv")
print(w.agg(["mean", "std"]).T.round(3))
print()

r = pd.read_csv("risks_fusion2.csv")
for c in ["risk_clinical", "risk_stack"]:
    g = (r[c] > r[c].median()).astype(int)
    print(c, "log-rank p =", survdiff(r["time"], r["event"], g)[1])