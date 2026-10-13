import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from statsmodels.duration.hazard_regression import PHReg

df = pd.read_csv("cohort_final.csv")
df = df[["bcr_patient_barcode", "age_at_initial_pathologic_diagnosis",
         "ajcc_pathologic_tumor_stage", "PFI", "PFI.time"]].copy()
df.columns = ["pid", "age", "stage", "event", "time"]


def stage_num(s):
    s = str(s)
    if not s.startswith("Stage "):
        return np.nan
    s = s[6:]
    if s.startswith("IV"):
        return 4
    if s.startswith("III"):
        return 3
    if s.startswith("II"):
        return 2
    if s.startswith("I"):
        return 1
    return np.nan


def c_index(time, event, risk):
    time, event, risk = map(np.asarray, (time, event, risk))
    num, den = 0.0, 0.0
    for i in np.where(event == 1)[0]:
        later = time > time[i]
        den += later.sum()
        num += (risk[i] > risk[later]).sum() + 0.5 * (risk[i] == risk[later]).sum()
    return num / den


df["stage"] = df["stage"].map(stage_num)
df = df.dropna(subset=["time", "event"])
df = df[df["time"] > 0]
print(len(df), int(df["event"].sum()))
print(df["stage"].value_counts(dropna=False).sort_index())

cols = ["age", "stage"]
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
scores = []
for tr, te in cv.split(df, df["event"]):
    a, b = df.iloc[tr].copy(), df.iloc[te].copy()
    for c in cols:
        m = a[c].median()
        a[c] = a[c].fillna(m)
        b[c] = b[c].fillna(m)
        mu, sd = a[c].mean(), a[c].std()
        a[c] = (a[c] - mu) / sd
        b[c] = (b[c] - mu) / sd
    fit = PHReg(a["time"], a[cols], status=a["event"]).fit(disp=0)
    risk = b[cols].values @ fit.params
    scores.append(c_index(b["time"], b["event"], risk))

print(round(np.mean(scores), 3), round(np.std(scores), 3))