import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from statsmodels.duration.hazard_regression import PHReg

N_TOP = 10

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
df = df[df["time"] > 0].reset_index(drop=True)

mut = pd.read_csv("brca_tcga_pan_can_atlas_2018/data_mutations.txt", sep="\t",
                  comment="#", low_memory=False,
                  usecols=["Hugo_Symbol", "Tumor_Sample_Barcode", "Variant_Classification"])
mut = mut[mut["Variant_Classification"] != "Silent"]
mut["pid"] = mut["Tumor_Sample_Barcode"].str[:12]
mut = mut[mut["pid"].isin(df["pid"])].drop_duplicates(["pid", "Hugo_Symbol"])

tmb = np.log1p(mut.groupby("pid").size().reindex(df["pid"]).fillna(0).values)
M = pd.crosstab(mut["pid"], mut["Hugo_Symbol"]).reindex(df["pid"]).fillna(0)
M = M.loc[:, M.sum() >= 10]
print(M.shape, int(df["event"].sum()))
print(M.mean().sort_values(ascending=False).head(N_TOP).round(3))

clin = ["age", "stage"]
mcols = ["m%d" % i for i in range(N_TOP)] + ["tmb"]
sets = {"clinical": clin, "mutations": mcols, "clinical+mutations": clin + mcols}
scores = {k: [] for k in sets}

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
for tr, te in cv.split(df, df["event"]):
    a, b = df.iloc[tr].copy(), df.iloc[te].copy()
    for c in clin:
        m = a[c].median()
        a[c] = a[c].fillna(m)
        b[c] = b[c].fillna(m)

    top = M.iloc[tr].mean().sort_values(ascending=False).index[:N_TOP]
    for i, g in enumerate(top):
        a["m%d" % i], b["m%d" % i] = M[g].values[tr], M[g].values[te]
    a["tmb"], b["tmb"] = tmb[tr], tmb[te]

    for name, cols in sets.items():
        m, s = a[cols].mean(), a[cols].std() + 1e-8
        fit = PHReg(a["time"], (a[cols] - m) / s, status=a["event"]).fit(disp=0)
        risk = ((b[cols] - m) / s).values @ fit.params
        scores[name].append(c_index(b["time"], b["event"], risk))

for name, v in scores.items():
    print(name, round(np.mean(v), 3), round(np.std(v), 3))
d = np.array(scores["clinical+mutations"]) - np.array(scores["clinical"])
print("gain", round(d.mean(), 3), "folds better:", int((d > 0).sum()), "of", len(d))
pd.DataFrame(scores).to_csv("scores_mut.csv", index=False)