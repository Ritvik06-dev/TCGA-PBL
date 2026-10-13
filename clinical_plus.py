import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from statsmodels.duration.hazard_regression import PHReg

df = pd.read_csv("cohort_final.csv")
df = df[["bcr_patient_barcode", "age_at_initial_pathologic_diagnosis",
         "ajcc_pathologic_tumor_stage", "histological_type",
         "menopause_status", "PFI", "PFI.time"]].copy()
df.columns = ["pid", "age", "stage", "hist", "meno", "event", "time"]

cb = pd.read_csv("brca_tcga_pan_can_atlas_2018/data_clinical_patient.txt",
                 sep="\t", comment="#")
cb = cb[["PATIENT_ID", "SUBTYPE", "PATH_T_STAGE", "PATH_N_STAGE"]]
cb.columns = ["pid", "subtype", "t", "n"]
df = df.merge(cb.drop_duplicates("pid"), on="pid", how="left")


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


def tn_num(s):
    s = str(s)
    if len(s) > 1 and s[1].isdigit():
        return int(s[1])
    return np.nan


def meno_num(s):
    s = str(s)
    if s.startswith("Post"):
        return 1
    if s.startswith("Pre") or s.startswith("Peri"):
        return 0
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
df["t"] = df["t"].map(tn_num)
df["n"] = df["n"].map(tn_num)
df["post"] = df["meno"].map(meno_num)
df["lobular"] = df["hist"].str.contains("Lobular", na=False).astype(float)
df["other_hist"] = ((df["lobular"] == 0) &
                    (df["hist"] != "Infiltrating Ductal Carcinoma")).astype(float)
for s in ["LumB", "Basal", "Her2", "Normal"]:
    df["sub_" + s] = (df["subtype"] == "BRCA_" + s).astype(float)
df["sub_missing"] = df["subtype"].isna().astype(float)

df = df.dropna(subset=["time", "event"])
df = df[df["time"] > 0].reset_index(drop=True)
print(len(df), int(df["event"].sum()))
print(df[["stage", "t", "n", "post"]].isna().sum())
print(df[["lobular", "other_hist", "sub_LumB", "sub_Basal", "sub_Her2",
          "sub_Normal", "sub_missing"]].sum())

basic = ["age", "stage"]
plus = basic + ["t", "n", "lobular", "other_hist", "post"]
sub = plus + ["sub_LumB", "sub_Basal", "sub_Her2", "sub_Normal", "sub_missing"]
sets = {"basic": basic, "plus": plus, "plus_subtype": sub}
scores = {k: [] for k in sets}
weights = []

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
for tr, te in cv.split(df, df["event"]):
    a, b = df.iloc[tr].copy(), df.iloc[te].copy()
    for c in ["age", "stage", "t", "n", "post"]:
        m = a[c].median()
        a[c] = a[c].fillna(m)
        b[c] = b[c].fillna(m)
    for name, cols in sets.items():
        m, s = a[cols].mean(), a[cols].std() + 1e-8
        fit = PHReg(a["time"], (a[cols] - m) / s, status=a["event"]).fit(disp=0)
        risk = ((b[cols] - m) / s).values @ fit.params
        scores[name].append(c_index(b["time"], b["event"], risk))
        if name == "plus_subtype":
            weights.append(fit.params)

base = np.array(scores["basic"])
for name, v in scores.items():
    d = np.array(v) - base
    print(name, round(np.mean(v), 3), round(np.std(v), 3),
          "gain", round(d.mean(), 3), "better", int((d > 0).sum()))
print(pd.Series(np.mean(weights, axis=0), index=sub).round(3))
pd.DataFrame(scores).to_csv("scores_clinical.csv", index=False)