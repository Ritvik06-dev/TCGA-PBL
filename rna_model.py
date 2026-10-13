import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedStratifiedKFold
from statsmodels.duration.hazard_regression import PHReg

N_GENES = 2000
N_PCS = 10

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

rna = pd.read_csv("brca_tcga_pan_can_atlas_2018/data_mrna_seq_v2_rsem.txt", sep="\t")
rna = rna.drop(columns=["Hugo_Symbol", "Entrez_Gene_Id"]).dropna()
rna = rna[sorted(rna.columns, key=lambda c: c[13:15])]
rna.columns = [c[:12] for c in rna.columns]
rna = rna.loc[:, ~rna.columns.duplicated()]
X = np.log2(rna[df["pid"]].T.values + 1)
print(X.shape, int(df["event"].sum()))

clin = ["age", "stage"]
pcs = ["pc%d" % i for i in range(N_PCS)]
sets = {"clinical": clin, "rna": pcs, "clinical+rna": clin + pcs}
scores = {k: [] for k in sets}

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
for tr, te in cv.split(df, df["event"]):
    a, b = df.iloc[tr].copy(), df.iloc[te].copy()
    for c in clin:
        m = a[c].median()
        a[c] = a[c].fillna(m)
        b[c] = b[c].fillna(m)

    top = np.argsort(X[tr].var(axis=0))[-N_GENES:]
    mu, sd = X[tr][:, top].mean(axis=0), X[tr][:, top].std(axis=0) + 1e-8
    pca = PCA(n_components=N_PCS, random_state=0)
    pa = pca.fit_transform((X[tr][:, top] - mu) / sd)
    pb = pca.transform((X[te][:, top] - mu) / sd)
    for i, p in enumerate(pcs):
        a[p], b[p] = pa[:, i], pb[:, i]

    for name, cols in sets.items():
        m, s = a[cols].mean(), a[cols].std()
        fit = PHReg(a["time"], (a[cols] - m) / s, status=a["event"]).fit(disp=0)
        risk = ((b[cols] - m) / s).values @ fit.params
        scores[name].append(c_index(b["time"], b["event"], risk))

for name, v in scores.items():
    print(name, round(np.mean(v), 3), round(np.std(v), 3))
d = np.array(scores["clinical+rna"]) - np.array(scores["clinical"])
print("gain", round(d.mean(), 3), "folds better:", int((d > 0).sum()), "of", len(d))
pd.DataFrame(scores).to_csv("scores_rna.csv", index=False)