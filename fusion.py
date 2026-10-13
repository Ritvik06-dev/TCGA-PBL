import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from statsmodels.duration.hazard_regression import PHReg

K = 10
G = "brca_tcga_pan_can_atlas_2018/"

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


def scale(a, b):
    m, s = a.mean(axis=0), a.std(axis=0) + 1e-8
    return (a - m) / s, (b - m) / s


def cox(x, t, e):
    return PHReg(t, x, status=e).fit(disp=0).params


df["stage"] = df["stage"].map(stage_num)
df = df.dropna(subset=["time", "event"])
df = df[df["time"] > 0].reset_index(drop=True)
T, E = df["time"].values, df["event"].values
pid = df["pid"]

clinical = df[["age", "stage"]].values

rna = pd.read_csv(G + "data_mrna_seq_v2_rsem.txt", sep="\t")
rna = rna.drop(columns=["Hugo_Symbol", "Entrez_Gene_Id"]).dropna()
rna = rna[sorted(rna.columns, key=lambda c: c[13:15])]
rna.columns = [c[:12] for c in rna.columns]
rna = rna.loc[:, ~rna.columns.duplicated()]
rna = np.log2(rna[pid].T.values + 1)

rep = pd.read_csv("TCGA_Reports.csv")
rep["pid"] = rep["patient_filename"].str[:12]
docs = rep.groupby("pid")["text"].apply(" ".join).reindex(pid).fillna("").values

row = {p: i for i, p in enumerate(pd.read_csv("slide_pids.csv")["pid"])}
slide = np.load("slide_mean.npy")[[row[p] for p in pid]]

mut = pd.read_csv(G + "data_mutations.txt", sep="\t", comment="#", low_memory=False,
                  usecols=["Hugo_Symbol", "Tumor_Sample_Barcode", "Variant_Classification"])
mut = mut[mut["Variant_Classification"] != "Silent"]
mut["pid"] = mut["Tumor_Sample_Barcode"].str[:12]
mut = mut[mut["pid"].isin(pid)].drop_duplicates(["pid", "Hugo_Symbol"])
tmb = np.log1p(mut.groupby("pid").size().reindex(pid).fillna(0).values)
M = pd.crosstab(mut["pid"], mut["Hugo_Symbol"]).reindex(pid).fillna(0)
M = M.loc[:, M.sum() >= 10]


def blocks(tr, te):
    out = {}

    a, b = clinical[tr].copy(), clinical[te].copy()
    med = np.nanmedian(a, axis=0)
    a = np.where(np.isnan(a), med, a)
    b = np.where(np.isnan(b), med, b)
    out["clinical"] = scale(a, b)

    top = np.argsort(rna[tr].var(axis=0))[-2000:]
    a, b = scale(rna[tr][:, top], rna[te][:, top])
    p = PCA(n_components=K, random_state=0).fit(a)
    out["rna"] = scale(p.transform(a), p.transform(b))

    vec = TfidfVectorizer(min_df=5, max_features=5000, sublinear_tf=True,
                          stop_words="english", ngram_range=(1, 2))
    svd = TruncatedSVD(n_components=K, random_state=0)
    a = svd.fit_transform(vec.fit_transform(docs[tr]))
    b = svd.transform(vec.transform(docs[te]))
    out["text"] = scale(a, b)

    a, b = scale(slide[tr], slide[te])
    p = PCA(n_components=K, random_state=0).fit(a)
    out["slide"] = scale(p.transform(a), p.transform(b))

    top = M.iloc[tr].mean().sort_values(ascending=False).index[:K]
    a = np.column_stack([M[top].values[tr], tmb[tr]])
    b = np.column_stack([M[top].values[te], tmb[te]])
    out["mutations"] = scale(a, b)
    return out


names = ["clinical", "rna", "text", "slide", "mutations"]
scores = {k: [] for k in ["clinical", "concat", "late_avg", "late_stack"]}
weights = []

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
for f, (tr, te) in enumerate(cv.split(df, E)):
    bl = blocks(tr, te)
    t, e = T[tr], E[tr]

    A = np.hstack([bl[n][0] for n in names])
    B = np.hstack([bl[n][1] for n in names])
    scores["concat"].append(c_index(T[te], E[te], B @ cox(A, t, e)))

    oof = np.zeros((len(tr), len(names)))
    full_tr = np.zeros((len(tr), len(names)))
    full_te = np.zeros((len(te), len(names)))
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for j, n in enumerate(names):
        a, b = bl[n]
        w = cox(a, t, e)
        full_tr[:, j], full_te[:, j] = a @ w, b @ w
        for itr, ite in inner.split(a, e):
            oof[ite, j] = a[ite] @ cox(a[itr], t[itr], e[itr])

    scores["clinical"].append(c_index(T[te], E[te], full_te[:, 0]))
    _, z_te = scale(full_tr, full_te)
    scores["late_avg"].append(c_index(T[te], E[te], z_te.mean(axis=1)))

    z_oof, _ = scale(oof, oof)
    w = cox(z_oof, t, e)
    weights.append(w)
    scores["late_stack"].append(c_index(T[te], E[te], z_te @ w))
    print("fold", f + 1, "of 25")

base = np.array(scores["clinical"])
for name, v in scores.items():
    d = np.array(v) - base
    print(name, round(np.mean(v), 3), round(np.std(v), 3),
          "gain", round(d.mean(), 3), "better", int((d > 0).sum()))
print(pd.Series(np.mean(weights, axis=0), index=names).round(3))
pd.DataFrame(scores).to_csv("scores_fusion.csv", index=False)
pd.DataFrame(weights, columns=names).to_csv("weights_fusion.csv", index=False)