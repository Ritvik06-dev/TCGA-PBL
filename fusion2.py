import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from statsmodels.duration.hazard_regression import PHReg

K = 10
G = "brca_tcga_pan_can_atlas_2018/"

df = pd.read_csv("cohort_final.csv")
df = df[["bcr_patient_barcode", "age_at_initial_pathologic_diagnosis",
         "ajcc_pathologic_tumor_stage", "histological_type",
         "menopause_status", "PFI", "PFI.time"]].copy()
df.columns = ["pid", "age", "stage", "hist", "meno", "event", "time"]
cb = pd.read_csv(G + "data_clinical_patient.txt", sep="\t", comment="#")
cb = cb[["PATIENT_ID", "PATH_T_STAGE", "PATH_N_STAGE"]]
cb.columns = ["pid", "t", "n"]
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


def scale(a, b):
    m, s = a.mean(axis=0), a.std(axis=0) + 1e-8
    return (a - m) / s, (b - m) / s


def cox(x, t, e):
    try:
        return PHReg(t, x, status=e).fit(disp=0).params
    except np.linalg.LinAlgError:
        return PHReg(t, x, status=e).fit_regularized(alpha=0.01, L1_wt=0).params


def corrected(d):
    k = len(d)
    se = np.sqrt((1 / k + 0.25) * d.var(ddof=1))
    h = stats.t.ppf(0.975, k - 1) * se
    p = 2 * stats.t.sf(abs(d.mean() / se), k - 1)
    return d.mean() - h, d.mean() + h, p


df["stage"] = df["stage"].map(stage_num)
df["t"] = df["t"].map(tn_num)
df["n"] = df["n"].map(tn_num)
df["post"] = df["meno"].map(meno_num)
df["lobular"] = df["hist"].str.contains("Lobular", na=False).astype(float)
df["other_hist"] = ((df["lobular"] == 0) &
                    (df["hist"] != "Infiltrating Ductal Carcinoma")).astype(float)
df = df.dropna(subset=["time", "event"])
df = df[df["time"] > 0].reset_index(drop=True)
T, E = df["time"].values, df["event"].values
pid = df["pid"]

clinical = df[["age", "stage", "t", "n", "lobular", "other_hist", "post"]].values

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
attn = np.load("slide_attn_risks.npz")
assert attn["full"].shape[1] == len(df)

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
    out["slide_mean"] = scale(p.transform(a), p.transform(b))

    top = M.iloc[tr].mean().sort_values(ascending=False).index[:K]
    a = np.column_stack([M[top].values[tr], tmb[tr]])
    b = np.column_stack([M[top].values[te], tmb[te]])
    out["mutations"] = scale(a, b)
    return out


names = ["clinical", "rna", "text", "slide", "mutations"]
keys = names + ["concat", "late_avg", "late_stack"]
scores = {k: [] for k in keys}
weights = []
risk_clin = np.full((25, len(df)), np.nan)
risk_stack = np.full((25, len(df)), np.nan)

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
for f, (tr, te) in enumerate(cv.split(df, E)):
    bl = blocks(tr, te)
    t, e = T[tr], E[tr]

    cat = ["clinical", "rna", "text", "slide_mean", "mutations"]
    A = np.hstack([bl[n][0] for n in cat])
    B = np.hstack([bl[n][1] for n in cat])
    scores["concat"].append(c_index(T[te], E[te], B @ cox(A, t, e)))

    oof = np.zeros((len(tr), len(names)))
    full_tr = np.zeros((len(tr), len(names)))
    full_te = np.zeros((len(te), len(names)))
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for j, n in enumerate(names):
        if n == "slide":
            full_tr[:, j] = attn["full"][f, tr]
            full_te[:, j] = attn["full"][f, te]
            oof[:, j] = attn["oof"][f, tr]
        else:
            a, b = bl[n]
            w = cox(a, t, e)
            full_tr[:, j], full_te[:, j] = a @ w, b @ w
            for itr, ite in inner.split(a, e):
                oof[ite, j] = a[ite] @ cox(a[itr], t[itr], e[itr])
        scores[n].append(c_index(T[te], E[te], full_te[:, j]))

    _, z_te = scale(full_tr, full_te)
    scores["late_avg"].append(c_index(T[te], E[te], z_te.mean(axis=1)))
    z_oof, _ = scale(oof, oof)
    w = cox(z_oof, t, e)
    weights.append(w)
    scores["late_stack"].append(c_index(T[te], E[te], z_te @ w))
    risk_clin[f, te] = z_te[:, 0]
    risk_stack[f, te] = z_te @ w
    print("fold", f + 1, "of 25")

base = np.array(scores["clinical"])
for name in keys:
    v = np.array(scores[name])
    d = v - base
    lo, hi, p = corrected(d) if name != "clinical" else (0, 0, 1)
    print(name.ljust(11), round(v.mean(), 3), round(v.std(), 3),
          "gain", round(d.mean(), 3), "CI", round(lo, 3), round(hi, 3),
          "p", round(p, 3), "better", int((d > 0).sum()))
print(pd.Series(np.mean(weights, axis=0), index=names).round(3))
pd.DataFrame(scores).to_csv("scores_fusion2.csv", index=False)
pd.DataFrame(weights, columns=names).to_csv("weights_fusion2.csv", index=False)
pd.DataFrame({"pid": pid, "time": T, "event": E,
              "risk_clinical": np.nanmean(risk_clin, axis=0),
              "risk_stack": np.nanmean(risk_stack, axis=0)}).to_csv(
    "risks_fusion2.csv", index=False)