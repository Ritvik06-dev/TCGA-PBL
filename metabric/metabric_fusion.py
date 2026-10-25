"""METABRIC external validation: same pipeline as fusion2.py (TCGA), on relapse-free survival.

Run from the metabric folder:
    python metabric_fusion.py                 # clinical + RNA + mutations
    python metabric_fusion.py --methylation   # adds methylation (patients who have it)
    python metabric_fusion.py --extended      # stronger clinical baseline: + grade, ER, HER2
    (flags can be combined)
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import survdiff

K = 10
D = "brca_metabric/"
USE_METH = "--methylation" in sys.argv
USE_EXT = "--extended" in sys.argv
TAG = "metabric" + ("_meth" if USE_METH else "") + ("_ext" if USE_EXT else "")

# ---------------------------------------------------------------- clinical + outcome
p = pd.read_csv(D + "data_clinical_patient.txt", sep="\t", comment="#")
s = pd.read_csv(D + "data_clinical_sample.txt", sep="\t", comment="#")
s = s.drop_duplicates("PATIENT_ID")
df = p.merge(s[["PATIENT_ID", "SAMPLE_ID", "TUMOR_STAGE", "TUMOR_SIZE", "GRADE", "ER_STATUS", "HER2_STATUS"]], on="PATIENT_ID", how="left")
df = df.rename(columns={"PATIENT_ID": "pid", "SAMPLE_ID": "sid"})
df["time"] = pd.to_numeric(df["RFS_MONTHS"], errors="coerce")
df["event"] = df["RFS_STATUS"].astype(str).str.startswith("1").astype(float)
df.loc[df["RFS_STATUS"].isna(), "event"] = np.nan
print("Patients in file:", len(df))

# ---------------------------------------------------------------- which patients have each data type
rna_path = D + "data_mrna_illumina_microarray.txt"
rna_cols = set(pd.read_csv(rna_path, sep="\t", nrows=0).columns[2:])

mut = pd.read_csv(D + "data_mutations.txt", sep="\t", comment="#", low_memory=False,
                  usecols=["Hugo_Symbol", "Tumor_Sample_Barcode", "Variant_Classification"])
sequenced = set(mut["Tumor_Sample_Barcode"])
try:
    gp = pd.read_csv(D + "data_gene_panel_matrix.txt", sep="\t", comment="#")
    mcol = [c for c in gp.columns if "mutation" in c.lower()]
    if mcol:
        sequenced |= set(gp.loc[gp[mcol[0]].notna(), gp.columns[0]])
except FileNotFoundError:
    pass

steps = [("with relapse-free outcome", df["time"].notna() & df["event"].notna() & (df["time"] > 0)),
         ("with RNA expression", df["sid"].isin(rna_cols)),
         ("with mutation sequencing", df["sid"].isin(sequenced))]
if USE_METH:
    meth_path = D + "data_methylation_promoters_rrbs.txt"
    meth_cols = set(pd.read_csv(meth_path, sep="\t", nrows=0).columns[1:])
    steps.append(("with methylation", df["sid"].isin(meth_cols)))
keep = pd.Series(True, index=df.index)
for label, mask in steps:
    keep &= mask
    print(f"  {label:28s} {int(keep.sum())}")
df = df[keep].reset_index(drop=True)
T, E = df["time"].values.astype(float), df["event"].values.astype(int)
sid = df["sid"]
print(f"Final cohort: {len(df)} patients, {E.sum()} relapse events ({E.mean():.1%})")

# ---------------------------------------------------------------- clinical features (same 7 as TCGA + missing-stage flag)
hist = df["HISTOLOGICAL_SUBTYPE"].astype(str)
clin = pd.DataFrame({
    "age": pd.to_numeric(df["AGE_AT_DIAGNOSIS"], errors="coerce"),
    "stage": pd.to_numeric(df["TUMOR_STAGE"], errors="coerce"),
    "t_size": np.log1p(pd.to_numeric(df["TUMOR_SIZE"], errors="coerce")),
    "nodes": np.log1p(pd.to_numeric(df["LYMPH_NODES_EXAMINED_POSITIVE"], errors="coerce")),
    "lobular": (hist == "Lobular").astype(float),
    "other_hist": (~hist.isin(["Lobular", "Ductal/NST", "nan"])).astype(float),
    "post": df["INFERRED_MENOPAUSAL_STATE"].map({"Post": 1.0, "Pre": 0.0}),
})
clin["stage_missing"] = clin["stage"].isna().astype(float)
if USE_EXT:
    clin["grade"] = pd.to_numeric(df["GRADE"], errors="coerce")
    clin["er_pos"] = df["ER_STATUS"].map({"Positive": 1.0, "Negative": 0.0})
    clin["her2_pos"] = df["HER2_STATUS"].map({"Positive": 1.0, "Negative": 0.0})
print("Clinical variables:", list(clin.columns))
clinical = clin.values.astype(float)

# ---------------------------------------------------------------- RNA (microarray, already log2)
rna = pd.read_csv(rna_path, sep="\t", usecols=["Hugo_Symbol"] + list(sid))
rna = rna.drop(columns=["Hugo_Symbol"]).dropna()
rna = rna[list(sid)].T.values.astype(np.float32)
if np.nanmax(rna) > 100:
    rna = np.log2(rna + 1)
print("RNA:", rna.shape)

# ---------------------------------------------------------------- mutations
mut = mut[(mut["Variant_Classification"] != "Silent") & mut["Tumor_Sample_Barcode"].isin(sid)]
mut = mut.drop_duplicates(["Tumor_Sample_Barcode", "Hugo_Symbol"])
tmb = np.log1p(mut.groupby("Tumor_Sample_Barcode").size().reindex(sid).fillna(0).values)
M = pd.crosstab(mut["Tumor_Sample_Barcode"], mut["Hugo_Symbol"]).reindex(sid).fillna(0)
M = M.loc[:, M.sum() >= 10]
print("Mutation genes kept:", M.shape[1])

# ---------------------------------------------------------------- methylation (optional)
if USE_METH:
    meth = pd.read_csv(meth_path, sep="\t", usecols=["Hugo_Symbol"] + list(sid))
    meth = meth.drop(columns=["Hugo_Symbol"])[list(sid)].T.values.astype(np.float32)
    print("Methylation:", meth.shape)


def c_index(time, event, risk):
    time, event, risk = map(np.asarray, (time, event, risk))
    num, den = 0.0, 0.0
    for i in np.where(event == 1)[0]:
        later = time > time[i]
        den += later.sum()
        num += (risk[i] > risk[later]).sum() + 0.5 * (risk[i] == risk[later]).sum()
    return num / den


def scale(a, b):
    m, sd = a.mean(axis=0), a.std(axis=0) + 1e-8
    return (a - m) / sd, (b - m) / sd


def cox(x, t, e):
    try:
        return PHReg(t, x, status=e).fit(disp=0).params
    except np.linalg.LinAlgError:
        return PHReg(t, x, status=e).fit_regularized(alpha=0.01, L1_wt=0).params


def corrected(d):
    k = len(d)
    se = np.sqrt((1 / k + 0.25) * d.var(ddof=1))
    h = stats.t.ppf(0.975, k - 1) * se
    pv = 2 * stats.t.sf(abs(d.mean() / se), k - 1)
    return d.mean() - h, d.mean() + h, pv


def impute(a, b):
    med = np.nanmedian(a, axis=0)
    med = np.where(np.isnan(med), 0, med)
    return np.where(np.isnan(a), med, a), np.where(np.isnan(b), med, b)


def reduce(x, tr, te):
    a, b = impute(x[tr], x[te])
    top = np.argsort(a.var(axis=0))[-2000:]
    a, b = scale(a[:, top], b[:, top])
    pca = PCA(n_components=K, random_state=0).fit(a)
    return scale(pca.transform(a), pca.transform(b))


def blocks(tr, te):
    out = {"clinical": scale(*impute(clinical[tr], clinical[te])), "rna": reduce(rna, tr, te)}
    top = M.iloc[tr].mean().sort_values(ascending=False).index[:K]
    a = np.column_stack([M[top].values[tr], tmb[tr]])
    b = np.column_stack([M[top].values[te], tmb[te]])
    out["mutations"] = scale(a, b)
    if USE_METH:
        out["methylation"] = reduce(meth, tr, te)
    return out


names = ["clinical", "rna", "mutations"] + (["methylation"] if USE_METH else [])
keys = names + ["concat", "late_avg", "late_stack"]
scores = {k: [] for k in keys}
weights = []
risk_clin = np.full((25, len(df)), np.nan)
risk_stack = np.full((25, len(df)), np.nan)

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
        scores[n].append(c_index(T[te], E[te], full_te[:, j]))

    _, z_te = scale(full_tr, full_te)
    scores["late_avg"].append(c_index(T[te], E[te], z_te.mean(axis=1)))
    z_oof, _ = scale(oof, oof)
    w = cox(z_oof, t, e)
    weights.append(w)
    scores["late_stack"].append(c_index(T[te], E[te], z_te @ w))
    risk_clin[f, te] = z_te[:, 0]
    risk_stack[f, te] = z_te @ w
    print(f"fold {f + 1} of 25", flush=True)

print(f"\n=== {TAG}: {len(df)} patients, {E.sum()} events ===")
base = np.array(scores["clinical"])
for name in keys:
    v = np.array(scores[name])
    d = v - base
    lo, hi, pv = corrected(d) if name != "clinical" else (0, 0, 1)
    print(f"{name:12s} {v.mean():.3f} +/- {v.std(ddof=1):.3f}  gain {d.mean():+.3f}  "
          f"CI {lo:+.3f} to {hi:+.3f}  p {pv:.3f}  better {int((d > 0).sum())}/25")
print("\nStacking weights:\n" + pd.Series(np.mean(weights, axis=0), index=names).round(3).to_string())

rc, rs = np.nanmean(risk_clin, axis=0), np.nanmean(risk_stack, axis=0)
for label, r in (("clinical", rc), ("stacking", rs)):
    g = (r > np.median(r)).astype(int)
    chi2, pv = survdiff(T, E, g)
    print(f"log-rank ({label}): chi2 {chi2:.1f}, p " + (f"{pv:.2e}" if pv > 0 else "< 1e-16"))

pd.DataFrame(scores).to_csv(f"scores_{TAG}.csv", index=False)
pd.DataFrame(weights, columns=names).to_csv(f"weights_{TAG}.csv", index=False)
pd.DataFrame({"pid": df["pid"], "time": T, "event": E, "risk_clinical": rc,
              "risk_stack": rs}).to_csv(f"risks_{TAG}.csv", index=False)
print(f"\nSaved scores_{TAG}.csv, weights_{TAG}.csv, risks_{TAG}.csv")