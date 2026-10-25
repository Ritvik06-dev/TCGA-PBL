"""Quick look at the METABRIC files: columns, patient counts and survival events."""
import pandas as pd

D = "brca_metabric/"

# ---------- clinical (patient level)
p = pd.read_csv(D + "data_clinical_patient.txt", sep="\t", comment="#")
print("=== PATIENT FILE:", p.shape)
print(list(p.columns))
for c in ["OS_STATUS", "RFS_STATUS", "VITAL_STATUS", "INFERRED_MENOPAUSAL_STATE", "COHORT"]:
    if c in p.columns:
        print(f"\n{c}:\n{p[c].value_counts(dropna=False).to_string()}")
for c in ["OS_MONTHS", "RFS_MONTHS", "AGE_AT_DIAGNOSIS"]:
    if c in p.columns:
        print(f"\n{c}: median {p[c].median():.1f}, missing {p[c].isna().sum()}")

# ---------- clinical (sample level)
s = pd.read_csv(D + "data_clinical_sample.txt", sep="\t", comment="#")
print("\n=== SAMPLE FILE:", s.shape)
print(list(s.columns))
for c in ["TUMOR_STAGE", "GRADE", "ER_STATUS"]:
    if c in s.columns:
        print(f"\n{c}:\n{s[c].value_counts(dropna=False).to_string()}")

# ---------- RNA, methylation, mutations (headers only for the big files)
for name in ["data_mrna_illumina_microarray.txt", "data_methylation_promoters_rrbs.txt"]:
    cols = pd.read_csv(D + name, sep="\t", nrows=0).columns
    print(f"\n=== {name}: {len(cols)} columns, first 4: {list(cols[:4])}")

m = pd.read_csv(D + "data_mutations.txt", sep="\t", comment="#", low_memory=False)
print("\n=== MUTATIONS:", m.shape, "| samples:", m["Tumor_Sample_Barcode"].nunique())
print("Top 10 genes:\n" + m["Hugo_Symbol"].value_counts().head(10).to_string())