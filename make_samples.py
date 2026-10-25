"""Make data_samples.xlsx: a small sample of every raw data type used in the project.
Run from the tcga folder:  python make_samples.py
"""
import glob
import os

import h5py
import numpy as np
import pandas as pd

OUT = "data_samples.xlsx"
CBIO = "brca_tcga_pan_can_atlas_2018"
N_PATIENTS = 6

# ---------- choose sample patients from the final cohort: half with an event, half without
cohort = pd.read_csv("cohort_final.csv")
ev = cohort[cohort["PFI"] == 1]["bcr_patient_barcode"].head(N_PATIENTS // 2)
no = cohort[cohort["PFI"] == 0]["bcr_patient_barcode"].head(N_PATIENTS - len(ev))
pids = list(ev) + list(no)
print("Sample patients:", pids)
sheets = {}
notes = []

# ---------- 1. outcomes + clinical (TCGA-CDR)
cdr = pd.read_excel("cdr.xlsx")
cdr = cdr.loc[:, ~cdr.columns.astype(str).str.startswith("Unnamed")]
keep = [c for c in ["bcr_patient_barcode", "type", "age_at_initial_pathologic_diagnosis", "gender",
                    "ajcc_pathologic_tumor_stage", "histological_type", "menopause_status",
                    "OS", "OS.time", "PFI", "PFI.time"] if c in cdr.columns]
sheets["1_Clinical_CDR"] = cdr[cdr["bcr_patient_barcode"].isin(pids)][keep]
notes.append(("1_Clinical_CDR", "cdr.xlsx (TCGA-CDR)", f"{len(cdr)} patients x {cdr.shape[1]} columns (all cancers)",
              "Age, stage, histology, menopause + outcomes. PFI = 1 means progression; PFI.time = days."))

# ---------- 2. extra clinical (cBioPortal patient table)
cp = pd.read_csv(os.path.join(CBIO, "data_clinical_patient.txt"), sep="\t", comment="#")
sheets["2_Clinical_cBioPortal"] = cp[cp["PATIENT_ID"].isin(pids)]
notes.append(("2_Clinical_cBioPortal", "data_clinical_patient.txt", f"{cp.shape[0]} patients x {cp.shape[1]} columns",
              "Source of T stage, N stage and subtype."))

# ---------- 3. RNA expression (genes x samples)
rna_path = os.path.join(CBIO, "data_mrna_seq_v2_rsem.txt")
header = pd.read_csv(rna_path, sep="\t", nrows=0).columns
cols = [c for c in header if c[:12] in pids and c.endswith("-01")]
rna = pd.read_csv(rna_path, sep="\t", usecols=["Hugo_Symbol"] + cols)
n_genes = len(rna)
sheets["3_RNA_expression"] = rna.dropna(subset=["Hugo_Symbol"]).head(25).round(2)
notes.append(("3_RNA_expression", "data_mrna_seq_v2_rsem.txt", f"{n_genes} genes x {len(header) - 2} samples",
              "RSEM expression counts; first 25 genes shown. We take log2, keep the top 2,000 variable genes, then PCA to 10."))

# ---------- 4. mutations (one row per mutation)
mut = pd.read_csv(os.path.join(CBIO, "data_mutations.txt"), sep="\t", comment="#", low_memory=False)
mcols = [c for c in ["Hugo_Symbol", "Tumor_Sample_Barcode", "Variant_Classification", "Variant_Type",
                     "Chromosome", "Start_Position", "HGVSp_Short"] if c in mut.columns]
m = mut[mut["Tumor_Sample_Barcode"].str[:12].isin(pids)][mcols]
sheets["4_Mutations"] = m.groupby(m["Tumor_Sample_Barcode"].str[:12]).head(5)
notes.append(("4_Mutations", "data_mutations.txt", f"{len(mut)} mutations in {mut['Tumor_Sample_Barcode'].nunique()} samples",
              "Up to 5 mutations per patient shown. We use the 10 most mutated genes (yes/no) + mutation count."))

# ---------- 5. pathology reports (free text)
rep = pd.read_csv("TCGA_Reports.csv")
rep["patient"] = rep["patient_filename"].str[:12]
r = rep[rep["patient"].isin(pids)].head(3).copy()
r["characters"] = r["text"].str.len()
sheets["5_Pathology_reports"] = r[["patient", "patient_filename", "characters", "text"]]
notes.append(("5_Pathology_reports", "TCGA_Reports.csv", f"{len(rep)} reports",
              "Full report text for 3 patients. We turn text into TF-IDF numbers, then SVD to 10."))

# ---------- 6. slide features (one .h5 file per slide)
files = sorted(f for f in glob.glob(os.path.join("features", "*.h5")) if os.path.basename(f)[:12] in pids)
if files:
    with h5py.File(files[0], "r") as h:
        feats = np.asarray(h["features"]).squeeze()
        coords = np.asarray(h["coords"]).squeeze() if "coords" in h else None
        keys = list(h.keys())
    df = pd.DataFrame(feats[:15, :12].round(4), columns=[f"feature_{i}" for i in range(12)])
    if coords is not None and coords.ndim == 2 and coords.shape[0] == feats.shape[0]:
        df.insert(0, "patch_x", coords[:15, 0]); df.insert(1, "patch_y", coords[:15, 1])
    df.insert(0, "patch", range(1, len(df) + 1))
    sheets["6_Slide_features"] = df
    notes.append(("6_Slide_features", os.path.basename(files[0]),
                  f"{feats.shape[0]} patches x {feats.shape[1]} features (keys: {', '.join(keys)})",
                  "Each row = one 256x256 tissue patch at 20x, turned into 1,536 numbers by UNI2-h. First 15 patches x 12 of 1,536 features shown."))
else:
    print("No .h5 file found for the sample patients in features/ - slide sheet skipped")

# ---------- write workbook
readme = pd.DataFrame(notes, columns=["Sheet", "Raw file", "Full size", "What it shows"])
readme.loc[len(readme)] = ["(all)", "", f"Sample patients: {', '.join(pids)}",
                           "Same patients across sheets, joined by the 12-character barcode. Raw slide images (.svs) were not "
                           "downloaded; we used precomputed UNI2-h patch features instead."]
with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
    readme.to_excel(xw, sheet_name="README", index=False)
    for name, df in sheets.items():
        df.to_excel(xw, sheet_name=name, index=False)
    for ws in xw.book.worksheets:
        ws.freeze_panes = "A2"
        for col in ws.columns:
            width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:30])
            ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 60)
    from openpyxl.styles import Alignment
    for ws_name, col_letter in (("README", "D"), ("5_Pathology_reports", "D")):
        ws = xw.book[ws_name]
        ws.column_dimensions[col_letter].width = 110
        for cell in ws[col_letter]:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
print("Saved", OUT, "with sheets:", ["README"] + list(sheets))