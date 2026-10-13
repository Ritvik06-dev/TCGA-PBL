import pandas as pd
import requests

cdr = pd.read_excel("cdr.xlsx")
brca = cdr[cdr["type"] == "BRCA"]
rep = pd.read_csv("TCGA_Reports.csv")
print(rep.columns)

rep["pid"] = rep.iloc[:, 0].str[:12]
both = set(brca["bcr_patient_barcode"]) & set(rep["pid"])
print(len(brca), len(both))

p = brca[brca["bcr_patient_barcode"].isin(both)]
print(p["OS"].value_counts()) #OS is overall survival, where 1 means the patient died.
print(p["PFI"].value_counts()) #PFI is progression-free interval, where 1 means the cancer progressed or came back.
print(p["OS.time"].median()) #OS.time is follow-up in days.

g = "brca_tcga_pan_can_atlas_2018/"
rna = pd.read_csv(g + "data_mrna_seq_v2_rsem.txt", sep="\t", nrows=5)
rna_p = {c[:12] for c in rna.columns[2:]}
mut = pd.read_csv(g + "data_mutations.txt", sep="\t",
                  usecols=["Tumor_Sample_Barcode"], comment="#")
mut_p = set(mut["Tumor_Sample_Barcode"].str[:12])
print(len(both & rna_p), len(both & rna_p & mut_p))

f = {"op": "and", "content": [
    {"op": "=", "content": {"field": "cases.project.project_id", "value": "TCGA-BRCA"}},
    {"op": "=", "content": {"field": "data_type", "value": "Slide Image"}},
    {"op": "=", "content": {"field": "experimental_strategy", "value": "Diagnostic Slide"}},
]}
r = requests.post("https://api.gdc.cancer.gov/files", json={
    "filters": f,
    "fields": "file_name,file_size,cases.submitter_id",
    "size": 5000,
})
hits = r.json()["data"]["hits"]
slide_p = {h["cases"][0]["submitter_id"] for h in hits}
gb = sum(h["file_size"] for h in hits) / 1e9
all4 = both & rna_p & mut_p & slide_p
print(len(hits), len(slide_p), round(gb), len(all4))

c = brca[brca["bcr_patient_barcode"].isin(all4)]
print(c["OS"].sum(), c["PFI"].sum())
c.to_csv("cohort.csv", index=False)