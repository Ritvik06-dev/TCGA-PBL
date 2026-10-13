import glob

import h5py
import pandas as pd

pd.set_option("display.width", 200)

c = pd.read_excel("cdr.xlsx")
c = c[c["type"] == "BRCA"]
print(c.shape)
print(list(c.columns))
print(c.head(3).T)

r = pd.read_csv("TCGA_Reports.csv")
print(r.shape)
print(r["text"][0][:800])

g = "brca_tcga_pan_can_atlas_2018/"
print(pd.read_csv(g + "data_mrna_seq_v2_rsem.txt", sep="\t", nrows=5).iloc[:, :6])
print(pd.read_csv(g + "data_mutations.txt", sep="\t", comment="#", nrows=5).iloc[:, :10])

f = sorted(glob.glob("features/*.h5"))[0]
with h5py.File(f, "r") as h:
    print(f)
    for k in h:
        print(k, h[k].shape)
    print(h["features"][0, 0, :8])