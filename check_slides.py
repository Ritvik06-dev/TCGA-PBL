import glob
import os

import h5py
import pandas as pd

files = glob.glob("features/*.h5")
print(len(files))

with h5py.File(files[0], "r") as f:
    for k in f.keys():
        print(k, f[k].shape, f[k].dtype)

cohort = set(pd.read_csv("cohort.csv")["bcr_patient_barcode"])
have = {os.path.basename(p)[:12] for p in files}
print(len(have), len(have & cohort), len(cohort - have))