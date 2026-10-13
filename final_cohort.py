import glob
import os

import pandas as pd

cohort = pd.read_csv("cohort.csv")
have = {os.path.basename(p)[:12] for p in glob.glob("features/*.h5")}

missing = cohort[~cohort["bcr_patient_barcode"].isin(have)]
print(missing[["bcr_patient_barcode", "OS", "PFI"]])

final = cohort[cohort["bcr_patient_barcode"].isin(have)]
print(len(final), final["OS"].sum(), final["PFI"].sum())
final.to_csv("cohort_final.csv", index=False)