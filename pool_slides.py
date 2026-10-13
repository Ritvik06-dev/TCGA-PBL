import glob
import os

import h5py
import numpy as np
import pandas as pd

pids = list(pd.read_csv("cohort_final.csv")["bcr_patient_barcode"])
sums = {p: np.zeros(1536) for p in pids}
counts = {p: 0 for p in pids}

files = sorted(glob.glob("features/*.h5"))
for i, f in enumerate(files):
    p = os.path.basename(f)[:12]
    if p not in sums:
        continue
    with h5py.File(f, "r") as h:
        x = h["features"][:].reshape(-1, 1536)
    sums[p] += x.sum(axis=0)
    counts[p] += len(x)
    if i % 100 == 0:
        print(i, "of", len(files))

X = np.stack([sums[p] / counts[p] for p in pids]).astype(np.float32)
np.save("slide_mean.npy", X)
pd.DataFrame({"pid": pids, "patches": [counts[p] for p in pids]}).to_csv(
    "slide_pids.csv", index=False)
c = np.array(list(counts.values()))
print(X.shape, c.min(), int(np.median(c)), c.max())