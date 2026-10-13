import glob
import os

import h5py
import numpy as np
import pandas as pd

N = 1024

pids = list(pd.read_csv("cohort_final.csv")["bcr_patient_barcode"])
files = {}
for f in sorted(glob.glob("features/*.h5")):
    files.setdefault(os.path.basename(f)[:12], []).append(f)

bags = np.lib.format.open_memmap("bags.npy", mode="w+", dtype=np.float16,
                                 shape=(len(pids), N, 1536))
lens = np.zeros(len(pids), dtype=np.int64)
rng = np.random.default_rng(0)

for i, p in enumerate(pids):
    xs = []
    for f in files[p]:
        with h5py.File(f, "r") as h:
            xs.append(h["features"][:].reshape(-1, 1536))
    x = np.concatenate(xs)
    k = min(N, len(x))
    idx = rng.choice(len(x), k, replace=False)
    bags[i, :k] = x[idx]
    lens[i] = k
    if i % 100 == 0:
        print(i, "of", len(pids))

bags.flush()
pd.DataFrame({"pid": pids, "n": lens}).to_csv("bags_pids.csv", index=False)
print(bags.shape, lens.min(), int((lens < N).sum()))