import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RepeatedStratifiedKFold
from statsmodels.duration.hazard_regression import PHReg

N_COMP = 10

df = pd.read_csv("cohort_final.csv")
df = df[["bcr_patient_barcode", "age_at_initial_pathologic_diagnosis",
         "ajcc_pathologic_tumor_stage", "PFI", "PFI.time"]].copy()
df.columns = ["pid", "age", "stage", "event", "time"]


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


def c_index(time, event, risk):
    time, event, risk = map(np.asarray, (time, event, risk))
    num, den = 0.0, 0.0
    for i in np.where(event == 1)[0]:
        later = time > time[i]
        den += later.sum()
        num += (risk[i] > risk[later]).sum() + 0.5 * (risk[i] == risk[later]).sum()
    return num / den


df["stage"] = df["stage"].map(stage_num)
df = df.dropna(subset=["time", "event"])
df = df[df["time"] > 0].reset_index(drop=True)

rep = pd.read_csv("TCGA_Reports.csv")
rep["pid"] = rep["patient_filename"].str[:12]
rep = rep.groupby("pid")["text"].apply(" ".join)
docs = rep.reindex(df["pid"]).fillna("").values
words = np.array([len(d.split()) for d in docs])
print(len(docs), int((words == 0).sum()), int(np.median(words)))

clin = ["age", "stage"]
tcols = ["t%d" % i for i in range(N_COMP)]
sets = {"clinical": clin, "text": tcols, "clinical+text": clin + tcols}
scores = {k: [] for k in sets}

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
for tr, te in cv.split(df, df["event"]):
    a, b = df.iloc[tr].copy(), df.iloc[te].copy()
    for c in clin:
        m = a[c].median()
        a[c] = a[c].fillna(m)
        b[c] = b[c].fillna(m)

    vec = TfidfVectorizer(min_df=5, max_features=5000, sublinear_tf=True,
                          stop_words="english", ngram_range=(1, 2))
    svd = TruncatedSVD(n_components=N_COMP, random_state=0)
    ta = svd.fit_transform(vec.fit_transform(docs[tr]))
    tb = svd.transform(vec.transform(docs[te]))
    for i, t in enumerate(tcols):
        a[t], b[t] = ta[:, i], tb[:, i]

    for name, cols in sets.items():
        m, s = a[cols].mean(), a[cols].std() + 1e-8
        fit = PHReg(a["time"], (a[cols] - m) / s, status=a["event"]).fit(disp=0)
        risk = ((b[cols] - m) / s).values @ fit.params
        scores[name].append(c_index(b["time"], b["event"], risk))

for name, v in scores.items():
    print(name, round(np.mean(v), 3), round(np.std(v), 3))
d = np.array(scores["clinical+text"]) - np.array(scores["clinical"])
print("gain", round(d.mean(), 3), "folds better:", int((d > 0).sum()), "of", len(d))
pd.DataFrame(scores).to_csv("scores_text.csv", index=False)