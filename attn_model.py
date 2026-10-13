import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

EPOCHS = 20
BATCH = 64
N_TRAIN = 512
dev = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv("cohort_final.csv")
df = df[["bcr_patient_barcode", "PFI", "PFI.time"]].copy()
df.columns = ["pid", "event", "time"]
df = df.dropna(subset=["time", "event"])
df = df[df["time"] > 0].reset_index(drop=True)
T, E = df["time"].values, df["event"].values

bp = pd.read_csv("bags_pids.csv")
row = {p: i for i, p in enumerate(bp["pid"])}
rows = [row[p] for p in df["pid"]]
bags = torch.from_numpy(np.load("bags.npy", mmap_mode="r")[rows]).to(dev)
lens = torch.tensor(bp["n"].values[rows], device=dev)
N = bags.shape[1]
mask = torch.arange(N, device=dev)[None, :] < lens[:, None]
tt = torch.tensor(T, dtype=torch.float32, device=dev)
ee = torch.tensor(E, dtype=torch.float32, device=dev)
print(dev, tuple(bags.shape), int(E.sum()))


class ABMIL(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(1536, 256), nn.ReLU(), nn.Dropout(0.25))
        self.attn = nn.Sequential(nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1))
        self.out = nn.Linear(256, 1)

    def forward(self, x, m):
        h = self.fc(x)
        a = self.attn(h).squeeze(-1).masked_fill(~m, float("-inf"))
        w = torch.softmax(a, dim=1)
        return self.out((w.unsqueeze(-1) * h).sum(dim=1)).squeeze(-1)


def cox_loss(risk, time, event):
    order = torch.argsort(time, descending=True)
    risk, event = risk[order], event[order]
    return -((risk - torch.logcumsumexp(risk, dim=0)) * event).sum() / event.sum()


def c_index(time, event, risk):
    time, event, risk = map(np.asarray, (time, event, risk))
    num, den = 0.0, 0.0
    for i in np.where(event == 1)[0]:
        later = time > time[i]
        den += later.sum()
        num += (risk[i] > risk[later]).sum() + 0.5 * (risk[i] == risk[later]).sum()
    return num / den


def train(idx, seed):
    torch.manual_seed(seed)
    idx = torch.tensor(idx, device=dev)
    model = ABMIL().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
    for _ in range(EPOCHS):
        model.train()
        for b in idx[torch.randperm(len(idx), device=dev)].split(BATCH):
            if ee[b].sum() == 0:
                continue
            p = torch.randperm(N, device=dev)[:N_TRAIN]
            risk = model(bags[b][:, p].float(), mask[b][:, p])
            loss = cox_loss(risk, tt[b], ee[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def predict(model):
    model.eval()
    out = []
    with torch.no_grad():
        for b in torch.arange(len(df), device=dev).split(128):
            out.append(model(bags[b].float(), mask[b]).cpu().numpy())
    return np.concatenate(out)


n = len(df)
full = np.zeros((25, n), dtype=np.float32)
oof = np.full((25, n), np.nan, dtype=np.float32)
scores = []

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
for f, (tr, te) in enumerate(cv.split(df, E)):
    full[f] = predict(train(tr, 100 * f))
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for j, (itr, ite) in enumerate(inner.split(tr, E[tr])):
        oof[f, tr[ite]] = predict(train(tr[itr], 100 * f + j + 1))[tr[ite]]
    scores.append(c_index(T[te], E[te], full[f, te]))
    print("fold", f + 1, "of 25", round(scores[-1], 3))

print("slide_attn", round(np.mean(scores), 3), round(np.std(scores), 3))
np.savez("slide_attn_risks.npz", full=full, oof=oof)
pd.DataFrame({"slide_attn": scores}).to_csv("scores_attn.csv", index=False)