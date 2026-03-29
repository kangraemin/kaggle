import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import psutil, os, threading, json, warnings
warnings.filterwarnings("ignore")

# ── 메모리 가드 ────────────────────────────────────────────────────────────────
def memory_guard():
    import time
    while True:
        if psutil.virtual_memory().percent > 85:
            print(f"\n⚠️ RAM {psutil.virtual_memory().percent:.1f}% — 종료")
            os.kill(os.getpid(), 9)
        time.sleep(3)
threading.Thread(target=memory_guard, daemon=True).start()

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}")

N_FOLDS = 5; SEEDS = [42, 0, 1, 2]; ALPHA = 10  # seed 4개만 (메모리 절약)
DATA_DIR = "/Users/ram/programming/vibecoding/kaggle/churn"
OUT_DIR  = f"{DATA_DIR}/submissions/sub_07/trial_042_mlp"

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")
train["target"] = (train["Churn"] == "Yes").astype(int)

for df in [train, test]:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    df["AvgMonthlyCharge"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["ChargeGap"]        = df["MonthlyCharges"] - df["AvgMonthlyCharge"]
    df["ChargeRatio"]      = df["MonthlyCharges"] / (df["TotalCharges"] + 1)
    df["is_electronic"]    = (df["PaymentMethod"] == "Electronic check").astype(int)
    df["is_fiber"]         = (df["InternetService"] == "Fiber optic").astype(int)
    df["is_monthly"]       = (df["Contract"] == "Month-to-month").astype(int)
    df["highest_risk"]     = df["is_monthly"] * df["is_fiber"] * df["is_electronic"]
    df["senior_electronic"]= df["SeniorCitizen"] * df["is_electronic"]

cat_cols = ["gender","Partner","Dependents","PhoneService","MultipleLines",
            "InternetService","OnlineSecurity","OnlineBackup","DeviceProtection",
            "TechSupport","StreamingTV","StreamingMovies","Contract",
            "PaperlessBilling","PaymentMethod"]
num_cols = [c for c in train.columns if c not in ["id","Churn","target"] + cat_cols]
y = train["target"]

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

all_oof = np.zeros(len(train)); all_test = np.zeros(len(test))

for seed in SEEDS:
    torch.manual_seed(seed)
    np.random.seed(seed)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    seed_oof = np.zeros(len(train)); seed_test = np.zeros(len(test))

    for fold,(tr_idx,val_idx) in enumerate(skf.split(train[num_cols],y)):
        tr_df, val_df = train.iloc[tr_idx], train.iloc[val_idx]
        gm = tr_df["target"].mean()
        te_tr,te_val,te_te = {},{},{}
        for col in cat_cols:
            stats = tr_df.groupby(col)["target"].agg(["sum","count"])
            smooth = (stats["sum"] + ALPHA*gm) / (stats["count"] + ALPHA)
            te_tr[f"te_{col}"]  = tr_df[col].map(smooth).fillna(gm).values
            te_val[f"te_{col}"] = val_df[col].map(smooth).fillna(gm).values
            te_te[f"te_{col}"]  = test[col].map(smooth).fillna(gm).values
        X_tr  = pd.concat([tr_df[num_cols].reset_index(drop=True),  pd.DataFrame(te_tr)],  axis=1).values.astype(np.float32)
        X_val = pd.concat([val_df[num_cols].reset_index(drop=True), pd.DataFrame(te_val)], axis=1).values.astype(np.float32)
        X_te  = pd.concat([test[num_cols].reset_index(drop=True),   pd.DataFrame(te_te)],  axis=1).values.astype(np.float32)

        scaler = StandardScaler()
        X_tr  = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)
        X_te  = scaler.transform(X_te)

        y_tr = y.iloc[tr_idx].values.astype(np.float32)

        ds_tr = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr))
        dl_tr = DataLoader(ds_tr, batch_size=4096, shuffle=True)

        model = MLP(X_tr.shape[1]).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        best_auc, best_state, patience = 0, None, 0
        for epoch in range(50):
            model.train()
            for xb, yb in dl_tr:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_pred = torch.sigmoid(model(torch.FloatTensor(X_val).to(DEVICE))).cpu().numpy()
            auc = roc_auc_score(y.iloc[val_idx], val_pred)
            if auc > best_auc:
                best_auc = auc; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}; patience = 0
            else:
                patience += 1
                if patience >= 5: break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            seed_oof[val_idx] = torch.sigmoid(model(torch.FloatTensor(X_val).to(DEVICE))).cpu().numpy()
            seed_test += torch.sigmoid(model(torch.FloatTensor(X_te).to(DEVICE))).cpu().numpy() / N_FOLDS

        # 메모리 해제
        del model; torch.mps.empty_cache() if DEVICE.type == "mps" else None

    seed_auc = roc_auc_score(y, seed_oof)
    print(f"SEED {seed}: {seed_auc:.5f}  RAM: {psutil.virtual_memory().percent:.1f}%")
    all_oof  += seed_oof  / len(SEEDS)
    all_test += seed_test / len(SEEDS)

oof_auc = roc_auc_score(y, all_oof)
print(f"\nOOF AUC: {oof_auc:.5f}")
np.save(f"{OUT_DIR}/oof_preds.npy", all_oof)
np.save(f"{OUT_DIR}/test_preds.npy", all_test)
pd.DataFrame({"id":test["id"],"Churn":all_test}).to_csv(f"{OUT_DIR}/trial_042_mlp.csv",index=False)
json.dump({"id":"042","status":"done","val_score":round(oof_auc,5),
           "notes":"MLP 3-layer (256-128-64) + MPS + smoothed TE + 4seeds"},
          open(f"{OUT_DIR}/results.json","w"),indent=2)
print("Done.")
