import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = "/Users/ram/programming/vibecoding/kaggle/churn"

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")

train["target"] = (train["Churn"] == "Yes").astype(int)
train["TotalCharges"] = pd.to_numeric(train["TotalCharges"], errors="coerce")
test["TotalCharges"]  = pd.to_numeric(test["TotalCharges"],  errors="coerce")

print("=" * 60)
print("1. 기본 정보")
print("=" * 60)
print(f"Train: {train.shape}, Test: {test.shape}")
print(f"Churn rate: {train['target'].mean():.4f} ({train['target'].sum():,} / {len(train):,})")
print(f"TotalCharges 결측 - Train: {train['TotalCharges'].isna().sum()}, Test: {test['TotalCharges'].isna().sum()}")

print("\n" + "=" * 60)
print("2. 피처별 Churn Rate")
print("=" * 60)
cat_cols = ["Contract","InternetService","PaymentMethod","PaperlessBilling",
            "OnlineSecurity","TechSupport","SeniorCitizen","Partner","Dependents"]
for col in cat_cols:
    print(f"\n[{col}]")
    cr = train.groupby(col)["target"].agg(["mean","count"]).sort_values("mean", ascending=False)
    cr.columns = ["churn_rate","count"]
    print(cr.to_string())

print("\n" + "=" * 60)
print("3. 수치형 피처 분포 (Train vs Test)")
print("=" * 60)
num_cols = ["tenure","MonthlyCharges","TotalCharges","SeniorCitizen"]
for col in num_cols:
    t_mean, t_std = train[col].mean(), train[col].std()
    ts_mean, ts_std = test[col].mean(), test[col].std()
    print(f"{col:20s} Train: {t_mean:.3f} ± {t_std:.3f}  |  Test: {ts_mean:.3f} ± {ts_std:.3f}  |  diff: {abs(t_mean-ts_mean):.4f}")

print("\n" + "=" * 60)
print("4. 범주형 분포 차이 (Train vs Test) — KL divergence 근사")
print("=" * 60)
cat_cols2 = ["Contract","InternetService","PaymentMethod","MultipleLines",
             "OnlineSecurity","StreamingTV","StreamingMovies"]
for col in cat_cols2:
    tr_dist = train[col].value_counts(normalize=True).sort_index()
    ts_dist = test[col].value_counts(normalize=True).sort_index()
    # 분포 차이 max abs
    diff = (tr_dist - ts_dist).abs().max()
    print(f"{col:25s} max_diff: {diff:.4f}")

print("\n" + "=" * 60)
print("5. Adversarial Validation (train vs test 구분 가능성)")
print("=" * 60)
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# train/test 구분 레이블
adv_train = train[num_cols].copy().fillna(0)
adv_test  = test[num_cols].copy().fillna(0)
X_adv = pd.concat([adv_train, adv_test], ignore_index=True)
y_adv = np.array([0]*len(train) + [1]*len(test))

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
adv_oof = np.zeros(len(X_adv))
for tr_idx, val_idx in skf.split(X_adv, y_adv):
    m = lgb.LGBMClassifier(n_estimators=100, verbose=-1, random_state=42)
    m.fit(X_adv.iloc[tr_idx], y_adv[tr_idx])
    adv_oof[val_idx] = m.predict_proba(X_adv.iloc[val_idx])[:,1]

adv_auc = roc_auc_score(y_adv, adv_oof)
print(f"Adversarial AUC: {adv_auc:.4f}")
print("(0.5에 가까울수록 train/test 분포 유사, 1.0에 가까울수록 다름)")
