"""
Trial 008 v3: Full 171 Pairwise (factorize only) + Multiclass TE on cats only
- 171 pairwise: factorize, NO target encoding on pairs
- Multiclass TE: cat_cols (8) only, 3 class = 24 TE features per fold
- Binary threshold features (Deotte formula)
- Original data: TE source only (no append)
- XGB + LGBM 2-model (CatBoost 제외 → 속도 2배)
- Threshold optimization
"""

import gc
import json
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import lightgbm as lgb
import sys; sys.stdout.reconfigure(line_buffering=True)

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
OUT_DIR = Path(__file__).resolve().parent

# ============ Load ============

train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
orig = pd.read_csv(DATA_DIR / "original" / "irrigation_prediction.csv")

target_map = {"Low": 0, "Medium": 1, "High": 2}
target_inv = {v: k for k, v in target_map.items()}
y = train["Irrigation_Need"].map(target_map).values

cat_cols = ["Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
            "Irrigation_Type", "Water_Source", "Mulching_Used", "Region"]
num_cols = ["Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
            "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
            "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm"]
all_raw_cols = cat_cols + num_cols

print(f"Train: {train.shape}, Test: {test.shape}")

# ============ Binary Threshold Features (Deotte) ============

for df in [train, test]:
    df["soil_lt_25"] = (df["Soil_Moisture"] < 25).astype(int)
    df["temp_gt_30"] = (df["Temperature_C"] > 30).astype(int)
    df["rain_lt_300"] = (df["Rainfall_mm"] < 300).astype(int)
    df["wind_gt_10"] = (df["Wind_Speed_kmh"] > 10).astype(int)

binary_cols = ["soil_lt_25", "temp_gt_30", "rain_lt_300", "wind_gt_10"]

# ============ Original Data TE (external) ============

orig["target_num"] = orig["Irrigation_Need"].map(target_map)
te_orig_cols = []
for col in all_raw_cols:
    te_map = orig.groupby(col)["target_num"].mean()
    col_name = f"{col}_te_orig"
    train[col_name] = train[col].map(te_map).fillna(te_map.mean())
    test[col_name] = test[col].map(te_map).fillna(te_map.mean())
    te_orig_cols.append(col_name)

# ============ ALL 171 Pairwise: factorize only ============

pair_cols = []
for c1, c2 in combinations(all_raw_cols, 2):
    col_name = f"{c1}_x_{c2}"
    combined = pd.concat([
        train[c1].astype(str) + "_" + train[c2].astype(str),
        test[c1].astype(str) + "_" + test[c2].astype(str)
    ])
    codes, _ = pd.factorize(combined)
    train[col_name] = codes[:len(train)]
    test[col_name] = codes[len(train):]
    pair_cols.append(col_name)

print(f"Added {len(pair_cols)} pairwise features (factorize only)")

# ============ Label encode categoricals ============

le_cols = []
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train[col].astype(str), test[col].astype(str)])
    le.fit(combined)
    col_le = f"{col}_le"
    train[col_le] = le.transform(train[col].astype(str))
    test[col_le] = le.transform(test[col].astype(str))
    le_cols.append(col_le)

# ============ Base features ============

base_features = num_cols + le_cols + binary_cols + te_orig_cols + pair_cols
print(f"Base features: {len(base_features)}")

# ============ Sample Weights ============

sample_weights = compute_sample_weight("balanced", y)

# ============ CV ============

N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_xgb_arr = np.zeros((len(train), 3))
oof_lgbm_arr = np.zeros((len(train), 3))
test_xgb_arr = np.zeros((len(test), 3))
test_lgbm_arr = np.zeros((len(test), 3))

for fold, (train_idx, val_idx) in enumerate(skf.split(train, y)):
    print(f"\n{'='*40} Fold {fold} {'='*40}")
    y_tr, y_val = y[train_idx], y[val_idx]
    sw_tr = sample_weights[train_idx]

    # --- Multiclass TE on cat_cols only (8 cols × 3 class = 24 TE) ---
    te_col_names = []
    te_train = {}
    te_test = {}

    for col in cat_cols:
        for cls in range(3):
            col_te = f"{col}_te{cls}"
            cls_mean = (y_tr == cls).mean()
            vals = train.iloc[train_idx][f"{col}_le"]
            labels = y_tr
            df_grp = pd.DataFrame({"val": vals, "is_cls": (labels == cls).astype(int)})
            grp_mean = df_grp.groupby("val")["is_cls"].mean()
            grp_count = df_grp.groupby("val")["is_cls"].count()
            m = 100
            smoothed = (grp_mean * grp_count + cls_mean * m) / (grp_count + m)
            te_train[col_te] = train[f"{col}_le"].map(smoothed).fillna(cls_mean).values
            te_test[col_te] = test[f"{col}_le"].map(smoothed).fillna(cls_mean).values
            te_col_names.append(col_te)

    # Build feature matrix
    all_cols = base_features + te_col_names
    X_tr_base = train[base_features].iloc[train_idx].copy()
    X_val_base = train[base_features].iloc[val_idx].copy()
    X_te_base = test[base_features].copy()

    for col_te in te_col_names:
        X_tr_base[col_te] = te_train[col_te][train_idx]
        X_val_base[col_te] = te_train[col_te][val_idx]
        X_te_base[col_te] = te_test[col_te]

    print(f"  Features: {X_tr_base.shape[1]} (base {len(base_features)} + TE {len(te_col_names)})")

    # --- XGBoost ---
    model_xgb = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_estimators=5000,
        learning_rate=0.01,
        max_depth=6,
        min_child_weight=30,
        subsample=0.8,
        colsample_bytree=0.6,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        tree_method="hist",
        max_bin=1024,
        verbosity=0,
    )
    model_xgb.fit(
        X_tr_base, y_tr,
        sample_weight=sw_tr,
        eval_set=[(X_val_base, y_val)],
        verbose=500,
    )
    oof_xgb_arr[val_idx] = model_xgb.predict_proba(X_val_base)
    test_xgb_arr += model_xgb.predict_proba(X_te_base) / N_FOLDS
    print(f"  XGB  bal_acc: {balanced_accuracy_score(y_val, oof_xgb_arr[val_idx].argmax(1)):.6f}")
    del model_xgb; gc.collect()

    # --- LightGBM ---
    model_lgb = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        metric="multi_logloss",
        verbosity=-1,
        n_estimators=5000,
        learning_rate=0.01,
        num_leaves=127,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.6,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        is_unbalance=True,
    )
    model_lgb.fit(
        X_tr_base, y_tr,
        sample_weight=sw_tr,
        eval_set=[(X_val_base, y_val)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)],
    )
    oof_lgbm_arr[val_idx] = model_lgb.predict_proba(X_val_base)
    test_lgbm_arr += model_lgb.predict_proba(X_te_base) / N_FOLDS
    print(f"  LGBM bal_acc: {balanced_accuracy_score(y_val, oof_lgbm_arr[val_idx].argmax(1)):.6f}")
    del model_lgb; gc.collect()

# ============ Ensemble + Threshold ============

for name, oof in [("XGB", oof_xgb_arr), ("LGBM", oof_lgbm_arr)]:
    print(f"\n{name} OOF balanced_accuracy: {balanced_accuracy_score(y, oof.argmax(1)):.6f}")

best_bal_acc = 0
best_w = (1, 1)
for w1 in range(1, 10):
    for w2 in range(0, 10):
        if w1 + w2 == 0:
            continue
        total = w1 + w2
        oof_ens = (w1 * oof_xgb_arr + w2 * oof_lgbm_arr) / total
        bal_acc = balanced_accuracy_score(y, oof_ens.argmax(1))
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_w = (w1, w2)

print(f"\nBest weights (xgb:lgbm): {best_w}")
print(f"Best ensemble OOF balanced_accuracy: {best_bal_acc:.6f}")

total = sum(best_w)
oof_ens = (best_w[0] * oof_xgb_arr + best_w[1] * oof_lgbm_arr) / total
test_ens = (best_w[0] * test_xgb_arr + best_w[1] * test_lgbm_arr) / total

best_threshold_acc = best_bal_acc
best_class_w = (1.0, 1.0, 1.0)
for w_low in np.arange(0.7, 1.3, 0.05):
    for w_med in np.arange(0.5, 1.3, 0.05):
        for w_high in np.arange(1.0, 4.0, 0.1):
            adjusted = oof_ens.copy()
            adjusted[:, 0] *= w_low
            adjusted[:, 1] *= w_med
            adjusted[:, 2] *= w_high
            preds = adjusted.argmax(1)
            bal_acc = balanced_accuracy_score(y, preds)
            if bal_acc > best_threshold_acc:
                best_threshold_acc = bal_acc
                best_class_w = (w_low, w_med, w_high)

print(f"\nThreshold class weights: {best_class_w}")
print(f"After threshold: balanced_accuracy {best_threshold_acc:.6f} (was {best_bal_acc:.6f})")

test_adjusted = test_ens.copy()
test_adjusted[:, 0] *= best_class_w[0]
test_adjusted[:, 1] *= best_class_w[1]
test_adjusted[:, 2] *= best_class_w[2]

# ============ Save ============

np.save(OUT_DIR / "oof_preds.npy", np.stack([oof_xgb_arr, oof_lgbm_arr]))
np.save(OUT_DIR / "test_preds.npy", np.stack([test_xgb_arr, test_lgbm_arr]))

sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
sub["Irrigation_Need"] = [target_inv[p] for p in test_adjusted.argmax(1)]
sub.to_csv(OUT_DIR / "submission.csv", index=False)

sub_no = pd.read_csv(DATA_DIR / "sample_submission.csv")
sub_no["Irrigation_Need"] = [target_inv[p] for p in test_ens.argmax(1)]
sub_no.to_csv(OUT_DIR / "submission_no_threshold.csv", index=False)

results = {
    "trial": "trial_008_multiclass_te_fullpair",
    "metric": "balanced_accuracy",
    "oof_bal_acc_xgb": round(balanced_accuracy_score(y, oof_xgb_arr.argmax(1)), 6),
    "oof_bal_acc_lgbm": round(balanced_accuracy_score(y, oof_lgbm_arr.argmax(1)), 6),
    "oof_bal_acc_ensemble": round(best_bal_acc, 6),
    "oof_bal_acc_threshold": round(best_threshold_acc, 6),
    "ensemble_weights_xgb_lgbm": list(best_w),
    "threshold_class_weights": [round(w, 3) for w in best_class_w],
    "n_base_features": len(base_features),
    "n_te_per_fold": len(te_col_names),
    "n_total_features": len(base_features) + len(te_col_names),
}
with open(OUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone. Files saved to {OUT_DIR}")
print(f"Prediction distribution: {pd.Series([target_inv[p] for p in test_adjusted.argmax(1)]).value_counts().to_dict()}")
