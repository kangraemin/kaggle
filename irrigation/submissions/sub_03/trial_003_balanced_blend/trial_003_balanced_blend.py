"""
Trial 003: Balanced Accuracy + Original Data Blend + Pairwise Interaction TE
- Fix metric: balanced_accuracy_score (not accuracy)
- class_weight='balanced' for all models
- Blend original 10K data
- Pairwise categorical interaction + target encoding
- Enhanced domain FE
- Threshold optimization for minority class
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
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
OUT_DIR = Path(__file__).resolve().parent

# ============ Load & Blend ============

train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
orig = pd.read_csv(DATA_DIR / "original" / "irrigation_prediction.csv")

# Original data에 id 추가 (음수로 구분)
orig["id"] = range(-len(orig), 0)

# Blend: train + original
train = pd.concat([train, orig], ignore_index=True)
print(f"After blending: train {train.shape}, test {test.shape}")

target_map = {"Low": 0, "Medium": 1, "High": 2}
target_inv = {v: k for k, v in target_map.items()}
y = train["Irrigation_Need"].map(target_map).values

cat_cols = ["Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
            "Irrigation_Type", "Water_Source", "Mulching_Used", "Region"]
num_cols = ["Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
            "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
            "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm"]


# ============ Feature Engineering ============

def add_features(df):
    # Domain FE (enhanced)
    df["Total_Water_Input"] = df["Rainfall_mm"] + df["Previous_Irrigation_mm"]
    df["Evaporative_Stress"] = (df["Temperature_C"] * df["Sunlight_Hours"] * df["Wind_Speed_kmh"]) / (df["Humidity"] + 1)
    df["ET_proxy"] = df["Temperature_C"] * df["Wind_Speed_kmh"] / (df["Humidity"] + 1)
    df["Water_Balance"] = df["Total_Water_Input"] - df["Evaporative_Stress"] * df["Field_Area_hectare"]
    df["Soil_Health"] = df["Organic_Carbon"] * df["Soil_Moisture"] / (df["Electrical_Conductivity"] + 0.1)
    df["Soil_Salinity_Risk"] = df["Electrical_Conductivity"] * df["Temperature_C"] / (df["Rainfall_mm"] + 1)
    df["pH_Deviation"] = (df["Soil_pH"] - 6.5).abs()
    df["Heat_Stress"] = df["Temperature_C"] * df["Sunlight_Hours"] / (df["Humidity"] + 1)
    df["Dryness_Index"] = (100 - df["Humidity"]) * df["Temperature_C"] / 100
    df["Wind_Evap"] = df["Wind_Speed_kmh"] * (100 - df["Humidity"]) / 100
    df["Moisture_Deficit"] = 50 - df["Soil_Moisture"]
    df["Irrigation_Per_Hectare"] = df["Previous_Irrigation_mm"] / (df["Field_Area_hectare"] + 0.1)
    df["Rainfall_Per_Hectare"] = df["Rainfall_mm"] / (df["Field_Area_hectare"] + 0.1)
    df["Moisture_Retention"] = df["Soil_Moisture"] * df["Organic_Carbon"]
    df["Moisture_Temp_Ratio"] = df["Soil_Moisture"] / (df["Temperature_C"] + 1)

    # Interactions
    df["SM_x_Temp"] = df["Soil_Moisture"] * df["Temperature_C"]
    df["SM_x_Humidity"] = df["Soil_Moisture"] * df["Humidity"]
    df["Temp_x_Humidity"] = df["Temperature_C"] * df["Humidity"]
    df["Rainfall_x_Temp"] = df["Rainfall_mm"] * df["Temperature_C"]

    # Binary indicators
    df["is_active_growth"] = df["Crop_Growth_Stage"].isin(["Vegetative", "Flowering"]).astype(int)
    df["is_dry_hot"] = ((df["Soil_Moisture"] < 25) & (df["Temperature_C"] > 30)).astype(int)
    df["is_low_rain"] = (df["Rainfall_mm"] < 500).astype(int)
    df["is_mulched"] = (df["Mulching_Used"] == "Yes").astype(int)

    return df


train = add_features(train)
test = add_features(test)

# ============ Pairwise Interaction Features ============

# All 2-way categorical combinations → factorize
for c1, c2 in combinations(cat_cols, 2):
    col_name = f"{c1}_x_{c2}"
    train[col_name] = train[c1].astype(str) + "_" + train[c2].astype(str)
    test[col_name] = test[c1].astype(str) + "_" + test[c2].astype(str)
    # Factorize
    all_vals = pd.concat([train[col_name], test[col_name]]).astype("category").cat.codes
    train[col_name] = all_vals[:len(train)].values
    test[col_name] = all_vals[len(train):].values

interaction_cols = [f"{c1}_x_{c2}" for c1, c2 in combinations(cat_cols, 2)]
print(f"Added {len(interaction_cols)} interaction features")

# ============ Target Encoding (from original data) ============

# Use original data to create target encoding for each cat feature
orig_te = pd.read_csv(DATA_DIR / "original" / "irrigation_prediction.csv")
orig_te["target_num"] = orig_te["Irrigation_Need"].map(target_map)

for col in cat_cols:
    te_map = orig_te.groupby(col)["target_num"].mean()
    train[f"{col}_te_orig"] = train[col].map(te_map)
    test[f"{col}_te_orig"] = test[col].map(te_map)

# Per-group Soil_Moisture deviation (from original)
for grp_col in ["Soil_Type", "Season", "Crop_Growth_Stage"]:
    grp_mean = orig_te.groupby(grp_col)["Soil_Moisture"].mean()
    train[f"SM_dev_{grp_col}"] = train["Soil_Moisture"] - train[grp_col].map(grp_mean)
    test[f"SM_dev_{grp_col}"] = test["Soil_Moisture"] - test[grp_col].map(grp_mean)

# ============ Prepare Features ============

drop_cols = ["id", "Irrigation_Need"]
features = [c for c in train.columns if c not in drop_cols]

# Label encode categorical for LGBM/XGB
train_enc = train.copy()
test_enc = test.copy()
label_encoders = {}
all_cat = cat_cols + interaction_cols
for col in all_cat:
    if col in features:
        le = LabelEncoder()
        combined = pd.concat([train_enc[col].astype(str), test_enc[col].astype(str)])
        le.fit(combined)
        train_enc[col] = le.transform(train_enc[col].astype(str))
        test_enc[col] = le.transform(test_enc[col].astype(str))
        label_encoders[col] = le

X = train_enc[features]
X_test = test_enc[features]

# For CatBoost
X_cat = train[features].copy()
X_test_cat = test[features].copy()
# CatBoost needs string types for cat features
for col in cat_cols:
    X_cat[col] = X_cat[col].astype(str)
    X_test_cat[col] = X_test_cat[col].astype(str)
# Interaction cols are already numeric (factorized)
cat_indices = [features.index(c) for c in cat_cols]

print(f"Total features: {len(features)}")

# ============ Sample Weights ============

sample_weights = compute_sample_weight("balanced", y)

# ============ Models ============

lgbm_params = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "verbosity": -1,
    "n_estimators": 3000,
    "learning_rate": 0.02,
    "num_leaves": 127,
    "min_child_samples": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "is_unbalance": True,
}

xgb_params = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "verbosity": 0,
    "n_estimators": 3000,
    "learning_rate": 0.02,
    "max_depth": 8,
    "min_child_weight": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "tree_method": "hist",
}

cat_params = {
    "iterations": 3000,
    "learning_rate": 0.02,
    "depth": 8,
    "l2_leaf_reg": 3,
    "random_seed": 42,
    "verbose": 200,
    "cat_features": cat_indices,
    "auto_class_weights": "Balanced",
}

# ============ CV ============

N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_lgbm = np.zeros((len(X), 3))
oof_xgb = np.zeros((len(X), 3))
oof_cat = np.zeros((len(X), 3))
test_lgbm = np.zeros((len(X_test), 3))
test_xgb = np.zeros((len(X_test), 3))
test_cat = np.zeros((len(X_test_cat), 3))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n{'='*40} Fold {fold} {'='*40}")
    y_tr, y_val = y[train_idx], y[val_idx]
    sw_tr = sample_weights[train_idx]

    # LightGBM
    model_lgb = lgb.LGBMClassifier(**lgbm_params)
    model_lgb.fit(
        X.iloc[train_idx], y_tr,
        sample_weight=sw_tr,
        eval_set=[(X.iloc[val_idx], y_val)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)],
    )
    oof_lgbm[val_idx] = model_lgb.predict_proba(X.iloc[val_idx])
    test_lgbm += model_lgb.predict_proba(X_test) / N_FOLDS
    print(f"  LGBM bal_acc: {balanced_accuracy_score(y_val, oof_lgbm[val_idx].argmax(1)):.6f}")
    del model_lgb; gc.collect()

    # XGBoost
    model_xgb = xgb.XGBClassifier(**xgb_params)
    model_xgb.fit(
        X.iloc[train_idx], y_tr,
        sample_weight=sw_tr,
        eval_set=[(X.iloc[val_idx], y_val)],
        verbose=500,
    )
    oof_xgb[val_idx] = model_xgb.predict_proba(X.iloc[val_idx])
    test_xgb += model_xgb.predict_proba(X_test) / N_FOLDS
    print(f"  XGB  bal_acc: {balanced_accuracy_score(y_val, oof_xgb[val_idx].argmax(1)):.6f}")
    del model_xgb; gc.collect()

    # CatBoost
    model_cb = CatBoostClassifier(**cat_params)
    model_cb.fit(
        X_cat.iloc[train_idx], y_tr,
        eval_set=(X_cat.iloc[val_idx], y_val),
        early_stopping_rounds=100,
    )
    oof_cat[val_idx] = model_cb.predict_proba(X_cat.iloc[val_idx])
    test_cat += model_cb.predict_proba(X_test_cat) / N_FOLDS
    print(f"  CAT  bal_acc: {balanced_accuracy_score(y_val, oof_cat[val_idx].argmax(1)):.6f}")
    del model_cb; gc.collect()

# ============ Ensemble + Threshold Optimization ============

# Individual scores
for name, oof in [("LGBM", oof_lgbm), ("XGB", oof_xgb), ("CAT", oof_cat)]:
    bal_acc = balanced_accuracy_score(y, oof.argmax(1))
    print(f"\n{name} OOF balanced_accuracy: {bal_acc:.6f}")

# Grid search ensemble weights
best_bal_acc = 0
best_w = (1, 1, 1)
for w1 in range(1, 8):
    for w2 in range(1, 8):
        for w3 in range(0, 5):  # Allow 0 for CatBoost exclusion
            if w1 + w2 + w3 == 0:
                continue
            total = w1 + w2 + w3
            oof_ens = (w1 * oof_lgbm + w2 * oof_xgb + w3 * oof_cat) / total
            bal_acc = balanced_accuracy_score(y, oof_ens.argmax(1))
            if bal_acc > best_bal_acc:
                best_bal_acc = bal_acc
                best_w = (w1, w2, w3)

print(f"\nBest ensemble weights (lgbm:xgb:cat): {best_w}")
print(f"Best ensemble OOF balanced_accuracy: {best_bal_acc:.6f}")

total = sum(best_w)
oof_ens = (best_w[0] * oof_lgbm + best_w[1] * oof_xgb + best_w[2] * oof_cat) / total
test_ens = (best_w[0] * test_lgbm + best_w[1] * test_xgb + best_w[2] * test_cat) / total

# ============ Threshold Optimization ============
# Search for class probability weights to boost minority class recall

best_threshold_acc = best_bal_acc
best_class_w = (1.0, 1.0, 1.0)

for w_low in np.arange(0.7, 1.3, 0.05):
    for w_med in np.arange(0.7, 1.3, 0.05):
        for w_high in np.arange(1.0, 3.0, 0.1):
            adjusted = oof_ens.copy()
            adjusted[:, 0] *= w_low
            adjusted[:, 1] *= w_med
            adjusted[:, 2] *= w_high
            preds = adjusted.argmax(1)
            bal_acc = balanced_accuracy_score(y, preds)
            if bal_acc > best_threshold_acc:
                best_threshold_acc = bal_acc
                best_class_w = (w_low, w_med, w_high)

print(f"\nThreshold optimization class weights: {best_class_w}")
print(f"After threshold: balanced_accuracy {best_threshold_acc:.6f} (was {best_bal_acc:.6f})")

# Apply threshold to test
test_adjusted = test_ens.copy()
test_adjusted[:, 0] *= best_class_w[0]
test_adjusted[:, 1] *= best_class_w[1]
test_adjusted[:, 2] *= best_class_w[2]

# Also apply to OOF for consistent scoring
oof_adjusted = oof_ens.copy()
oof_adjusted[:, 0] *= best_class_w[0]
oof_adjusted[:, 1] *= best_class_w[1]
oof_adjusted[:, 2] *= best_class_w[2]

# ============ Save ============

np.save(OUT_DIR / "oof_preds.npy", np.stack([oof_lgbm, oof_xgb, oof_cat]))
np.save(OUT_DIR / "test_preds.npy", np.stack([test_lgbm, test_xgb, test_cat]))

sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
sub["Irrigation_Need"] = [target_inv[p] for p in test_adjusted.argmax(1)]
sub.to_csv(OUT_DIR / "submission.csv", index=False)

# Also save without threshold for comparison
sub_no_thresh = pd.read_csv(DATA_DIR / "sample_submission.csv")
sub_no_thresh["Irrigation_Need"] = [target_inv[p] for p in test_ens.argmax(1)]
sub_no_thresh.to_csv(OUT_DIR / "submission_no_threshold.csv", index=False)

results = {
    "trial": "trial_003_balanced_blend",
    "metric": "balanced_accuracy",
    "oof_bal_acc_lgbm": round(balanced_accuracy_score(y, oof_lgbm.argmax(1)), 6),
    "oof_bal_acc_xgb": round(balanced_accuracy_score(y, oof_xgb.argmax(1)), 6),
    "oof_bal_acc_cat": round(balanced_accuracy_score(y, oof_cat.argmax(1)), 6),
    "oof_bal_acc_ensemble": round(best_bal_acc, 6),
    "oof_bal_acc_threshold": round(best_threshold_acc, 6),
    "ensemble_weights": list(best_w),
    "threshold_class_weights": [round(w, 3) for w in best_class_w],
    "n_features": len(features),
    "n_interaction_features": len(interaction_cols),
    "original_data_rows": len(orig),
}
with open(OUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone. Files saved to {OUT_DIR}")
print(f"Prediction distribution: {pd.Series([target_inv[p] for p in test_adjusted.argmax(1)]).value_counts().to_dict()}")
