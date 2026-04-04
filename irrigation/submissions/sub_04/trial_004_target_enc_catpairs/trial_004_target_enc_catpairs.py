"""
Trial 004: Target Encoding + Pairwise Categorical Combinations
- 8 categorical -> 28 pairwise combinations (label encoded)
- Target encoding (within CV fold, smoothing m=100) for all categoricals
- Frequency encoding for all categoricals
- Existing domain FE from trial_002
- LGBM + XGB + CatBoost ensemble
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
OUT_DIR = Path(__file__).resolve().parent

# Load
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

target_map = {"Low": 0, "Medium": 1, "High": 2}
target_inv = {v: k for k, v in target_map.items()}
y = train["Irrigation_Need"].map(target_map).values

cat_cols = ["Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
            "Irrigation_Type", "Water_Source", "Mulching_Used", "Region"]


# ============ Feature Engineering ============

def add_domain_features(df):
    """Domain FE from trial_002."""
    df["ET_proxy"] = df["Temperature_C"] * df["Wind_Speed_kmh"] / (df["Humidity"] + 1)
    df["water_balance"] = df["Rainfall_mm"] - df["ET_proxy"] * 100
    df["SM_x_Temp"] = df["Soil_Moisture"] * df["Temperature_C"]
    df["SM_x_Humidity"] = df["Soil_Moisture"] * df["Humidity"]
    df["Temp_x_Humidity"] = df["Temperature_C"] * df["Humidity"]
    df["Rainfall_x_Temp"] = df["Rainfall_mm"] * df["Temperature_C"]
    df["is_active_growth"] = (df["Crop_Growth_Stage"].isin(["Vegetative", "Flowering"])).astype(int)
    df["is_dry_hot"] = ((df["Soil_Moisture"] < 25) & (df["Temperature_C"] > 30)).astype(int)
    df["is_low_rain"] = (df["Rainfall_mm"] < 500).astype(int)
    df["is_mulched"] = (df["Mulching_Used"] == "Yes").astype(int)
    df["rain_per_area"] = df["Rainfall_mm"] / (df["Field_Area_hectare"] + 0.1)
    df["prev_irr_per_area"] = df["Previous_Irrigation_mm"] / (df["Field_Area_hectare"] + 0.1)
    df["moisture_deficit"] = 50 - df["Soil_Moisture"]
    return df


def add_pairwise_combinations(df, cat_cols):
    """Create pairwise categorical combinations."""
    pair_cols = []
    for c1, c2 in combinations(cat_cols, 2):
        col_name = f"{c1}_x_{c2}"
        df[col_name] = df[c1].astype(str) + "_" + df[c2].astype(str)
        pair_cols.append(col_name)
    return df, pair_cols


def add_frequency_encoding(train_df, test_df, cols):
    """Frequency encoding based on train set."""
    freq_cols = []
    for col in cols:
        freq = train_df[col].value_counts(normalize=True)
        col_name = f"{col}_freq"
        train_df[col_name] = train_df[col].map(freq).fillna(0)
        test_df[col_name] = test_df[col].map(freq).fillna(0)
        freq_cols.append(col_name)
    return train_df, test_df, freq_cols


# Apply domain FE
train = add_domain_features(train)
test = add_domain_features(test)

# Add pairwise combinations
train, pair_cols = add_pairwise_combinations(train, cat_cols)
test, _ = add_pairwise_combinations(test, cat_cols)

all_cat_cols = cat_cols + pair_cols  # 8 + 28 = 36

# Frequency encoding (on all categoricals including pairs)
train, test, freq_cols = add_frequency_encoding(train, test, all_cat_cols)

# Label encode for LGBM/XGB
train_enc = train.copy()
test_enc = test.copy()
label_encoders = {}
for col in all_cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train_enc[col], test_enc[col]], axis=0).astype(str)
    le.fit(combined)
    train_enc[col] = le.transform(train_enc[col].astype(str))
    test_enc[col] = le.transform(test_enc[col].astype(str))
    label_encoders[col] = le

# Feature lists
drop_cols = ["id", "Irrigation_Need"]
features_base = [c for c in train_enc.columns if c not in drop_cols]

# ============ Models ============

lgbm_params = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "verbosity": -1,
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "num_leaves": 127,
    "min_child_samples": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 2.0,  # slightly higher for more features
    "random_state": 42,
}

xgb_params = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "verbosity": 0,
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "max_depth": 8,
    "min_child_weight": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 2.0,
    "random_state": 42,
    "tree_method": "hist",
}

cat_params = {
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 8,
    "l2_leaf_reg": 3,
    "random_seed": 42,
    "verbose": 200,
    "auto_class_weights": "Balanced",
}

# ============ Target Encoding Function ============

def target_encode_fold(train_df, val_df, test_df, cat_columns, y_train, num_classes=3, m=100):
    """
    Target encoding within fold. Returns new columns for val and test.
    For each class c in [1, 2] (skip class 0 = linearly dependent):
        TE_col_c = smoothed P(y==c | category_value)
    """
    te_cols = []
    val_te = pd.DataFrame(index=val_df.index)
    test_te = pd.DataFrame(index=test_df.index)
    train_te = pd.DataFrame(index=train_df.index)  # for internal use

    for col in cat_columns:
        for c in range(1, num_classes):  # classes 1 and 2 (Medium, High)
            col_name = f"TE_{col}_c{c}"
            te_cols.append(col_name)

            # Global mean
            global_mean = (y_train == c).mean()

            # Per-category stats from training fold
            tmp = pd.DataFrame({"cat": train_df[col].astype(str), "target": (y_train == c).astype(int)})
            stats = tmp.groupby("cat")["target"].agg(["mean", "count"])
            stats["smoothed"] = (stats["count"] * stats["mean"] + m * global_mean) / (stats["count"] + m)

            mapping = stats["smoothed"].to_dict()

            val_te[col_name] = val_df[col].astype(str).map(mapping).fillna(global_mean)
            test_te[col_name] = test_df[col].astype(str).map(mapping).fillna(global_mean)
            train_te[col_name] = train_df[col].astype(str).map(mapping).fillna(global_mean)

    return train_te, val_te, test_te, te_cols


# ============ CV ============

N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

n_train = len(train)
n_test = len(test)

oof_lgbm = np.zeros((n_train, 3))
oof_xgb = np.zeros((n_train, 3))
oof_cat = np.zeros((n_train, 3))
test_lgbm = np.zeros((n_test, 3))
test_xgb = np.zeros((n_test, 3))
test_cat = np.zeros((n_test, 3))

X_lgbm_base = train_enc[features_base]
X_test_lgbm_base = test_enc[features_base]

for fold, (train_idx, val_idx) in enumerate(skf.split(X_lgbm_base, y)):
    print(f"\n{'='*40} Fold {fold} {'='*40}")
    y_tr, y_val = y[train_idx], y[val_idx]

    # Target encoding (within fold)
    te_train, te_val, te_test, te_cols = target_encode_fold(
        train_enc.iloc[train_idx], train_enc.iloc[val_idx], test_enc,
        all_cat_cols, y_tr, num_classes=3, m=100
    )

    # LGBM/XGB features: base + target encoding
    X_tr = pd.concat([X_lgbm_base.iloc[train_idx].reset_index(drop=True), te_train.reset_index(drop=True)], axis=1)
    X_val = pd.concat([X_lgbm_base.iloc[val_idx].reset_index(drop=True), te_val.reset_index(drop=True)], axis=1)
    X_te = pd.concat([X_test_lgbm_base.reset_index(drop=True), te_test.reset_index(drop=True)], axis=1)

    # LightGBM
    model_lgb = lgb.LGBMClassifier(**lgbm_params)
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
    )
    oof_lgbm[val_idx] = model_lgb.predict_proba(X_val)
    test_lgbm += model_lgb.predict_proba(X_te) / N_FOLDS
    print(f"  LGBM acc: {accuracy_score(y_val, oof_lgbm[val_idx].argmax(1)):.6f}")

    # XGBoost (with early stopping)
    model_xgb = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=100)
    model_xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=200,
    )
    oof_xgb[val_idx] = model_xgb.predict_proba(X_val)
    test_xgb += model_xgb.predict_proba(X_te) / N_FOLDS
    print(f"  XGB  acc: {accuracy_score(y_val, oof_xgb[val_idx].argmax(1)):.6f}")

    # CatBoost: use label-encoded features + TE, no native cat (faster)
    model_cb = CatBoostClassifier(**cat_params)
    model_cb.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100,
    )
    oof_cat[val_idx] = model_cb.predict_proba(X_val)
    test_cat += model_cb.predict_proba(X_te) / N_FOLDS
    print(f"  CAT  acc: {accuracy_score(y_val, oof_cat[val_idx].argmax(1)):.6f}")

# ============ Ensemble ============

for name, oof in [("LGBM", oof_lgbm), ("XGB", oof_xgb), ("CAT", oof_cat)]:
    acc = accuracy_score(y, oof.argmax(1))
    print(f"\n{name} OOF accuracy: {acc:.6f}")

# Grid search ensemble weights
best_acc = 0
best_w = (1, 1, 1)
for w1 in range(1, 6):
    for w2 in range(1, 6):
        for w3 in range(1, 6):
            total = w1 + w2 + w3
            oof_ens = (w1 * oof_lgbm + w2 * oof_xgb + w3 * oof_cat) / total
            acc = accuracy_score(y, oof_ens.argmax(1))
            if acc > best_acc:
                best_acc = acc
                best_w = (w1, w2, w3)

print(f"\nBest ensemble weights (lgbm:xgb:cat): {best_w}")
print(f"Best ensemble OOF accuracy: {best_acc:.6f}")

total = sum(best_w)
test_ens = (best_w[0] * test_lgbm + best_w[1] * test_xgb + best_w[2] * test_cat) / total

# Save
np.save(OUT_DIR / "oof_preds.npy", np.stack([oof_lgbm, oof_xgb, oof_cat]))
np.save(OUT_DIR / "test_preds.npy", np.stack([test_lgbm, test_xgb, test_cat]))

sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
sub["Irrigation_Need"] = [target_inv[p] for p in test_ens.argmax(1)]
sub.to_csv(OUT_DIR / "submission.csv", index=False)

results = {
    "trial": "trial_004_target_enc_catpairs",
    "oof_accuracy_lgbm": round(accuracy_score(y, oof_lgbm.argmax(1)), 6),
    "oof_accuracy_xgb": round(accuracy_score(y, oof_xgb.argmax(1)), 6),
    "oof_accuracy_cat": round(accuracy_score(y, oof_cat.argmax(1)), 6),
    "oof_accuracy_ensemble": round(best_acc, 6),
    "ensemble_weights": best_w,
    "n_features_lgbm_xgb": len(X_tr.columns),
    "n_features_catboost": len(X_tr.columns),
    "te_columns": len(te_cols),
    "pair_columns": len(pair_cols),
}
with open(OUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nDone. Files saved to", OUT_DIR)
