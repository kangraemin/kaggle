"""
Trial 006: Full Pairwise + Multiclass TE + Multi-Model Ensemble
- Pairwise combinations (NUMS_binned + CATS), cardinality-filtered
- Manual multiclass target encoding within fold (fast)
- balanced_accuracy custom eval metric for XGB early stopping
- LGBM + XGB + CatBoost ensemble, bal_acc weight search
- Richer domain FE + Deotte binary thresholds
- Original data TE only (Mahog approach, no append)
- Frequency encoding on CATS
- Balanced class weights
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from itertools import combinations

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
OUT_DIR = Path(__file__).resolve().parent

# ============ Load Data ============

train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
original = pd.read_csv(DATA_DIR / "original" / "irrigation_prediction.csv")

target_map = {"Low": 0, "Medium": 1, "High": 2}
target_inv = {v: k for k, v in target_map.items()}

y = train["Irrigation_Need"].map(target_map).values
y_orig = original["Irrigation_Need"].map(target_map).values

CATS = ["Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
        "Irrigation_Type", "Water_Source", "Mulching_Used", "Region"]
NUMS = ["Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
        "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
        "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm"]

print(f"Train: {len(train)}, Test: {len(test)}, Original: {len(original)}")

# ============ Feature Engineering ============

def add_domain_features(df):
    df["ET_proxy"] = df["Temperature_C"] * df["Wind_Speed_kmh"] / (df["Humidity"] + 1)
    df["water_balance"] = df["Rainfall_mm"] - df["ET_proxy"] * 100
    df["SM_x_Temp"] = df["Soil_Moisture"] * df["Temperature_C"]
    df["SM_x_Humidity"] = df["Soil_Moisture"] * df["Humidity"]
    df["Temp_x_Humidity"] = df["Temperature_C"] * df["Humidity"]
    df["Rainfall_x_Temp"] = df["Rainfall_mm"] * df["Temperature_C"]
    df["is_active_growth"] = df["Crop_Growth_Stage"].isin(["Vegetative", "Flowering"]).astype(int)
    df["rain_per_area"] = df["Rainfall_mm"] / (df["Field_Area_hectare"] + 0.1)
    df["prev_irr_per_area"] = df["Previous_Irrigation_mm"] / (df["Field_Area_hectare"] + 0.1)
    df["moisture_deficit"] = 50 - df["Soil_Moisture"]
    # Ektarr features
    df["evaporation"] = df["Temperature_C"] * df["Sunlight_Hours"] * (1 - df["Humidity"] / 100)
    df["water_deficit"] = df["evaporation"] - (df["Rainfall_mm"] + df["Previous_Irrigation_mm"])
    df["heat_stress"] = df["Temperature_C"] * df["Sunlight_Hours"]
    df["drying_force"] = df["Wind_Speed_kmh"] * df["Temperature_C"] / (df["Humidity"] + 1)
    # Deotte binary thresholds
    df["soil_lt_25"] = (df["Soil_Moisture"] < 25).astype(int)
    df["temp_gt_30"] = (df["Temperature_C"] > 30).astype(int)
    df["rain_lt_300"] = (df["Rainfall_mm"] < 300).astype(int)
    df["wind_gt_10"] = (df["Wind_Speed_kmh"] > 10).astype(int)
    df["is_dry_hot"] = ((df["Soil_Moisture"] < 25) & (df["Temperature_C"] > 30)).astype(int)
    df["is_mulched"] = (df["Mulching_Used"] == "Yes").astype(int)
    df["is_low_rain"] = (df["Rainfall_mm"] < 500).astype(int)
    return df


print("Applying domain features...")
for df in [train, test, original]:
    add_domain_features(df)

# ============ Frequency Encoding ============
freq_cols = []
for col in CATS:
    col_name = f"freq_{col}"
    freq_map = train[col].value_counts(normalize=True).to_dict()
    train[col_name] = train[col].map(freq_map).fillna(0)
    test[col_name] = test[col].map(freq_map).fillna(0)
    freq_cols.append(col_name)
print(f"Frequency encoding: {len(freq_cols)} cols")

# ============ Original Data TE (Mahog: TE only, no append) ============
orig_te_cols = []
for col in CATS:
    for cls in range(3):
        col_name = f"orig_te_{col}_cls{cls}"
        mean_val = pd.Series(y_orig == cls).groupby(original[col]).mean()
        global_mean = (y_orig == cls).mean()
        train[col_name] = train[col].map(mean_val).fillna(global_mean)
        test[col_name] = test[col].map(mean_val).fillna(global_mean)
        orig_te_cols.append(col_name)
print(f"Original data TE: {len(orig_te_cols)} cols")

# ============ Pairwise Combinations (factorize-based, fast) ============

# Bin numerics into 5 quantile bins for pairwise
print("Binning numerics for pairwise...")
NUMS_BINNED = []
for col in NUMS:
    bin_col = f"{col}_bin"
    # Use combined quantiles for consistent bins
    all_vals = pd.concat([train[col], test[col]])
    bins_edges = pd.qcut(all_vals, q=5, duplicates="drop", retbins=True)[1]
    train[bin_col] = pd.cut(train[col], bins=bins_edges, labels=False, include_lowest=True).fillna(0).astype(int)
    test[bin_col] = pd.cut(test[col], bins=bins_edges, labels=False, include_lowest=True).fillna(0).astype(int)
    NUMS_BINNED.append(bin_col)

# Label encode CATS for fast integer pairwise
cat_encoded = {}
for col in CATS:
    le = LabelEncoder()
    all_vals = pd.concat([train[col], test[col]]).astype(str)
    le.fit(all_vals)
    cat_encoded[col] = le
    train[f"{col}_enc"] = le.transform(train[col].astype(str))
    test[f"{col}_enc"] = le.transform(test[col].astype(str))

combo_source = [(col, f"{col}_enc") for col in CATS] + [(col, col) for col in NUMS_BINNED]

print(f"Building pairwise from {len(combo_source)} columns...")
pairwise_cols = []
pw_train_dict = {}
pw_test_dict = {}
max_card = 500  # cardinality limit

for (name1, col1), (name2, col2) in combinations(combo_source, 2):
    pw_name = f"pw_{name1}_x_{name2}"
    n1 = max(train[col1].max(), test[col1].max()) + 1
    combined_train = train[col1].values * int(n1) + train[col2].values
    combined_test = test[col1].values * int(n1) + test[col2].values
    nunique = len(set(combined_train))
    if nunique > max_card:
        continue
    all_combined = np.concatenate([combined_train, combined_test])
    codes, _ = pd.factorize(all_combined)
    pw_train_dict[pw_name] = codes[:len(train)]
    pw_test_dict[pw_name] = codes[len(train):]
    pairwise_cols.append(pw_name)

# Concat all at once to avoid fragmentation
train = pd.concat([train, pd.DataFrame(pw_train_dict, index=train.index)], axis=1)
test = pd.concat([test, pd.DataFrame(pw_test_dict, index=test.index)], axis=1)
print(f"Pairwise combinations: {len(pairwise_cols)} cols (card limit={max_card})")

# Clean up temp columns
for col in NUMS_BINNED:
    train.drop(columns=[col], inplace=True, errors="ignore")
    test.drop(columns=[col], inplace=True, errors="ignore")
for col in CATS:
    train.drop(columns=[f"{col}_enc"], inplace=True, errors="ignore")
    test.drop(columns=[f"{col}_enc"], inplace=True, errors="ignore")

# ============ Label Encode Original CATS ============
label_encoders = {}
for col in CATS:
    le = LabelEncoder()
    all_vals = pd.concat([train[col], test[col]]).astype(str)
    le.fit(all_vals)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
    label_encoders[col] = le

# Final feature list
drop_cols = {"id", "Irrigation_Need"}
features = [c for c in train.columns if c not in drop_cols]
print(f"Total features (before fold TE): {len(features)}")

X = train[features].copy()
X_test = test[features].copy()


# ============ Manual Multiclass Target Encoding (within fold) ============

def fold_multiclass_te(X_tr, X_val, X_test_fold, te_cols, y_tr, n_classes=3, smoothing=10):
    """Fast manual multiclass TE within fold. Returns new DataFrames (no fragmentation)."""
    te_train_dict = {}
    te_val_dict = {}
    te_test_dict = {}
    new_col_names = []

    for col in te_cols:
        for cls in range(n_classes):
            col_name = f"mte_{col}_cls{cls}"
            new_col_names.append(col_name)
            target_binary = (y_tr == cls).astype(float)
            global_mean = target_binary.mean()

            stats = pd.DataFrame({"cat": X_tr[col].values, "target": target_binary})
            agg = stats.groupby("cat")["target"].agg(["mean", "count"])
            smooth = (agg["count"] * agg["mean"] + smoothing * global_mean) / (agg["count"] + smoothing)
            smooth_dict = smooth.to_dict()

            te_train_dict[col_name] = X_tr[col].map(smooth_dict).fillna(global_mean).values
            te_val_dict[col_name] = X_val[col].map(smooth_dict).fillna(global_mean).values
            te_test_dict[col_name] = X_test_fold[col].map(smooth_dict).fillna(global_mean).values

    # Concat all at once
    X_tr = pd.concat([X_tr.copy(), pd.DataFrame(te_train_dict, index=X_tr.index)], axis=1)
    X_val = pd.concat([X_val.copy(), pd.DataFrame(te_val_dict, index=X_val.index)], axis=1)
    X_test_fold = pd.concat([X_test_fold.copy(), pd.DataFrame(te_test_dict, index=X_test_fold.index)], axis=1)
    return X_tr, X_val, X_test_fold, new_col_names


# ============ Custom Eval Metrics ============

def xgb_balanced_accuracy(y_true, y_pred):
    """Custom XGB sklearn eval metric: (y_true, y_pred) -> float."""
    y_true = np.asarray(y_true).astype(int)
    if y_pred.ndim == 1:
        preds = y_pred.reshape(len(y_true), 3)
    else:
        preds = y_pred
    y_hat = preds.argmax(axis=1)
    return balanced_accuracy_score(y_true, y_hat)


def lgbm_balanced_accuracy(y_true, y_pred):
    # LightGBM multiclass: y_pred shape is (n_samples * n_classes,) row-major
    n_samples = len(y_true)
    y_pred_reshaped = y_pred.reshape(n_samples, 3)
    y_hat = y_pred_reshaped.argmax(axis=1)
    score = balanced_accuracy_score(y_true.astype(int), y_hat)
    return "bal_acc", score, True


# ============ Model Parameters ============

lgbm_params = {
    "objective": "multiclass",
    "num_class": 3,
    "verbosity": -1,
    "n_estimators": 5000,
    "learning_rate": 0.01,
    "num_leaves": 63,
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
    "verbosity": 0,
    "n_estimators": 50000,
    "learning_rate": 0.01,
    "max_depth": 6,
    "min_child_weight": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "tree_method": "hist",
    "max_bin": 1024,
}

cat_params = {
    "iterations": 5000,
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 3,
    "random_seed": 42,
    "verbose": 500,
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
test_cat = np.zeros((len(X_test), 3))

# CatBoost cat feature names (label-encoded integers, but CatBoost can handle)
cat_feature_names_for_cb = CATS  # already label-encoded

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n{'='*50} Fold {fold} {'='*50}")
    y_tr, y_val = y[train_idx], y[val_idx]

    X_tr = X.iloc[train_idx].copy()
    X_val = X.iloc[val_idx].copy()
    X_test_fold = X_test.copy()

    # Multiclass TE on CATS only (pairwise cols used as-is, TE on 179 cols is too heavy)
    te_target_cols = CATS
    X_tr, X_val, X_test_fold, te_new_cols = fold_multiclass_te(
        X_tr, X_val, X_test_fold, te_target_cols, y_tr, n_classes=3, smoothing=10
    )
    fold_features = list(X_tr.columns)

    # Balanced sample weights
    sample_w = compute_sample_weight("balanced", y_tr)

    # --- LightGBM ---
    model_lgb = lgb.LGBMClassifier(**lgbm_params)
    model_lgb.fit(
        X_tr[fold_features], y_tr,
        eval_set=[(X_val[fold_features], y_val)],
        eval_metric=lgbm_balanced_accuracy,
        callbacks=[lgb.early_stopping(200, first_metric_only=True), lgb.log_evaluation(500)],
    )
    oof_lgbm[val_idx] = model_lgb.predict_proba(X_val[fold_features])
    test_lgbm += model_lgb.predict_proba(X_test_fold[fold_features]) / N_FOLDS
    lgb_bal = balanced_accuracy_score(y_val, oof_lgbm[val_idx].argmax(1))
    print(f"  LGBM bal_acc: {lgb_bal:.6f}")

    # --- XGBoost ---
    model_xgb = xgb.XGBClassifier(
        **xgb_params,
        eval_metric=xgb_balanced_accuracy,
        callbacks=[
            xgb.callback.EarlyStopping(rounds=500, maximize=True, data_name="validation_0"),
            xgb.callback.EvaluationMonitor(period=2000),
        ],
    )
    model_xgb.fit(
        X_tr[fold_features], y_tr,
        eval_set=[(X_val[fold_features], y_val)],
        sample_weight=sample_w,
        verbose=False,
    )
    oof_xgb[val_idx] = model_xgb.predict_proba(X_val[fold_features])
    test_xgb += model_xgb.predict_proba(X_test_fold[fold_features]) / N_FOLDS
    xgb_bal = balanced_accuracy_score(y_val, oof_xgb[val_idx].argmax(1))
    print(f"  XGB  bal_acc: {xgb_bal:.6f}")

    # --- CatBoost ---
    model_cb = CatBoostClassifier(**cat_params)
    model_cb.fit(
        X_tr[fold_features], y_tr,
        eval_set=(X_val[fold_features], y_val),
        early_stopping_rounds=200,
    )
    oof_cat[val_idx] = model_cb.predict_proba(X_val[fold_features])
    test_cat += model_cb.predict_proba(X_test_fold[fold_features]) / N_FOLDS
    cat_bal = balanced_accuracy_score(y_val, oof_cat[val_idx].argmax(1))
    print(f"  CAT  bal_acc: {cat_bal:.6f}")


# ============ Ensemble ============

for name, oof in [("LGBM", oof_lgbm), ("XGB", oof_xgb), ("CAT", oof_cat)]:
    bal = balanced_accuracy_score(y, oof.argmax(1))
    acc = accuracy_score(y, oof.argmax(1))
    print(f"\n{name} OOF: bal_acc={bal:.6f}, acc={acc:.6f}")

# Grid search ensemble weights on balanced_accuracy
best_bal_acc = 0
best_w = (1, 1, 1)
for w1 in range(0, 11):
    for w2 in range(0, 11):
        for w3 in range(0, 11):
            if w1 + w2 + w3 == 0:
                continue
            total = w1 + w2 + w3
            oof_ens = (w1 * oof_lgbm + w2 * oof_xgb + w3 * oof_cat) / total
            bal = balanced_accuracy_score(y, oof_ens.argmax(1))
            if bal > best_bal_acc:
                best_bal_acc = bal
                best_w = (w1, w2, w3)

print(f"\nBest ensemble weights (lgbm:xgb:cat): {best_w}")
print(f"Best ensemble OOF balanced_accuracy: {best_bal_acc:.6f}")
total = sum(best_w)
oof_final = (best_w[0]*oof_lgbm + best_w[1]*oof_xgb + best_w[2]*oof_cat) / total
ens_acc = accuracy_score(y, oof_final.argmax(1))
print(f"Best ensemble OOF accuracy: {ens_acc:.6f}")

test_ens = (best_w[0] * test_lgbm + best_w[1] * test_xgb + best_w[2] * test_cat) / total

# Classification report
print("\nClassification Report:")
print(classification_report(y, oof_final.argmax(1), target_names=["Low", "Medium", "High"]))

# ============ Save ============

np.save(OUT_DIR / "oof_preds.npy", np.stack([oof_lgbm, oof_xgb, oof_cat]))
np.save(OUT_DIR / "test_preds.npy", np.stack([test_lgbm, test_xgb, test_cat]))

sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
sub["Irrigation_Need"] = [target_inv[p] for p in test_ens.argmax(1)]
sub.to_csv(OUT_DIR / "submission.csv", index=False)

results = {
    "trial": "trial_006_full_pairwise_ensemble",
    "oof_balanced_accuracy": round(best_bal_acc, 6),
    "oof_accuracy": round(ens_acc, 6),
    "oof_accuracy_ensemble": round(best_bal_acc, 6),
    "ensemble_weights": best_w,
    "individual_bal_acc": {
        "lgbm": round(balanced_accuracy_score(y, oof_lgbm.argmax(1)), 6),
        "xgb": round(balanced_accuracy_score(y, oof_xgb.argmax(1)), 6),
        "cat": round(balanced_accuracy_score(y, oof_cat.argmax(1)), 6),
    },
    "n_features_base": len(features),
    "n_pairwise": len(pairwise_cols),
    "n_te_cols": len(te_new_cols),
}
with open(OUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone. Files saved to {OUT_DIR}")
