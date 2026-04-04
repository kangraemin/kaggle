"""
Trial 008: sklearn Multiclass TE on ALL Pairwise + bal_acc Eval + Full Orig Merge + Non-Linear FE
- sklearn TargetEncoder(target_type=multiclass, cv=5) on ALL pairwise + CATS + binary features
- XGB balanced_accuracy custom eval callback for early stopping
- Full original data merge (weight=1.0)
- Non-linear features: squared, log1p, rank
- Binary features in pairwise pool
- Single XGB (15k iters) + bias tuning
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, TargetEncoder
from sklearn.metrics import balanced_accuracy_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from itertools import combinations
import xgboost as xgb

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
    # Non-linear features (NEW)
    df["moisture_sq"] = df["Soil_Moisture"] ** 2
    df["temp_sq"] = df["Temperature_C"] ** 2
    df["wind_sq"] = df["Wind_Speed_kmh"] ** 2
    df["rainfall_log"] = np.log1p(df["Rainfall_mm"])
    df["prev_irrig_log"] = np.log1p(df["Previous_Irrigation_mm"])
    df["field_area_log"] = np.log1p(df["Field_Area_hectare"])
    return df


print("Applying domain features...")
for df in [train, test, original]:
    add_domain_features(df)

# Rank features (NEW) - computed on train+test combined
for col in ["Soil_Moisture", "Temperature_C", "Rainfall_mm"]:
    rank_col = f"{col}_rank"
    all_vals = pd.concat([train[col], test[col]])
    ranks = all_vals.rank(pct=True)
    train[rank_col] = ranks.iloc[:len(train)].values
    test[rank_col] = ranks.iloc[len(train):].values
    # For original, rank within original
    original[rank_col] = original[col].rank(pct=True)

BINARY_FEATURES = ["soil_lt_25", "temp_gt_30", "rain_lt_300", "wind_gt_10"]

# ============ Frequency Encoding ============
freq_cols = []
for col in CATS:
    col_name = f"freq_{col}"
    freq_map = train[col].value_counts(normalize=True).to_dict()
    train[col_name] = train[col].map(freq_map).fillna(0)
    test[col_name] = test[col].map(freq_map).fillna(0)
    original[col_name] = original[col].map(freq_map).fillna(0)
    freq_cols.append(col_name)
print(f"Frequency encoding: {len(freq_cols)} cols")

# ============ Original Data TE (prior features) ============
orig_te_cols = []
for col in CATS:
    for cls in range(3):
        col_name = f"orig_te_{col}_cls{cls}"
        mean_val = pd.Series(y_orig == cls).groupby(original[col]).mean()
        global_mean = (y_orig == cls).mean()
        train[col_name] = train[col].map(mean_val).fillna(global_mean)
        test[col_name] = test[col].map(mean_val).fillna(global_mean)
        original[col_name] = original[col].map(mean_val).fillna(global_mean)
        orig_te_cols.append(col_name)
print(f"Original data TE: {len(orig_te_cols)} cols")

# ============ Pairwise Combinations (factorize-based) ============

print("Binning numerics for pairwise...")
NUMS_BINNED = []
for col in NUMS:
    bin_col = f"{col}_bin"
    all_vals = pd.concat([train[col], test[col], original[col]])
    bins_edges = pd.qcut(all_vals, q=5, duplicates="drop", retbins=True)[1]
    for df in [train, test, original]:
        df[bin_col] = pd.cut(df[col], bins=bins_edges, labels=False, include_lowest=True).fillna(0).astype(int)
    NUMS_BINNED.append(bin_col)

# Label encode CATS for fast integer pairwise
for col in CATS:
    le = LabelEncoder()
    all_vals = pd.concat([train[col], test[col], original[col]]).astype(str)
    le.fit(all_vals)
    for df in [train, test, original]:
        df[f"{col}_enc"] = le.transform(df[col].astype(str))

# Include binary features in pairwise pool (NEW)
combo_source = (
    [(col, f"{col}_enc") for col in CATS]
    + [(col, col) for col in NUMS_BINNED]
    + [(col, col) for col in BINARY_FEATURES]
)

print(f"Building pairwise from {len(combo_source)} columns...")
pairwise_cols = []
pw_train_dict = {}
pw_test_dict = {}
pw_orig_dict = {}
max_card = 500
n_train = len(train)
n_test = len(test)

for (name1, col1), (name2, col2) in combinations(combo_source, 2):
    pw_name = f"pw_{name1}_x_{name2}"
    n1 = max(train[col1].max(), test[col1].max(), original[col1].max()) + 1
    combined_all = []
    for df in [train, test, original]:
        combined_all.append(df[col1].values * int(n1) + df[col2].values)
    nunique = len(set(combined_all[0]))
    if nunique > max_card:
        continue
    all_combined = np.concatenate(combined_all)
    codes, _ = pd.factorize(all_combined)
    pw_train_dict[pw_name] = codes[:n_train]
    pw_test_dict[pw_name] = codes[n_train:n_train + n_test]
    pw_orig_dict[pw_name] = codes[n_train + n_test:]
    pairwise_cols.append(pw_name)

train = pd.concat([train, pd.DataFrame(pw_train_dict, index=train.index)], axis=1)
test = pd.concat([test, pd.DataFrame(pw_test_dict, index=test.index)], axis=1)
original = pd.concat([original, pd.DataFrame(pw_orig_dict, index=original.index)], axis=1)
print(f"Pairwise combinations: {len(pairwise_cols)} cols (card limit={max_card})")

# Clean up temp columns
for col in NUMS_BINNED:
    for df in [train, test, original]:
        df.drop(columns=[col], inplace=True, errors="ignore")
for col in CATS:
    for df in [train, test, original]:
        df.drop(columns=[f"{col}_enc"], inplace=True, errors="ignore")

# ============ Label Encode CATS ============
label_encoders = {}
for col in CATS:
    le = LabelEncoder()
    all_vals = pd.concat([train[col], test[col], original[col]]).astype(str)
    le.fit(all_vals)
    for df in [train, test, original]:
        df[col] = le.transform(df[col].astype(str))
    label_encoders[col] = le

# Final feature list (before fold TE)
drop_cols = {"id", "Irrigation_Need"}
features = [c for c in train.columns if c not in drop_cols]
orig_features = [c for c in features if c in original.columns]
assert len(orig_features) == len(features), f"Original missing features: {set(features) - set(orig_features)}"

print(f"Total features (before fold TE): {len(features)}")

X = train[features].copy()
X_test = test[features].copy()
X_orig = original[features].copy()

# ============ Columns for sklearn TargetEncoder ============
# Apply multiclass TE to ALL categorical-like columns: CATS + pairwise + binary
te_source_cols = CATS + pairwise_cols + BINARY_FEATURES

print(f"sklearn TargetEncoder will be applied to {len(te_source_cols)} columns")

# ============ Bias Tuning ============

def bias_tune(proba, y_true, n_classes=3, steps=None):
    if steps is None:
        steps = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
    log_proba = np.log(np.clip(proba, 1e-15, 1.0))
    bias = np.zeros(n_classes)
    best_score = balanced_accuracy_score(y_true, proba.argmax(1))
    best_bias = bias.copy()
    for step in steps:
        improved = True
        while improved:
            improved = False
            for cls in range(n_classes):
                for direction in [-1, 1]:
                    new_bias = best_bias.copy()
                    new_bias[cls] += direction * step
                    preds = (log_proba + new_bias).argmax(1)
                    score = balanced_accuracy_score(y_true, preds)
                    if score > best_score:
                        best_score = score
                        best_bias = new_bias
                        improved = True
    return best_bias, best_score


# ============ XGB Custom Eval: balanced_accuracy ============

def xgb_bal_acc_eval(predt, dtrain):
    """Custom XGB eval metric for balanced accuracy (for callback-based early stopping)."""
    y_true = dtrain.get_label().astype(int)
    preds = predt.reshape(len(y_true), 3).argmax(axis=1)
    score = balanced_accuracy_score(y_true, preds)
    return "bal_acc", score


# ============ XGB Parameters ============

xgb_params = {
    "objective": "multi:softprob",
    "num_class": 3,
    "verbosity": 0,
    "learning_rate": 0.01,
    "max_depth": 6,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "max_bin": 1024,
    "random_state": 42,
}

# ============ CV ============

N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_xgb = np.zeros((len(X), 3))
test_xgb = np.zeros((len(X_test), 3))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n{'='*50} Fold {fold} {'='*50}")
    y_tr, y_val = y[train_idx], y[val_idx]

    # --- Full original data append (weight=1.0) ---
    X_tr = pd.concat([X.iloc[train_idx], X_orig], ignore_index=True)
    y_tr_full = np.concatenate([y_tr, y_orig])
    X_val = X.iloc[val_idx].copy()
    X_test_fold = X_test.copy()

    # Sample weights: balanced for all
    sample_w = compute_sample_weight("balanced", y_tr_full)

    # --- sklearn TargetEncoder(multiclass) on ALL pairwise + CATS + binary ---
    print(f"  Applying sklearn TargetEncoder on {len(te_source_cols)} cols...")
    te_new_col_names = []
    tr_te_dict = {}
    val_te_dict = {}
    test_te_dict = {}

    # Process in batches to manage memory
    batch_size = 50
    for batch_start in range(0, len(te_source_cols), batch_size):
        batch_cols = te_source_cols[batch_start:batch_start + batch_size]
        te = TargetEncoder(
            categories="auto",
            target_type="multiclass",
            cv=5,
            random_state=42,
        )
        te.fit(X_tr[batch_cols].values, y_tr_full)

        tr_encoded = te.transform(X_tr[batch_cols].values)
        val_encoded = te.transform(X_val[batch_cols].values)
        test_encoded = te.transform(X_test_fold[batch_cols].values)

        n_out_cols = tr_encoded.shape[1]
        cols_per_input = n_out_cols // len(batch_cols)

        for i, col in enumerate(batch_cols):
            for cls in range(cols_per_input):
                col_name = f"ste_{col}_cls{cls}"
                col_idx = i * cols_per_input + cls
                tr_te_dict[col_name] = tr_encoded[:, col_idx]
                val_te_dict[col_name] = val_encoded[:, col_idx]
                test_te_dict[col_name] = test_encoded[:, col_idx]
                te_new_col_names.append(col_name)

    # Batch concat all TE columns at once
    X_tr = pd.concat([X_tr, pd.DataFrame(tr_te_dict, index=X_tr.index)], axis=1)
    X_val = pd.concat([X_val, pd.DataFrame(val_te_dict, index=X_val.index)], axis=1)
    X_test_fold = pd.concat([X_test_fold, pd.DataFrame(test_te_dict, index=X_test_fold.index)], axis=1)

    fold_features = list(X_tr.columns)
    print(f"  Fold features: {len(fold_features)} (TE: {len(te_new_col_names)})")

    # --- XGBoost with bal_acc eval metric ---
    dtrain = xgb.DMatrix(X_tr[fold_features], label=y_tr_full, weight=sample_w)
    dval = xgb.DMatrix(X_val[fold_features], label=y_val)
    dtest = xgb.DMatrix(X_test_fold[fold_features])

    xgb_train_params = {
        **xgb_params,
        "eval_metric": "mlogloss",  # built-in metric for logging
    }

    bst = xgb.train(
        xgb_train_params,
        dtrain,
        num_boost_round=5000,
        evals=[(dval, "val")],
        custom_metric=xgb_bal_acc_eval,
        early_stopping_rounds=300,
        maximize=True,
        verbose_eval=100,
    )

    oof_xgb[val_idx] = bst.predict(dval).reshape(-1, 3)
    test_xgb += bst.predict(dtest).reshape(-1, 3) / N_FOLDS

    fold_bal = balanced_accuracy_score(y_val, oof_xgb[val_idx].argmax(1))
    fold_acc = accuracy_score(y_val, oof_xgb[val_idx].argmax(1))
    print(f"  XGB bal_acc: {fold_bal:.6f}, acc: {fold_acc:.6f}, best_iter: {bst.best_iteration}")

# ============ Overall Score ============

xgb_bal = balanced_accuracy_score(y, oof_xgb.argmax(1))
xgb_acc = accuracy_score(y, oof_xgb.argmax(1))
print(f"\nXGB OOF: bal_acc={xgb_bal:.6f}, acc={xgb_acc:.6f}")

# ============ Bias Tuning ============

print("\nBias tuning...")
bias, biased_bal = bias_tune(oof_xgb, y)
print(f"  Bias: {bias}")
print(f"  After bias: bal_acc={biased_bal:.6f} (before: {xgb_bal:.6f})")

# Apply bias to test
if biased_bal > xgb_bal:
    log_test = np.log(np.clip(test_xgb, 1e-15, 1.0))
    final_test_preds = (log_test + bias).argmax(1)
    log_oof = np.log(np.clip(oof_xgb, 1e-15, 1.0))
    final_oof_preds = (log_oof + bias).argmax(1)
    final_bal = biased_bal
    final_method = "xgb+bias"
    final_bias = bias.tolist()
else:
    final_test_preds = test_xgb.argmax(1)
    final_oof_preds = oof_xgb.argmax(1)
    final_bal = xgb_bal
    final_method = "xgb_raw"
    final_bias = None

final_acc = accuracy_score(y, final_oof_preds)
print(f"\nFinal: method={final_method}, bal_acc={final_bal:.6f}, acc={final_acc:.6f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y, final_oof_preds, target_names=["Low", "Medium", "High"]))

# ============ Save ============

np.save(OUT_DIR / "oof_preds.npy", oof_xgb)
np.save(OUT_DIR / "test_preds.npy", test_xgb)

sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
sub["Irrigation_Need"] = [target_inv[p] for p in final_test_preds]
sub.to_csv(OUT_DIR / "submission.csv", index=False)

results = {
    "trial": "trial_008_sklearn_multiclass_te",
    "oof_balanced_accuracy": round(final_bal, 6),
    "oof_accuracy": round(final_acc, 6),
    "oof_accuracy_ensemble": round(final_bal, 6),
    "method": final_method,
    "bias": final_bias,
    "xgb_raw_bal_acc": round(xgb_bal, 6),
    "n_features_base": len(features),
    "n_pairwise": len(pairwise_cols),
    "n_te_source_cols": len(te_source_cols),
}
with open(OUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone. Files saved to {OUT_DIR}")
