"""
Trial 005: External Data + Formula Features + Balanced Accuracy
- Fix metric to balanced_accuracy_score (competition metric)
- Original data: target encoding + append (weight=0.1)
- Deotte formula features (binary thresholds)
- Balanced class weights
- Pairwise categorical combinations + sklearn TargetEncoder
- XGBoost single model (Mahog approach)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
OUT_DIR = Path(__file__).resolve().parent

# ============ Load Data ============

train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
original = pd.read_csv(DATA_DIR / "original" / "irrigation_prediction.csv")

target_map = {"Low": 0, "Medium": 1, "High": 2}
target_inv = {v: k for k, v in target_map.items()}

# Map targets
y_train = train["Irrigation_Need"].map(target_map).values
y_orig = original["Irrigation_Need"].map(target_map).values

CATS = ["Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
        "Irrigation_Type", "Water_Source", "Mulching_Used", "Region"]
NUMS = ["Soil_Moisture", "Temperature_C", "Humidity", "Rainfall_mm",
        "Wind_Speed_kmh", "Sunlight_Hours", "Field_Area_hectare",
        "Previous_Irrigation_mm", "Water_Usage_m3", "Fertilizer_Used_kg",
        "Pest_Infestation_Level", "Crop_Yield_ton"]


# ============ Feature Engineering ============

def add_domain_features(df):
    """Domain-driven features from trial_002."""
    df["ET_proxy"] = df["Temperature_C"] * df["Wind_Speed_kmh"] / (df["Humidity"] + 1)
    df["water_balance"] = df["Rainfall_mm"] - df["ET_proxy"] * 100
    df["SM_x_Temp"] = df["Soil_Moisture"] * df["Temperature_C"]
    df["SM_x_Humidity"] = df["Soil_Moisture"] * df["Humidity"]
    df["Temp_x_Humidity"] = df["Temperature_C"] * df["Humidity"]
    df["Rainfall_x_Temp"] = df["Rainfall_mm"] * df["Temperature_C"]
    df["is_active_growth"] = (df["Crop_Growth_Stage"].isin(["Vegetative", "Flowering"])).astype(int)
    df["rain_per_area"] = df["Rainfall_mm"] / (df["Field_Area_hectare"] + 0.1)
    df["prev_irr_per_area"] = df["Previous_Irrigation_mm"] / (df["Field_Area_hectare"] + 0.1)
    df["moisture_deficit"] = 50 - df["Soil_Moisture"]
    return df


def add_formula_features(df):
    """Deotte exact formula binary thresholds."""
    df["soil_lt_25"] = (df["Soil_Moisture"] < 25).astype(int)
    df["temp_gt_30"] = (df["Temperature_C"] > 30).astype(int)
    df["rain_lt_300"] = (df["Rainfall_mm"] < 300).astype(int)
    df["wind_gt_10"] = (df["Wind_Speed_kmh"] > 10).astype(int)
    df["is_dry_hot"] = ((df["Soil_Moisture"] < 25) & (df["Temperature_C"] > 30)).astype(int)
    df["is_mulched"] = (df["Mulching_Used"] == "Yes").astype(int)
    return df


def add_pairwise_cats(df, cat_cols):
    """Create pairwise categorical combinations."""
    new_cols = []
    for i, c1 in enumerate(cat_cols):
        for c2 in cat_cols[i+1:]:
            col_name = f"{c1}_x_{c2}"
            df[col_name] = df[c1].astype(str) + "_" + df[c2].astype(str)
            new_cols.append(col_name)
    return df, new_cols


# ============ Target Encoding from Original Data ============

def compute_original_te(original_df, y_orig, cat_cols, n_classes=3):
    """Compute target encoding statistics from original data."""
    te_maps = {}
    for col in cat_cols:
        te_map = {}
        for cls in range(n_classes):
            mean_val = pd.Series(y_orig == cls).groupby(original_df[col]).mean()
            te_map[cls] = mean_val.to_dict()
        te_maps[col] = te_map
    return te_maps


def apply_original_te(df, te_maps, prefix="orig_te"):
    """Apply pre-computed target encoding from original data."""
    new_cols = []
    for col, cls_maps in te_maps.items():
        for cls, mapping in cls_maps.items():
            col_name = f"{prefix}_{col}_cls{cls}"
            global_mean = np.mean(list(mapping.values()))
            df[col_name] = df[col].map(mapping).fillna(global_mean)
            new_cols.append(col_name)
    return df, new_cols


# ============ In-fold Target Encoding ============

def fold_target_encode(train_df, val_df, test_df, cat_cols, y_train_fold, n_classes=3, smoothing=10):
    """Target encoding within fold to prevent leakage."""
    te_cols = []
    global_means = {}
    for cls in range(n_classes):
        global_means[cls] = (y_train_fold == cls).mean()

    for col in cat_cols:
        for cls in range(n_classes):
            col_name = f"te_{col}_cls{cls}"
            te_cols.append(col_name)

            # Compute stats on train fold
            target_binary = (y_train_fold == cls).astype(float)
            stats = pd.DataFrame({"col": train_df[col].values, "target": target_binary})
            agg = stats.groupby("col")["target"].agg(["mean", "count"])
            # Smoothed encoding
            smooth = (agg["count"] * agg["mean"] + smoothing * global_means[cls]) / (agg["count"] + smoothing)
            smooth_dict = smooth.to_dict()

            train_df[col_name] = train_df[col].map(smooth_dict).fillna(global_means[cls])
            val_df[col_name] = val_df[col].map(smooth_dict).fillna(global_means[cls])
            test_df[col_name] = test_df[col].map(smooth_dict).fillna(global_means[cls])

    return train_df, val_df, test_df, te_cols


# ============ Prepare Data ============

# Apply FE to all datasets
for df in [train, test, original]:
    add_domain_features(df)
    add_formula_features(df)

# Pairwise categoricals
train, pair_cols = add_pairwise_cats(train, CATS)
test, _ = add_pairwise_cats(test, CATS)
original, _ = add_pairwise_cats(original, CATS)

# Original data target encoding
te_maps = compute_original_te(original, y_orig, CATS)
train, orig_te_cols = apply_original_te(train, te_maps)
test, _ = apply_original_te(test, te_maps)

# Label encode all categoricals (original + pairwise)
all_cats = CATS + pair_cols
label_encoders = {}
for col in all_cats:
    le = LabelEncoder()
    all_vals = pd.concat([train[col], test[col], original[col]], axis=0).astype(str)
    le.fit(all_vals)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
    original[col] = le.transform(original[col].astype(str))
    label_encoders[col] = le

# Append original data to train with weight
print(f"Train size before append: {len(train)}")
# Keep only columns that exist in both
common_cols = [c for c in train.columns if c in original.columns and c not in ["id", "Irrigation_Need"]]
orig_subset = original[common_cols].copy()
train_subset_cols = common_cols  # Will use these as features

# Create weight array: 1.0 for competition data, 0.1 for original
n_train = len(train)
n_orig = len(original)

# Combine
train_combined = pd.concat([train[common_cols], orig_subset], axis=0, ignore_index=True)
y_combined = np.concatenate([y_train, y_orig])
sample_weights_base = np.concatenate([np.ones(n_train), np.full(n_orig, 0.1)])
print(f"Train size after append: {len(train_combined)}")

# Identify feature columns
drop_cols = {"id", "Irrigation_Need"}
features = [c for c in common_cols if c not in drop_cols]
print(f"Number of features: {len(features)}")

# ============ XGBoost with Balanced Accuracy ============

xgb_params = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "verbosity": 0,
    "n_estimators": 5000,
    "learning_rate": 0.02,
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

N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# OOF only for competition data (first n_train rows)
oof_preds = np.zeros((n_train, 3))
test_preds = np.zeros((len(test), 3))

X_all = train_combined[features]
X_test = test[features]

fold_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_all.iloc[:n_train], y_train)):
    print(f"\n{'='*40} Fold {fold} {'='*40}")

    # Training: competition train fold + all original data
    # Original data indices are n_train to n_train+n_orig
    orig_indices = list(range(n_train, n_train + n_orig))
    combined_train_idx = list(train_idx) + orig_indices

    X_tr = X_all.iloc[combined_train_idx].copy()
    y_tr = y_combined[combined_train_idx]
    X_val = X_all.iloc[val_idx].copy()
    y_val = y_train[val_idx]

    # In-fold target encoding for pairwise cols
    X_test_fold = X_test.copy()
    X_tr, X_val, X_test_fold, te_cols = fold_target_encode(
        X_tr, X_val, X_test_fold,
        pair_cols, y_tr, n_classes=3, smoothing=10
    )
    fold_features = features + te_cols

    # Sample weights: base weight * class balance weight
    w_tr = sample_weights_base[combined_train_idx]
    # Apply balanced class weights
    class_w = compute_sample_weight("balanced", y_tr)
    w_tr = w_tr * class_w

    model = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=200)
    model.fit(
        X_tr[fold_features], y_tr,
        eval_set=[(X_val[fold_features], y_val)],
        sample_weight=w_tr,
        verbose=500,
    )

    oof_preds[val_idx] = model.predict_proba(X_val[fold_features])
    test_preds += model.predict_proba(X_test_fold[fold_features]) / N_FOLDS

    bal_acc = balanced_accuracy_score(y_val, oof_preds[val_idx].argmax(1))
    acc = accuracy_score(y_val, oof_preds[val_idx].argmax(1))
    fold_scores.append(bal_acc)
    print(f"  Fold {fold} balanced_acc: {bal_acc:.6f}, acc: {acc:.6f}")

# ============ Results ============

oof_bal_acc = balanced_accuracy_score(y_train, oof_preds.argmax(1))
oof_acc = accuracy_score(y_train, oof_preds.argmax(1))

print(f"\nOOF balanced accuracy: {oof_bal_acc:.6f}")
print(f"OOF accuracy: {oof_acc:.6f}")
print(f"Per-fold balanced acc: {[f'{s:.6f}' for s in fold_scores]}")

# Per-class recall
from sklearn.metrics import classification_report
print("\nClassification Report:")
print(classification_report(y_train, oof_preds.argmax(1), target_names=["Low", "Medium", "High"]))

# Save
np.save(OUT_DIR / "oof_preds.npy", oof_preds)
np.save(OUT_DIR / "test_preds.npy", test_preds)

sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
sub["Irrigation_Need"] = [target_inv[p] for p in test_preds.argmax(1)]
sub.to_csv(OUT_DIR / "submission.csv", index=False)

results = {
    "trial": "trial_005_ext_data_balanced_acc",
    "oof_balanced_accuracy": round(oof_bal_acc, 6),
    "oof_accuracy": round(oof_acc, 6),
    "oof_accuracy_ensemble": round(oof_bal_acc, 6),  # main metric
    "per_fold_balanced_acc": [round(s, 6) for s in fold_scores],
    "n_features": len(features) + len(te_cols),
    "n_train_original": n_orig,
}
with open(OUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone. Files saved to {OUT_DIR}")
