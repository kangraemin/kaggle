"""
Trial 012: Slow XGBoost (lr=0.01, 15k rounds + proper early stopping)
- Base: trial_011 architecture (171 pairwise factorize, sklearn TE on 171 pairwise, Deotte binary, orig TE)
- Key change 1: n_estimators 4000 → 15000 (original plan, was hard-capped in trial_011)
- Key change 2: Add early_stopping_rounds=200 (was missing in trial_011)
- Hypothesis: trial_011 capped at 4000 rounds without early stopping — model may still be converging.
  At lr=0.01, optimal point likely requires 5000-12000 rounds.
"""

import gc
import json
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, TargetEncoder
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
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

# ============ Original Data TE (external, leakage-free global stats) ============

orig["target_num"] = orig["Irrigation_Need"].map(target_map)
te_orig_cols = []
for col in all_raw_cols:
    te_map = orig.groupby(col)["target_num"].mean()
    col_name = f"{col}_te_orig"
    train[col_name] = train[col].map(te_map).fillna(te_map.mean())
    test[col_name] = test[col].map(te_map).fillna(te_map.mean())
    te_orig_cols.append(col_name)

# ============ ALL 171 Pairwise: factorize ============

pair_cols = []
pair_definitions = []  # (c1, c2) tuples for sklearn TE later
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
    pair_definitions.append((c1, c2, col_name))

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

# ============ Base features (non-TE) ============

base_features = num_cols + le_cols + binary_cols + te_orig_cols + pair_cols
print(f"Base features: {len(base_features)}")

# ============ Sample Weights ============

sample_weights = compute_sample_weight("balanced", y)

# ============ Single Seed CV ============

SEED = 42
N_FOLDS = 5

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_xgb_arr = np.zeros((len(train), 3))
test_xgb_arr = np.zeros((len(test), 3))
best_rounds_list = []

for fold, (train_idx, val_idx) in enumerate(skf.split(train, y)):
    print(f"\n{'='*50} Fold {fold} {'='*50}")
    y_tr, y_val = y[train_idx], y[val_idx]
    sw_tr = sample_weights[train_idx]

    # ----------------------------------------------------------------
    # 1) Manual Multiclass TE on cat_cols (8 cols × 3 classes = 24 TE)
    # ----------------------------------------------------------------
    manual_te_names = []
    manual_te_train = {}
    manual_te_test = {}

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
            manual_te_train[col_te] = train[f"{col}_le"].map(smoothed).fillna(cls_mean).values
            manual_te_test[col_te] = test[f"{col}_le"].map(smoothed).fillna(cls_mean).values
            manual_te_names.append(col_te)

    # ----------------------------------------------------------------
    # 2) sklearn TargetEncoder on ALL 171 pairwise (multiclass → 3 classes × 171 = 513)
    # ----------------------------------------------------------------
    X_pair_tr = train[pair_cols].iloc[train_idx].values  # (n_tr, 171)
    X_pair_val = train[pair_cols].iloc[val_idx].values
    X_pair_te = test[pair_cols].values

    sklearn_te = TargetEncoder(target_type="multiclass", smooth="auto", random_state=SEED)
    X_pair_tr_te = sklearn_te.fit_transform(X_pair_tr, y_tr)   # (n_tr, 171*3)
    X_pair_val_te = sklearn_te.transform(X_pair_val)
    X_pair_test_te = sklearn_te.transform(X_pair_te)

    n_sklearn_te_cols = X_pair_tr_te.shape[1]
    if fold == 0:
        print(f"  sklearn pairwise TE: {n_sklearn_te_cols} columns")

    # ----------------------------------------------------------------
    # 3) Build full feature matrix
    # ----------------------------------------------------------------
    X_tr_base = train[base_features].iloc[train_idx].copy()
    X_val_base = train[base_features].iloc[val_idx].copy()
    X_te_base = test[base_features].copy()

    # Add manual TE
    for col_te in manual_te_names:
        X_tr_base[col_te] = manual_te_train[col_te][train_idx]
        X_val_base[col_te] = manual_te_train[col_te][val_idx]
        X_te_base[col_te] = manual_te_test[col_te]

    # Add sklearn pairwise TE as numpy array concatenation
    X_tr_np = np.hstack([X_tr_base.values, X_pair_tr_te])
    X_val_np = np.hstack([X_val_base.values, X_pair_val_te])
    X_te_np = np.hstack([X_te_base.values, X_pair_test_te])

    if fold == 0:
        total_features = X_tr_np.shape[1]
        print(f"  Total features: {total_features} (base {len(base_features)} + manual_TE {len(manual_te_names)} + sklearn_TE {n_sklearn_te_cols})")

    # ----------------------------------------------------------------
    # 4) XGBoost: Slow (lr=0.01, 15k rounds, early_stopping=200)
    #    KEY CHANGE from trial_011: n_estimators 4000→15000, early_stopping_rounds=200 added
    # ----------------------------------------------------------------
    model_xgb = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_estimators=15000,
        learning_rate=0.01,
        max_depth=6,
        min_child_weight=30,
        subsample=0.8,
        colsample_bytree=0.4,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=SEED,
        tree_method="hist",
        max_bin=1024,
        early_stopping_rounds=200,
        verbosity=0,
    )
    model_xgb.fit(
        X_tr_np, y_tr,
        sample_weight=sw_tr,
        eval_set=[(X_val_np, y_val)],
        verbose=500,
    )

    best_rounds = model_xgb.best_iteration + 1
    best_rounds_list.append(best_rounds)
    oof_xgb_arr[val_idx] = model_xgb.predict_proba(X_val_np)
    test_xgb_arr += model_xgb.predict_proba(X_te_np) / N_FOLDS

    fold_bal = balanced_accuracy_score(y_val, oof_xgb_arr[val_idx].argmax(1))
    print(f"  XGB fold {fold}: bal_acc={fold_bal:.6f}, best_rounds={best_rounds}")
    del model_xgb, sklearn_te; gc.collect()

# ============ OOF Score ============

raw_bal_acc = balanced_accuracy_score(y, oof_xgb_arr.argmax(1))
print(f"\nOOF balanced_accuracy (raw argmax): {raw_bal_acc:.6f}")
print(f"Best rounds per fold: {best_rounds_list}, avg={np.mean(best_rounds_list):.0f}")

# ============ Threshold Optimization ============

best_threshold_acc = raw_bal_acc
best_class_w = (1.0, 1.0, 1.0)

# Pass 1: High class weight sweep
for w_high in np.arange(1.5, 5.0, 0.1):
    adjusted = oof_xgb_arr.copy()
    adjusted[:, 2] *= w_high
    preds = adjusted.argmax(1)
    bal_acc = balanced_accuracy_score(y, preds)
    if bal_acc > best_threshold_acc:
        best_threshold_acc = bal_acc
        best_class_w = (1.0, 1.0, w_high)

# Pass 2: Full 3D sweep
for w_low in np.arange(0.7, 1.3, 0.05):
    for w_med in np.arange(0.5, 1.3, 0.05):
        for w_high in np.arange(1.0, 5.0, 0.1):
            adjusted = oof_xgb_arr.copy()
            adjusted[:, 0] *= w_low
            adjusted[:, 1] *= w_med
            adjusted[:, 2] *= w_high
            preds = adjusted.argmax(1)
            bal_acc = balanced_accuracy_score(y, preds)
            if bal_acc > best_threshold_acc:
                best_threshold_acc = bal_acc
                best_class_w = (w_low, w_med, w_high)

print(f"Threshold class weights: {best_class_w}")
print(f"After threshold: balanced_accuracy {best_threshold_acc:.6f} (was {raw_bal_acc:.6f})")

test_adjusted = test_xgb_arr.copy()
test_adjusted[:, 0] *= best_class_w[0]
test_adjusted[:, 1] *= best_class_w[1]
test_adjusted[:, 2] *= best_class_w[2]

# ============ Save ============

np.save(OUT_DIR / "oof_preds.npy", oof_xgb_arr)
np.save(OUT_DIR / "test_preds.npy", test_xgb_arr)

sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
sub["Irrigation_Need"] = [target_inv[p] for p in test_adjusted.argmax(1)]
sub.to_csv(OUT_DIR / "submission.csv", index=False)

sub_no = pd.read_csv(DATA_DIR / "sample_submission.csv")
sub_no["Irrigation_Need"] = [target_inv[p] for p in test_xgb_arr.argmax(1)]
sub_no.to_csv(OUT_DIR / "submission_no_threshold.csv", index=False)

results = {
    "trial": "trial_012_extend_rounds",
    "metric": "balanced_accuracy",
    "seed": SEED,
    "n_folds": N_FOLDS,
    "n_total_models": N_FOLDS,
    "xgb_lr": 0.01,
    "xgb_n_estimators_max": 15000,
    "xgb_early_stopping": 200,
    "best_rounds_per_fold": best_rounds_list,
    "best_rounds_avg": round(float(np.mean(best_rounds_list)), 1),
    "oof_accuracy": round(raw_bal_acc, 6),
    "oof_accuracy_ensemble": round(best_threshold_acc, 6),
    "threshold_class_weights": [round(w, 4) for w in best_class_w],
    "n_base_features": len(base_features),
    "n_manual_te": len(manual_te_names),
    "n_sklearn_pairwise_te": n_sklearn_te_cols,
    "n_total_features": len(base_features) + len(manual_te_names) + n_sklearn_te_cols,
}
with open(OUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone. Files saved to {OUT_DIR}")
print(f"Prediction distribution: {pd.Series([target_inv[p] for p in test_adjusted.argmax(1)]).value_counts().to_dict()}")
