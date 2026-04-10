"""
Trial 015: Pseudo-Labeling on trial_011 architecture
- Round 1: trial_011 그대로 학습 → test predictions
- Round 2: 고확신 test predictions(>0.95)을 train에 추가 → 재학습
- High class(3.3%)가 보강되어 bal_acc 개선 기대
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
train_orig = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
orig = pd.read_csv(DATA_DIR / "original" / "irrigation_prediction.csv")

target_map = {"Low": 0, "Medium": 1, "High": 2}
target_inv = {v: k for k, v in target_map.items()}
y_orig = train_orig["Irrigation_Need"].map(target_map).values

cat_cols = ["Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
            "Irrigation_Type", "Water_Source", "Mulching_Used", "Region"]
num_cols = ["Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
            "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
            "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm"]
all_raw_cols = cat_cols + num_cols

SEED = 42
N_FOLDS = 5
CONFIDENCE_THRESHOLD = 0.95

print(f"Train: {train_orig.shape}, Test: {test.shape}")

# ============ Feature Engineering (shared function) ============

def prepare_features(train_df, test_df):
    """Prepare features identical to trial_011"""
    train = train_df.copy()
    tst = test_df.copy()

    # Binary threshold
    for df in [train, tst]:
        df["soil_lt_25"] = (df["Soil_Moisture"] < 25).astype(int)
        df["temp_gt_30"] = (df["Temperature_C"] > 30).astype(int)
        df["rain_lt_300"] = (df["Rainfall_mm"] < 300).astype(int)
        df["wind_gt_10"] = (df["Wind_Speed_kmh"] > 10).astype(int)
    binary_cols_local = ["soil_lt_25", "temp_gt_30", "rain_lt_300", "wind_gt_10"]

    # Original TE
    te_orig_cols_local = []
    for col in all_raw_cols:
        te_map = orig.groupby(col)["target_num"].mean()
        col_name = f"{col}_te_orig"
        train[col_name] = train[col].map(te_map).fillna(te_map.mean())
        tst[col_name] = tst[col].map(te_map).fillna(te_map.mean())
        te_orig_cols_local.append(col_name)

    # 171 Pairwise factorize
    pair_cols_local = []
    for c1, c2 in combinations(all_raw_cols, 2):
        col_name = f"{c1}_x_{c2}"
        combined = pd.concat([
            train[c1].astype(str) + "_" + train[c2].astype(str),
            tst[c1].astype(str) + "_" + tst[c2].astype(str)
        ])
        codes, _ = pd.factorize(combined)
        train[col_name] = codes[:len(train)]
        tst[col_name] = codes[len(train):]
        pair_cols_local.append(col_name)

    # Label encode
    le_cols_local = []
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([train[col].astype(str), tst[col].astype(str)])
        le.fit(combined)
        col_le = f"{col}_le"
        train[col_le] = le.transform(train[col].astype(str))
        tst[col_le] = le.transform(tst[col].astype(str))
        le_cols_local.append(col_le)

    base_features = num_cols + le_cols_local + binary_cols_local + te_orig_cols_local + pair_cols_local
    return train, tst, base_features, pair_cols_local

orig["target_num"] = orig["Irrigation_Need"].map(target_map)

# ============ ROUND 1: Normal training (like trial_011) ============

print("\n" + "="*60 + " ROUND 1: Normal " + "="*60)
train_r1, test_r1, base_features, pair_cols = prepare_features(train_orig, test)
sample_weights = compute_sample_weight("balanced", y_orig)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
test_proba_r1 = np.zeros((len(test), 3))

for fold, (train_idx, val_idx) in enumerate(skf.split(train_r1, y_orig)):
    print(f"\n--- Round 1 Fold {fold} ---")
    y_tr, y_val = y_orig[train_idx], y_orig[val_idx]
    sw_tr = sample_weights[train_idx]

    # Manual TE
    te_train, te_test_dict = {}, {}
    te_names = []
    for col in cat_cols:
        for cls in range(3):
            col_te = f"{col}_te{cls}"
            cls_mean = (y_tr == cls).mean()
            vals = train_r1.iloc[train_idx][f"{col}_le"]
            df_grp = pd.DataFrame({"val": vals, "is_cls": (y_tr == cls).astype(int)})
            grp_mean = df_grp.groupby("val")["is_cls"].mean()
            grp_count = df_grp.groupby("val")["is_cls"].count()
            m = 100
            smoothed = (grp_mean * grp_count + cls_mean * m) / (grp_count + m)
            te_train[col_te] = train_r1[f"{col}_le"].map(smoothed).fillna(cls_mean).values
            te_test_dict[col_te] = test_r1[f"{col}_le"].map(smoothed).fillna(cls_mean).values
            te_names.append(col_te)

    # sklearn pairwise TE
    X_pair_tr = train_r1[pair_cols].iloc[train_idx].values
    X_pair_te = test_r1[pair_cols].values
    sklearn_te = TargetEncoder(target_type="multiclass", smooth="auto", random_state=SEED)
    X_pair_tr_te = sklearn_te.fit_transform(X_pair_tr, y_tr)
    X_pair_test_te = sklearn_te.transform(X_pair_te)

    # Build features
    X_tr_base = train_r1[base_features].iloc[train_idx].copy()
    X_te_base = test_r1[base_features].copy()
    for col_te in te_names:
        X_tr_base[col_te] = te_train[col_te][train_idx]
        X_te_base[col_te] = te_test_dict[col_te]

    X_tr_np = np.hstack([X_tr_base.values, X_pair_tr_te])
    X_te_np = np.hstack([X_te_base.values, X_pair_test_te])

    model = xgb.XGBClassifier(
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        n_estimators=4000, learning_rate=0.01, max_depth=6,
        min_child_weight=30, subsample=0.8, colsample_bytree=0.4,
        reg_alpha=0.1, reg_lambda=1.0, random_state=SEED,
        tree_method="hist", max_bin=1024, verbosity=0,
    )
    model.fit(X_tr_np, y_tr, sample_weight=sw_tr, verbose=1000)
    test_proba_r1 += model.predict_proba(X_te_np) / N_FOLDS
    del model, sklearn_te; gc.collect()

# ============ Select high-confidence pseudo labels ============

test_preds_r1 = test_proba_r1.argmax(axis=1)
test_confidence = test_proba_r1.max(axis=1)
high_conf_mask = test_confidence >= CONFIDENCE_THRESHOLD

n_pseudo = high_conf_mask.sum()
pseudo_labels = test_preds_r1[high_conf_mask]
pseudo_dist = pd.Series(pseudo_labels).map(target_inv).value_counts().to_dict()
print(f"\nPseudo labels: {n_pseudo}/{len(test)} ({n_pseudo/len(test)*100:.1f}%) with confidence >= {CONFIDENCE_THRESHOLD}")
print(f"Pseudo distribution: {pseudo_dist}")

# ============ ROUND 2: Train with pseudo labels ============

print("\n" + "="*60 + " ROUND 2: Pseudo-Label " + "="*60)

# Combine train + pseudo-labeled test
test_pseudo = test.iloc[high_conf_mask].copy()
test_pseudo["Irrigation_Need"] = pd.Series(pseudo_labels).map(target_inv).values

train_combined = pd.concat([train_orig, test_pseudo], ignore_index=True)
y_combined = train_combined["Irrigation_Need"].map(target_map).values
print(f"Combined train: {train_combined.shape} (orig {len(train_orig)} + pseudo {n_pseudo})")

# Re-prepare features with combined train
train_r2, test_r2, base_features_r2, pair_cols_r2 = prepare_features(train_combined, test)
sample_weights_r2 = compute_sample_weight("balanced", y_combined)

# Only evaluate on original train samples
orig_indices = np.arange(len(train_orig))

skf2 = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_r2 = np.zeros((len(train_combined), 3))
test_proba_r2 = np.zeros((len(test), 3))

for fold, (train_idx, val_idx) in enumerate(skf2.split(train_r2, y_combined)):
    print(f"\n--- Round 2 Fold {fold} ---")
    y_tr = y_combined[train_idx]
    sw_tr = sample_weights_r2[train_idx]

    # Manual TE
    te_train, te_test_dict = {}, {}
    te_names = []
    for col in cat_cols:
        for cls in range(3):
            col_te = f"{col}_te{cls}"
            cls_mean = (y_tr == cls).mean()
            vals = train_r2.iloc[train_idx][f"{col}_le"]
            df_grp = pd.DataFrame({"val": vals, "is_cls": (y_tr == cls).astype(int)})
            grp_mean = df_grp.groupby("val")["is_cls"].mean()
            grp_count = df_grp.groupby("val")["is_cls"].count()
            m = 100
            smoothed = (grp_mean * grp_count + cls_mean * m) / (grp_count + m)
            te_train[col_te] = train_r2[f"{col}_le"].map(smoothed).fillna(cls_mean).values
            te_test_dict[col_te] = test_r2[f"{col}_le"].map(smoothed).fillna(cls_mean).values
            te_names.append(col_te)

    # sklearn pairwise TE
    X_pair_tr = train_r2[pair_cols_r2].iloc[train_idx].values
    X_pair_val = train_r2[pair_cols_r2].iloc[val_idx].values
    X_pair_te = test_r2[pair_cols_r2].values
    sklearn_te = TargetEncoder(target_type="multiclass", smooth="auto", random_state=SEED)
    X_pair_tr_te = sklearn_te.fit_transform(X_pair_tr, y_tr)
    X_pair_val_te = sklearn_te.transform(X_pair_val)
    X_pair_test_te = sklearn_te.transform(X_pair_te)

    # Build features
    X_tr_base = train_r2[base_features_r2].iloc[train_idx].copy()
    X_val_base = train_r2[base_features_r2].iloc[val_idx].copy()
    X_te_base = test_r2[base_features_r2].copy()
    for col_te in te_names:
        X_tr_base[col_te] = te_train[col_te][train_idx]
        X_val_base[col_te] = te_train[col_te][val_idx]
        X_te_base[col_te] = te_test_dict[col_te]

    X_tr_np = np.hstack([X_tr_base.values, X_pair_tr_te])
    X_val_np = np.hstack([X_val_base.values, X_pair_val_te])
    X_te_np = np.hstack([X_te_base.values, X_pair_test_te])

    model = xgb.XGBClassifier(
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        n_estimators=4000, learning_rate=0.01, max_depth=6,
        min_child_weight=30, subsample=0.8, colsample_bytree=0.4,
        reg_alpha=0.1, reg_lambda=1.0, random_state=SEED,
        tree_method="hist", max_bin=1024, verbosity=0,
    )
    model.fit(X_tr_np, y_tr, sample_weight=sw_tr, verbose=1000)
    oof_r2[val_idx] = model.predict_proba(X_val_np)
    test_proba_r2 += model.predict_proba(X_te_np) / N_FOLDS

    # Score on original samples in val
    orig_val = [i for i in val_idx if i < len(train_orig)]
    if orig_val:
        fold_bal = balanced_accuracy_score(y_orig[orig_val], oof_r2[orig_val].argmax(1))
        print(f"  Fold {fold} bal_acc (orig only): {fold_bal:.6f}")

    del model, sklearn_te; gc.collect()

# OOF on original train only
oof_orig = oof_r2[:len(train_orig)]
raw_bal_acc = balanced_accuracy_score(y_orig, oof_orig.argmax(1))
print(f"\nRound 2 OOF balanced_accuracy (orig only): {raw_bal_acc:.6f}")

# ============ Threshold ============
best_threshold_acc = raw_bal_acc
best_class_w = (1.0, 1.0, 1.0)

for w_low in np.arange(0.7, 1.3, 0.05):
    for w_med in np.arange(0.5, 1.3, 0.05):
        for w_high in np.arange(1.0, 5.0, 0.1):
            adjusted = oof_orig.copy()
            adjusted[:, 0] *= w_low
            adjusted[:, 1] *= w_med
            adjusted[:, 2] *= w_high
            bal_acc = balanced_accuracy_score(y_orig, adjusted.argmax(1))
            if bal_acc > best_threshold_acc:
                best_threshold_acc = bal_acc
                best_class_w = (w_low, w_med, w_high)

print(f"Threshold: {best_class_w}")
print(f"After threshold: {best_threshold_acc:.6f} (was {raw_bal_acc:.6f})")

test_adjusted = test_proba_r2.copy()
test_adjusted[:, 0] *= best_class_w[0]
test_adjusted[:, 1] *= best_class_w[1]
test_adjusted[:, 2] *= best_class_w[2]

# ============ Save ============
np.save(OUT_DIR / "oof_preds.npy", oof_orig)
np.save(OUT_DIR / "test_preds.npy", test_proba_r2)

sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
sub["Irrigation_Need"] = [target_inv[p] for p in test_adjusted.argmax(1)]
sub.to_csv(OUT_DIR / "submission.csv", index=False)

results = {
    "trial": "trial_015_pseudo_label",
    "metric": "balanced_accuracy",
    "confidence_threshold": CONFIDENCE_THRESHOLD,
    "n_pseudo_labels": int(n_pseudo),
    "pseudo_distribution": pseudo_dist,
    "oof_bal_acc_raw": round(raw_bal_acc, 6),
    "oof_bal_acc_threshold": round(best_threshold_acc, 6),
    "threshold_class_weights": [round(w, 3) for w in best_class_w],
}
with open(OUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone. {OUT_DIR}")
