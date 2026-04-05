"""
Trial 010: Multi-Seed XGB + CatBoost on trial_008b setup
- trial_008b 기반 (171 pairwise factorize + 24 multiclass TE + Deotte binary)
- XGB: 3 seeds (42, 123, 456) → proba 평균
- CatBoost 추가 (auto_class_weights=Balanced)
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
from catboost import CatBoostClassifier
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

# ============ Binary Threshold (Deotte) ============
for df in [train, test]:
    df["soil_lt_25"] = (df["Soil_Moisture"] < 25).astype(int)
    df["temp_gt_30"] = (df["Temperature_C"] > 30).astype(int)
    df["rain_lt_300"] = (df["Rainfall_mm"] < 300).astype(int)
    df["wind_gt_10"] = (df["Wind_Speed_kmh"] > 10).astype(int)
binary_cols = ["soil_lt_25", "temp_gt_30", "rain_lt_300", "wind_gt_10"]

# ============ Original TE ============
orig["target_num"] = orig["Irrigation_Need"].map(target_map)
te_orig_cols = []
for col in all_raw_cols:
    te_map = orig.groupby(col)["target_num"].mean()
    col_name = f"{col}_te_orig"
    train[col_name] = train[col].map(te_map).fillna(te_map.mean())
    test[col_name] = test[col].map(te_map).fillna(te_map.mean())
    te_orig_cols.append(col_name)

# ============ 171 Pairwise factorize ============
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
print(f"Added {len(pair_cols)} pairwise features")

# ============ Label encode ============
le_cols = []
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train[col].astype(str), test[col].astype(str)])
    le.fit(combined)
    col_le = f"{col}_le"
    train[col_le] = le.transform(train[col].astype(str))
    test[col_le] = le.transform(test[col].astype(str))
    le_cols.append(col_le)

base_features = num_cols + le_cols + binary_cols + te_orig_cols + pair_cols
sample_weights = compute_sample_weight("balanced", y)

# ============ Multi-Seed CV ============
SEEDS = [42, 123, 456]
N_FOLDS = 5

# Accumulators: average across seeds
oof_xgb_all = np.zeros((len(train), 3))
oof_cat_all = np.zeros((len(train), 3))
test_xgb_all = np.zeros((len(test), 3))
test_cat_all = np.zeros((len(test), 3))

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n{'#'*60} SEED {seed} ({seed_idx+1}/{len(SEEDS)}) {'#'*60}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof_xgb = np.zeros((len(train), 3))
    oof_cat = np.zeros((len(train), 3))
    test_xgb = np.zeros((len(test), 3))
    test_cat = np.zeros((len(test), 3))

    for fold, (train_idx, val_idx) in enumerate(skf.split(train, y)):
        print(f"\n{'='*40} Seed {seed} Fold {fold} {'='*40}")
        y_tr, y_val = y[train_idx], y[val_idx]
        sw_tr = sample_weights[train_idx]

        # Multiclass TE on cat_cols
        te_train = {}
        te_test = {}
        te_col_names = []
        for col in cat_cols:
            for cls in range(3):
                col_te = f"{col}_te{cls}"
                cls_mean = (y_tr == cls).mean()
                vals = train.iloc[train_idx][f"{col}_le"]
                df_grp = pd.DataFrame({"val": vals, "is_cls": (y_tr == cls).astype(int)})
                grp_mean = df_grp.groupby("val")["is_cls"].mean()
                grp_count = df_grp.groupby("val")["is_cls"].count()
                m = 100
                smoothed = (grp_mean * grp_count + cls_mean * m) / (grp_count + m)
                te_train[col_te] = train[f"{col}_le"].map(smoothed).fillna(cls_mean).values
                te_test[col_te] = test[f"{col}_le"].map(smoothed).fillna(cls_mean).values
                te_col_names.append(col_te)

        X_tr = train[base_features].iloc[train_idx].copy()
        X_val = train[base_features].iloc[val_idx].copy()
        X_te = test[base_features].copy()
        for col_te in te_col_names:
            X_tr[col_te] = te_train[col_te][train_idx]
            X_val[col_te] = te_train[col_te][val_idx]
            X_te[col_te] = te_test[col_te]

        if fold == 0 and seed_idx == 0:
            print(f"  Features: {X_tr.shape[1]}")

        # XGB
        model_xgb = xgb.XGBClassifier(
            objective="multi:softprob", num_class=3, eval_metric="mlogloss",
            n_estimators=5000, learning_rate=0.01, max_depth=6,
            min_child_weight=30, subsample=0.8, colsample_bytree=0.6,
            reg_alpha=0.1, reg_lambda=1.0, random_state=seed,
            tree_method="hist", max_bin=1024, verbosity=0,
        )
        model_xgb.fit(X_tr, y_tr, sample_weight=sw_tr,
                      eval_set=[(X_val, y_val)], verbose=500)
        oof_xgb[val_idx] = model_xgb.predict_proba(X_val)
        test_xgb += model_xgb.predict_proba(X_te) / N_FOLDS
        print(f"  XGB  bal_acc: {balanced_accuracy_score(y_val, oof_xgb[val_idx].argmax(1)):.6f}")
        del model_xgb; gc.collect()

        # CatBoost
        model_cb = CatBoostClassifier(
            iterations=3000, learning_rate=0.03, depth=6,
            l2_leaf_reg=3, random_seed=seed, verbose=200,
            auto_class_weights="Balanced",
        )
        model_cb.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=100)
        oof_cat[val_idx] = model_cb.predict_proba(X_val)
        test_cat += model_cb.predict_proba(X_te) / N_FOLDS
        print(f"  CAT  bal_acc: {balanced_accuracy_score(y_val, oof_cat[val_idx].argmax(1)):.6f}")
        del model_cb; gc.collect()

    # Per-seed results
    xgb_acc = balanced_accuracy_score(y, oof_xgb.argmax(1))
    cat_acc = balanced_accuracy_score(y, oof_cat.argmax(1))
    print(f"\nSeed {seed} — XGB: {xgb_acc:.6f}, CAT: {cat_acc:.6f}")

    oof_xgb_all += oof_xgb / len(SEEDS)
    oof_cat_all += oof_cat / len(SEEDS)
    test_xgb_all += test_xgb / len(SEEDS)
    test_cat_all += test_cat / len(SEEDS)

# ============ Ensemble + Threshold ============
for name, oof in [("XGB_avg", oof_xgb_all), ("CAT_avg", oof_cat_all)]:
    print(f"\n{name} OOF balanced_accuracy: {balanced_accuracy_score(y, oof.argmax(1)):.6f}")

best_bal_acc = 0
best_w = (1, 0)
for w1 in range(0, 10):
    for w2 in range(0, 10):
        if w1 + w2 == 0: continue
        total = w1 + w2
        oof_ens = (w1 * oof_xgb_all + w2 * oof_cat_all) / total
        bal_acc = balanced_accuracy_score(y, oof_ens.argmax(1))
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_w = (w1, w2)

print(f"\nBest weights (xgb:cat): {best_w}")
print(f"Best ensemble OOF balanced_accuracy: {best_bal_acc:.6f}")

total = sum(best_w)
oof_ens = (best_w[0] * oof_xgb_all + best_w[1] * oof_cat_all) / total
test_ens = (best_w[0] * test_xgb_all + best_w[1] * test_cat_all) / total

best_threshold_acc = best_bal_acc
best_class_w = (1.0, 1.0, 1.0)
for w_low in np.arange(0.7, 1.3, 0.05):
    for w_med in np.arange(0.5, 1.3, 0.05):
        for w_high in np.arange(1.0, 4.0, 0.1):
            adjusted = oof_ens.copy()
            adjusted[:, 0] *= w_low
            adjusted[:, 1] *= w_med
            adjusted[:, 2] *= w_high
            bal_acc = balanced_accuracy_score(y, adjusted.argmax(1))
            if bal_acc > best_threshold_acc:
                best_threshold_acc = bal_acc
                best_class_w = (w_low, w_med, w_high)

print(f"\nThreshold: {best_class_w}")
print(f"After threshold: {best_threshold_acc:.6f} (was {best_bal_acc:.6f})")

test_adjusted = test_ens.copy()
test_adjusted[:, 0] *= best_class_w[0]
test_adjusted[:, 1] *= best_class_w[1]
test_adjusted[:, 2] *= best_class_w[2]

# ============ Save ============
np.save(OUT_DIR / "oof_preds.npy", np.stack([oof_xgb_all, oof_cat_all]))
np.save(OUT_DIR / "test_preds.npy", np.stack([test_xgb_all, test_cat_all]))

sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
sub["Irrigation_Need"] = [target_inv[p] for p in test_adjusted.argmax(1)]
sub.to_csv(OUT_DIR / "submission.csv", index=False)

results = {
    "trial": "trial_010_multiseed_cat",
    "metric": "balanced_accuracy",
    "seeds": SEEDS,
    "oof_bal_acc_xgb_avg": round(balanced_accuracy_score(y, oof_xgb_all.argmax(1)), 6),
    "oof_bal_acc_cat_avg": round(balanced_accuracy_score(y, oof_cat_all.argmax(1)), 6),
    "oof_bal_acc_ensemble": round(best_bal_acc, 6),
    "oof_bal_acc_threshold": round(best_threshold_acc, 6),
    "ensemble_weights_xgb_cat": list(best_w),
    "threshold_class_weights": [round(w, 3) for w in best_class_w],
}
with open(OUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone. {OUT_DIR}")
print(f"Dist: {pd.Series([target_inv[p] for p in test_adjusted.argmax(1)]).value_counts().to_dict()}")
