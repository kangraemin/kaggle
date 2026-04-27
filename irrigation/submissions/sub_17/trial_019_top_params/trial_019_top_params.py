"""
Trial 019: Top Notebook Params (max_depth=4, lr=0.1, strong reg)
- Base: trial_013 architecture (750 features, orig append, bias tuning)
- Key change: XGB params from yunsuxiaozi/include4eto (CV 0.980)
  - max_depth=4, lr=0.1, alpha=5, lambda=5, max_bin=10000
  - early_stopping on balanced_accuracy
- 5-seed averaging
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
orig_y = orig["Irrigation_Need"].map(target_map).values

cat_cols = ["Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
            "Irrigation_Type", "Water_Source", "Mulching_Used", "Region"]
num_cols = ["Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
            "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
            "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm"]
all_raw_cols = cat_cols + num_cols

print(f"Train: {train.shape}, Test: {test.shape}, Orig: {orig.shape}")

# ============ Binary Threshold Features ============

for df in [train, test, orig]:
    df["soil_lt_25"] = (df["Soil_Moisture"] < 25).astype(int)
    df["temp_gt_30"] = (df["Temperature_C"] > 30).astype(int)
    df["rain_lt_300"] = (df["Rainfall_mm"] < 300).astype(int)
    df["wind_gt_10"] = (df["Wind_Speed_kmh"] > 10).astype(int)

binary_cols = ["soil_lt_25", "temp_gt_30", "rain_lt_300", "wind_gt_10"]

# ============ Original Data TE ============

orig["target_num"] = orig["Irrigation_Need"].map(target_map)
te_orig_cols = []
for col in all_raw_cols:
    te_map = orig.groupby(col)["target_num"].mean()
    col_name = f"{col}_te_orig"
    train[col_name] = train[col].map(te_map).fillna(te_map.mean())
    test[col_name] = test[col].map(te_map).fillna(te_map.mean())
    orig[col_name] = orig[col].map(te_map).fillna(te_map.mean())
    te_orig_cols.append(col_name)

# ============ ALL 171 Pairwise: factorize ============

pair_cols = []
for c1, c2 in combinations(all_raw_cols, 2):
    col_name = f"{c1}_x_{c2}"
    combined = pd.concat([
        train[c1].astype(str) + "_" + train[c2].astype(str),
        test[c1].astype(str) + "_" + test[c2].astype(str),
        orig[c1].astype(str) + "_" + orig[c2].astype(str),
    ])
    codes, _ = pd.factorize(combined)
    train[col_name] = codes[:len(train)]
    test[col_name] = codes[len(train):len(train)+len(test)]
    orig[col_name] = codes[len(train)+len(test):]
    pair_cols.append(col_name)

# ============ Label encode categoricals ============

le_cols = []
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train[col].astype(str), test[col].astype(str), orig[col].astype(str)])
    le.fit(combined)
    col_le = f"{col}_le"
    train[col_le] = le.transform(train[col].astype(str))
    test[col_le] = le.transform(test[col].astype(str))
    orig[col_le] = le.transform(orig[col].astype(str))
    le_cols.append(col_le)

base_features = num_cols + le_cols + binary_cols + te_orig_cols + pair_cols
print(f"Base features: {len(base_features)}")

X_orig_base_df = orig[base_features].copy()
X_orig_pair_arr = orig[pair_cols].values
N_ORIG = len(orig)

# ============ Bias Tuning ============

def tune_bias(proba, y_true):
    best_bias = np.zeros(3)
    best_score = balanced_accuracy_score(y_true, proba.argmax(1))
    for step in [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]:
        improved = True
        while improved:
            improved = False
            for ci in range(3):
                for d in (-1.0, 1.0):
                    c = best_bias.copy()
                    c[ci] += d * step
                    preds = np.argmax(np.log(proba + 1e-15) + c, axis=1)
                    s = balanced_accuracy_score(y_true, preds)
                    if s > best_score + 1e-9:
                        best_bias, best_score, improved = c, s, True
    return best_bias, best_score

# ============ Multi-seed CV ============

SEEDS = [42, 123, 456, 789, 2024]
N_FOLDS = 5

oof_seeds = np.zeros((len(SEEDS), len(train), 3))
test_seeds = np.zeros((len(SEEDS), len(test), 3))

for si, seed in enumerate(SEEDS):
    print(f"\n{'#'*60}")
    print(f"# SEED {seed} ({si+1}/{len(SEEDS)})")
    print(f"{'#'*60}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof = np.zeros((len(train), 3))
    test_preds = np.zeros((len(test), 3))

    for fold, (train_idx, val_idx) in enumerate(skf.split(train, y)):
        print(f"\n{'='*50} Seed {seed} Fold {fold} {'='*50}")
        y_tr = y[train_idx]
        y_val = y[val_idx]

        sw_synth = compute_sample_weight("balanced", y_tr)
        sw_orig = np.full(N_ORIG, 0.35)
        sw_combined = np.concatenate([sw_synth, sw_orig])
        y_combined = np.concatenate([y_tr, orig_y])

        # Manual Multiclass TE on cat_cols
        manual_te_names = []
        manual_te_train_all = {}
        manual_te_test_all = {}
        manual_te_orig_all = {}

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
                manual_te_train_all[col_te] = train[f"{col}_le"].map(smoothed).fillna(cls_mean).values
                manual_te_test_all[col_te] = test[f"{col}_le"].map(smoothed).fillna(cls_mean).values
                manual_te_orig_all[col_te] = orig[f"{col}_le"].map(smoothed).fillna(cls_mean).values
                manual_te_names.append(col_te)

        # sklearn TargetEncoder on pairwise
        X_pair_tr = train[pair_cols].iloc[train_idx].values
        X_pair_val = train[pair_cols].iloc[val_idx].values
        X_pair_te = test[pair_cols].values
        X_pair_combined = np.vstack([X_pair_tr, X_orig_pair_arr])

        sklearn_te = TargetEncoder(target_type="multiclass", smooth="auto", random_state=seed)
        X_pair_combined_te = sklearn_te.fit_transform(X_pair_combined, y_combined)
        X_pair_tr_te = X_pair_combined_te[:len(train_idx)]
        X_pair_orig_te = X_pair_combined_te[len(train_idx):]
        X_pair_val_te = sklearn_te.transform(X_pair_val)
        X_pair_test_te = sklearn_te.transform(X_pair_te)

        n_sklearn_te_cols = X_pair_tr_te.shape[1]

        # Build matrices
        X_tr_base = train[base_features].iloc[train_idx].copy()
        for col_te in manual_te_names:
            X_tr_base[col_te] = manual_te_train_all[col_te][train_idx]
        X_tr_np_synth = np.hstack([X_tr_base.values, X_pair_tr_te])

        X_orig_base_fold = X_orig_base_df.copy()
        for col_te in manual_te_names:
            X_orig_base_fold[col_te] = manual_te_orig_all[col_te]
        X_orig_np = np.hstack([X_orig_base_fold.values, X_pair_orig_te])

        X_tr_np = np.vstack([X_tr_np_synth, X_orig_np])

        X_val_base = train[base_features].iloc[val_idx].copy()
        for col_te in manual_te_names:
            X_val_base[col_te] = manual_te_train_all[col_te][val_idx]
        X_val_np = np.hstack([X_val_base.values, X_pair_val_te])

        X_te_base = test[base_features].copy()
        for col_te in manual_te_names:
            X_te_base[col_te] = manual_te_test_all[col_te]
        X_te_np = np.hstack([X_te_base.values, X_pair_test_te])

        if fold == 0 and si == 0:
            print(f"  Total features: {X_tr_np.shape[1]}")

        # ---- TOP PARAMS: shallow tree + fast lr + strong reg ----
        model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            n_estimators=50000,
            learning_rate=0.1,
            max_depth=4,
            max_leaves=30,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=5.0,
            reg_lambda=5.0,
            random_state=seed,
            tree_method="hist",
            max_bin=10000,
            verbosity=0,
            early_stopping_rounds=200,
        )
        model.fit(
            X_tr_np, y_combined,
            sample_weight=sw_combined,
            eval_set=[(X_val_np, y_val)],
            verbose=500,
        )

        best_rounds = model.best_iteration
        oof[val_idx] = model.predict_proba(X_val_np)
        test_preds += model.predict_proba(X_te_np) / N_FOLDS

        fold_bal = balanced_accuracy_score(y_val, oof[val_idx].argmax(1))
        print(f"  XGB fold {fold}: bal_acc={fold_bal:.6f}, best_rounds={best_rounds}")
        del model, sklearn_te; gc.collect()

    oof_seeds[si] = oof
    test_seeds[si] = test_preds
    seed_bal = balanced_accuracy_score(y, oof.argmax(1))
    print(f"\nSeed {seed} OOF: {seed_bal:.6f}")

# ============ Average ============

oof_avg = oof_seeds.mean(0)
test_avg = test_seeds.mean(0)

avg_bal = balanced_accuracy_score(y, oof_avg.argmax(1))
print(f"\nSeed-averaged OOF: {avg_bal:.6f}")

best_bias, best_bias_score = tune_bias(oof_avg, y)
print(f"Bias tuning: {best_bias.tolist()} -> {best_bias_score:.6f}")

test_logits = np.log(test_avg + 1e-15) + best_bias
test_final_preds = test_logits.argmax(1)

# ============ Save ============

np.save(OUT_DIR / "oof_preds.npy", oof_avg)
np.save(OUT_DIR / "test_preds.npy", test_avg)

sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
sub["Irrigation_Need"] = [target_inv[p] for p in test_final_preds]
sub.to_csv(OUT_DIR / "submission.csv", index=False)

print(f"\nDistribution: {pd.Series([target_inv[p] for p in test_final_preds]).value_counts().to_dict()}")

results = {
    "trial": "trial_019_top_params",
    "metric": "balanced_accuracy",
    "seeds": SEEDS,
    "n_folds": N_FOLDS,
    "n_total_models": len(SEEDS) * N_FOLDS,
    "xgb_params": {
        "lr": 0.1, "max_depth": 4, "max_leaves": 30,
        "min_child_weight": 2, "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 5.0, "reg_lambda": 5.0, "max_bin": 10000,
        "early_stopping": 200,
    },
    "oof_raw": round(avg_bal, 6),
    "bias": [round(float(b), 4) for b in best_bias],
    "oof_with_bias": round(best_bias_score, 6),
}
with open(OUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone. OOF: {best_bias_score:.6f}")
