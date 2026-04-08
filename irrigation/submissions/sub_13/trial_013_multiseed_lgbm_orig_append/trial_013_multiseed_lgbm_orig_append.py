"""
Trial 013: Multi-seed XGB+LGBM Ensemble + Original Data Append (w=0.35) + Log-odds Bias Tuning
- Base: trial_011 architecture (750 features: 213 base + 24 manual TE + 513 sklearn pairwise TE)
- Key change 1: 3 seeds x 5 folds x 2 models (XGB + LGBM) = 30 total models
- Key change 2: Original rows APPENDED to each training fold (weight=0.35)
- Key change 3: Coordinate descent log-odds bias tuning (UtaAzu method)
- Key change 4: Hill climbing alpha blend (XGB vs LGBM)
- Expected OOF: 0.980~0.981
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
orig_y = orig["Irrigation_Need"].map(target_map).values

cat_cols = ["Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
            "Irrigation_Type", "Water_Source", "Mulching_Used", "Region"]
num_cols = ["Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
            "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
            "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm"]
all_raw_cols = cat_cols + num_cols

print(f"Train: {train.shape}, Test: {test.shape}, Orig: {orig.shape}")

# ============ Binary Threshold Features (Deotte) ============

for df in [train, test, orig]:
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
    orig[col_name] = orig[col].map(te_map).fillna(te_map.mean())
    te_orig_cols.append(col_name)

# ============ ALL 171 Pairwise: factorize (incl. orig) ============

pair_cols = []
pair_definitions = []
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
    pair_definitions.append((c1, c2, col_name))

print(f"Added {len(pair_cols)} pairwise features (factorize)")

# ============ Label encode categoricals (incl. orig) ============

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

# ============ Base features ============

base_features = num_cols + le_cols + binary_cols + te_orig_cols + pair_cols
print(f"Base features: {len(base_features)}")

# ============ Bias Tuning (coordinate descent, log-odds space) ============

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

# ============ Multi-seed CV Loop ============

SEEDS = [42, 123, 456]
N_FOLDS = 5
N_ORIG = len(orig)

oof_xgb_seeds = np.zeros((len(SEEDS), len(train), 3))
oof_lgbm_seeds = np.zeros((len(SEEDS), len(train), 3))
test_xgb_seeds = np.zeros((len(SEEDS), len(test), 3))
test_lgbm_seeds = np.zeros((len(SEEDS), len(test), 3))

# Precompute orig base features (same columns as base_features)
X_orig_base_df = orig[base_features].copy()
X_orig_pair_arr = orig[pair_cols].values  # (N_ORIG, 171)

for si, seed in enumerate(SEEDS):
    print(f"\n{'#'*60}")
    print(f"# SEED {seed} ({si+1}/{len(SEEDS)})")
    print(f"{'#'*60}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof_xgb = np.zeros((len(train), 3))
    oof_lgbm = np.zeros((len(train), 3))
    test_xgb = np.zeros((len(test), 3))
    test_lgbm = np.zeros((len(test), 3))

    for fold, (train_idx, val_idx) in enumerate(skf.split(train, y)):
        print(f"\n{'='*50} Seed {seed} Fold {fold} {'='*50}")
        y_tr = y[train_idx]
        y_val = y[val_idx]

        # Balanced sample weights for synthetic train rows
        sw_synth = compute_sample_weight("balanced", y_tr)
        # Fixed weight for orig rows
        sw_orig = np.full(N_ORIG, 0.35)
        # Combined
        sw_combined = np.concatenate([sw_synth, sw_orig])
        y_combined = np.concatenate([y_tr, orig_y])

        # ----------------------------------------------------------------
        # 1) Manual Multiclass TE on cat_cols (8 cols × 3 classes = 24 TE)
        # ----------------------------------------------------------------
        manual_te_names = []
        manual_te_train_all = {}  # all train rows
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

        # ----------------------------------------------------------------
        # 2) sklearn TargetEncoder on ALL 171 pairwise (fit incl. orig rows)
        # ----------------------------------------------------------------
        X_pair_tr = train[pair_cols].iloc[train_idx].values   # (n_tr, 171)
        X_pair_val = train[pair_cols].iloc[val_idx].values
        X_pair_te = test[pair_cols].values

        # Combine train fold + orig for fitting
        X_pair_combined = np.vstack([X_pair_tr, X_orig_pair_arr])  # (n_tr+N_ORIG, 171)

        sklearn_te = TargetEncoder(target_type="multiclass", smooth="auto", random_state=seed)
        X_pair_combined_te = sklearn_te.fit_transform(X_pair_combined, y_combined)  # (n_tr+N_ORIG, 513)

        X_pair_tr_te = X_pair_combined_te[:len(train_idx)]     # synthetic train part
        X_pair_orig_te = X_pair_combined_te[len(train_idx):]   # orig part

        X_pair_val_te = sklearn_te.transform(X_pair_val)
        X_pair_test_te = sklearn_te.transform(X_pair_te)

        n_sklearn_te_cols = X_pair_tr_te.shape[1]
        if fold == 0 and si == 0:
            print(f"  sklearn pairwise TE: {n_sklearn_te_cols} columns")

        # ----------------------------------------------------------------
        # 3) Build full feature matrices (synthetic train + orig append)
        # ----------------------------------------------------------------
        # Synthetic training rows
        X_tr_base = train[base_features].iloc[train_idx].copy()
        for col_te in manual_te_names:
            X_tr_base[col_te] = manual_te_train_all[col_te][train_idx]
        X_tr_np_synth = np.hstack([X_tr_base.values, X_pair_tr_te])

        # Orig rows (appended to training)
        X_orig_base_fold = X_orig_base_df.copy()
        for col_te in manual_te_names:
            X_orig_base_fold[col_te] = manual_te_orig_all[col_te]
        X_orig_np = np.hstack([X_orig_base_fold.values, X_pair_orig_te])

        # Combined training set
        X_tr_np = np.vstack([X_tr_np_synth, X_orig_np])

        # Val and test
        X_val_base = train[base_features].iloc[val_idx].copy()
        for col_te in manual_te_names:
            X_val_base[col_te] = manual_te_train_all[col_te][val_idx]
        X_val_np = np.hstack([X_val_base.values, X_pair_val_te])

        X_te_base = test[base_features].copy()
        for col_te in manual_te_names:
            X_te_base[col_te] = manual_te_test_all[col_te]
        X_te_np = np.hstack([X_te_base.values, X_pair_test_te])

        if fold == 0 and si == 0:
            total_features = X_tr_np.shape[1]
            print(f"  Total features: {total_features} (base {len(base_features)} + manual_TE {len(manual_te_names)} + sklearn_TE {n_sklearn_te_cols})")
            print(f"  X_tr shape (with orig): {X_tr_np.shape}")

        # ----------------------------------------------------------------
        # 4) XGBoost: Same as trial_011 (lr=0.01, 4000 rounds hard cap)
        # ----------------------------------------------------------------
        model_xgb = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            n_estimators=4000,
            learning_rate=0.01,
            max_depth=6,
            min_child_weight=30,
            subsample=0.8,
            colsample_bytree=0.4,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=seed,
            tree_method="hist",
            max_bin=1024,
            verbosity=0,
        )
        model_xgb.fit(
            X_tr_np, y_combined,
            sample_weight=sw_combined,
            eval_set=[(X_val_np, y_val)],
            verbose=1000,
        )

        oof_xgb[val_idx] = model_xgb.predict_proba(X_val_np)
        test_xgb += model_xgb.predict_proba(X_te_np) / N_FOLDS

        fold_xgb_bal = balanced_accuracy_score(y_val, oof_xgb[val_idx].argmax(1))
        print(f"  XGB fold {fold}: bal_acc={fold_xgb_bal:.6f}")
        del model_xgb; gc.collect()

        # ----------------------------------------------------------------
        # 5) LightGBM: num_leaves=127, lr=0.03, 3000 rounds, early_stop=100
        # ----------------------------------------------------------------
        model_lgbm = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            metric="multi_logloss",
            num_leaves=127,
            learning_rate=0.03,
            n_estimators=3000,
            min_child_samples=50,
            colsample_bytree=0.6,
            subsample=0.8,
            subsample_freq=1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=seed,
            verbosity=-1,
            n_jobs=-1,
        )
        model_lgbm.fit(
            X_tr_np, y_combined,
            sample_weight=sw_combined,
            eval_set=[(X_val_np, y_val)],
            callbacks=[
                lgb.early_stopping(100, verbose=False),
                lgb.log_evaluation(500),
            ],
        )

        best_lgbm_rounds = model_lgbm.best_iteration_
        oof_lgbm[val_idx] = model_lgbm.predict_proba(X_val_np)
        test_lgbm += model_lgbm.predict_proba(X_te_np) / N_FOLDS

        fold_lgbm_bal = balanced_accuracy_score(y_val, oof_lgbm[val_idx].argmax(1))
        print(f"  LGBM fold {fold}: bal_acc={fold_lgbm_bal:.6f}, best_rounds={best_lgbm_rounds}")
        del model_lgbm, sklearn_te; gc.collect()

    oof_xgb_seeds[si] = oof_xgb
    oof_lgbm_seeds[si] = oof_lgbm
    test_xgb_seeds[si] = test_xgb
    test_lgbm_seeds[si] = test_lgbm

    xgb_seed_bal = balanced_accuracy_score(y, oof_xgb.argmax(1))
    lgbm_seed_bal = balanced_accuracy_score(y, oof_lgbm.argmax(1))
    print(f"\nSeed {seed} OOF: XGB={xgb_seed_bal:.6f}, LGBM={lgbm_seed_bal:.6f}")

# ============ Average across seeds ============

oof_xgb_avg = oof_xgb_seeds.mean(0)
oof_lgbm_avg = oof_lgbm_seeds.mean(0)
test_xgb_avg = test_xgb_seeds.mean(0)
test_lgbm_avg = test_lgbm_seeds.mean(0)

xgb_avg_bal = balanced_accuracy_score(y, oof_xgb_avg.argmax(1))
lgbm_avg_bal = balanced_accuracy_score(y, oof_lgbm_avg.argmax(1))
print(f"\nSeed-averaged OOF: XGB={xgb_avg_bal:.6f}, LGBM={lgbm_avg_bal:.6f}")

# ============ Hill Climbing Ensemble Weights ============

best_alpha = 0.5
best_blend_score = 0.0

for alpha in np.arange(0.0, 1.05, 0.05):
    blended = alpha * oof_xgb_avg + (1 - alpha) * oof_lgbm_avg
    score = balanced_accuracy_score(y, blended.argmax(1))
    if score > best_blend_score:
        best_blend_score = score
        best_alpha = alpha

print(f"\nHill climbing: best alpha (XGB weight) = {best_alpha:.2f}, OOF = {best_blend_score:.6f}")

blended_oof = best_alpha * oof_xgb_avg + (1 - best_alpha) * oof_lgbm_avg
blended_test = best_alpha * test_xgb_avg + (1 - best_alpha) * test_lgbm_avg

# ============ Coordinate Descent Log-odds Bias Tuning ============

print("\nRunning coordinate descent bias tuning...")
best_bias, best_bias_score = tune_bias(blended_oof, y)

print(f"Bias tuning: {best_bias.tolist()} -> OOF bal_acc = {best_bias_score:.6f} (was {best_blend_score:.6f})")

# Apply bias to test predictions
test_logits = np.log(blended_test + 1e-15) + best_bias
test_final_preds = test_logits.argmax(1)

# ============ Save ============

# Save OOF and test arrays (blended, pre-bias)
np.save(OUT_DIR / "oof_preds.npy", blended_oof)
np.save(OUT_DIR / "test_preds.npy", blended_test)

sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
sub["Irrigation_Need"] = [target_inv[p] for p in test_final_preds]
sub.to_csv(OUT_DIR / "submission.csv", index=False)

print(f"\nPrediction distribution: {pd.Series([target_inv[p] for p in test_final_preds]).value_counts().to_dict()}")

results = {
    "trial": "trial_013_multiseed_lgbm_orig_append",
    "metric": "balanced_accuracy",
    "seeds": SEEDS,
    "n_folds": N_FOLDS,
    "n_total_models": len(SEEDS) * N_FOLDS * 2,
    "xgb_lr": 0.01,
    "xgb_n_estimators": 4000,
    "xgb_early_stopping": "none_hard_cap",
    "lgbm_num_leaves": 127,
    "lgbm_lr": 0.03,
    "lgbm_n_estimators": 3000,
    "lgbm_early_stopping": 100,
    "orig_append_weight": 0.35,
    "oof_xgb_avg": round(xgb_avg_bal, 6),
    "oof_lgbm_avg": round(lgbm_avg_bal, 6),
    "hill_climb_alpha_xgb": round(best_alpha, 2),
    "oof_blend_before_bias": round(best_blend_score, 6),
    "bias": [round(float(b), 4) for b in best_bias],
    "oof_accuracy": round(best_blend_score, 6),
    "oof_accuracy_ensemble": round(best_bias_score, 6),
    "n_base_features": len(base_features),
    "n_manual_te": len(manual_te_names),
    "n_sklearn_pairwise_te": n_sklearn_te_cols,
    "n_total_features": len(base_features) + len(manual_te_names) + n_sklearn_te_cols,
}
with open(OUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone. Files saved to {OUT_DIR}")
print(f"OOF balanced_accuracy (final): {best_bias_score:.6f}")
