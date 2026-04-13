"""
Trial 016: Digit Features + Magic Score + Deotte Logit Features
- Base: trial_013 architecture (3-seed XGB, 750 features, orig append, bias tuning)
- Key change 1: Digit features — extract decimal digits from num_cols (synthetic data artifact)
- Key change 2: Magic score — hand-crafted high/low irrigation signal
- Key change 3: Deotte logit features — exact formula logistic regression logit as feature (3 cols)
- Key change 4: 5 seeds (was 3) for more variance reduction
- Expected: close 0.0025 gap to top LB (0.9808)
"""

import gc
import json
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, TargetEncoder
from sklearn.linear_model import LogisticRegression
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

# ============ Binary Threshold Features (Deotte) ============

for df in [train, test, orig]:
    df["soil_lt_25"] = (df["Soil_Moisture"] < 25).astype(int)
    df["temp_gt_30"] = (df["Temperature_C"] > 30).astype(int)
    df["rain_lt_300"] = (df["Rainfall_mm"] < 300).astype(int)
    df["wind_gt_10"] = (df["Wind_Speed_kmh"] > 10).astype(int)

binary_cols = ["soil_lt_25", "temp_gt_30", "rain_lt_300", "wind_gt_10"]

# ============ NEW: Magic Score (BlamerX) ============

for df in [train, test, orig]:
    df["is_harvest"] = (df["Crop_Growth_Stage"] == "Harvest").astype(int)
    df["is_sowing"] = (df["Crop_Growth_Stage"] == "Sowing").astype(int)
    df["mulching_yes"] = (df["Mulching_Used"] == "Yes").astype(int)

    high_score = 2 * df["soil_lt_25"] + 2 * df["rain_lt_300"] + df["temp_gt_30"] + df["wind_gt_10"]
    low_score = 2 * df["is_harvest"] + 2 * df["is_sowing"] + df["mulching_yes"]
    df["magic_score"] = high_score - low_score

magic_cols = ["magic_score", "is_harvest", "is_sowing", "mulching_yes"]

# ============ NEW: Deotte Logit Features ============
# Fit logistic regression on original data's exact 6-feature formula
# Then use logit scores (log-odds for each class) as features

deotte_features = ["soil_lt_25", "temp_gt_30", "rain_lt_300", "wind_gt_10"]
# Also need Crop_Growth_Stage and Mulching_Used as dummies
for stage in orig["Crop_Growth_Stage"].unique():
    col = f"stage_{stage}"
    for df in [train, test, orig]:
        df[col] = (df["Crop_Growth_Stage"] == stage).astype(int)

for mul in orig["Mulching_Used"].unique():
    col = f"mulch_{mul}"
    for df in [train, test, orig]:
        df[col] = (df["Mulching_Used"] == mul).astype(int)

# Build feature matrix for logistic regression
stage_dummies = [f"stage_{s}" for s in orig["Crop_Growth_Stage"].unique()]
mulch_dummies = [f"mulch_{m}" for m in orig["Mulching_Used"].unique()]
lr_features = deotte_features + stage_dummies[:-1] + mulch_dummies[:-1]  # drop last for collinearity

lr_model = LogisticRegression(max_iter=1000, C=1e6, solver="lbfgs")
lr_model.fit(orig[lr_features].values, orig_y)

# Get logit scores (decision_function returns raw logits)
logit_cols = []
for df_name, df in [("train", train), ("test", test), ("orig", orig)]:
    logits = lr_model.decision_function(df[lr_features].values)  # (N, 3)
    for ci in range(3):
        col = f"deotte_logit_{ci}"
        df[col] = logits[:, ci]
        if df_name == "train" and ci not in [c for c in range(3) if f"deotte_logit_{c}" in logit_cols]:
            logit_cols.append(col)

print(f"Deotte logit features: {logit_cols}")
print(f"LR accuracy on orig: {lr_model.score(orig[lr_features].values, orig_y):.4f}")

# ============ NEW: Digit Features ============
# Extract individual digits at each decimal position from numerical columns

digit_cols = []
for col in num_cols:
    for df in [train, test, orig]:
        # Multiply by powers of 10 to access different digit positions
        for k in range(-2, 3):  # positions: 0.01, 0.1, 1, 10, 100
            digit_col = f"{col}_digit_{k}"
            df[digit_col] = ((df[col].abs() * (10 ** (-k))).astype(np.int64) % 10).astype(np.int8)
            if digit_col not in digit_cols:
                digit_cols.append(digit_col)

print(f"Digit features: {len(digit_cols)} columns")

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

print(f"Pairwise features: {len(pair_cols)}")

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

# ============ Base features (now with new features) ============

base_features = num_cols + le_cols + binary_cols + te_orig_cols + pair_cols + magic_cols + logit_cols + digit_cols
print(f"Base features: {len(base_features)}")

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

# ============ Multi-seed CV Loop (XGB only, 5 seeds) ============

SEEDS = [42, 123, 456, 789, 2024]
N_FOLDS = 5
N_ORIG = len(orig)

oof_seeds = np.zeros((len(SEEDS), len(train), 3))
test_seeds = np.zeros((len(SEEDS), len(test), 3))

X_orig_base_df = orig[base_features].copy()
X_orig_pair_arr = orig[pair_cols].values

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

        # Manual Multiclass TE on cat_cols (24 cols)
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

        # sklearn TargetEncoder on ALL 171 pairwise
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

        # Build full feature matrices
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
            total_features = X_tr_np.shape[1]
            print(f"  Total features: {total_features}")
            print(f"  X_tr shape (with orig): {X_tr_np.shape}")

        # XGBoost
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

        oof[val_idx] = model_xgb.predict_proba(X_val_np)
        test_preds += model_xgb.predict_proba(X_te_np) / N_FOLDS

        fold_bal = balanced_accuracy_score(y_val, oof[val_idx].argmax(1))
        print(f"  XGB fold {fold}: bal_acc={fold_bal:.6f}")
        del model_xgb, sklearn_te; gc.collect()

    oof_seeds[si] = oof
    test_seeds[si] = test_preds

    seed_bal = balanced_accuracy_score(y, oof.argmax(1))
    print(f"\nSeed {seed} OOF: XGB={seed_bal:.6f}")

# ============ Average across seeds ============

oof_avg = oof_seeds.mean(0)
test_avg = test_seeds.mean(0)

avg_bal = balanced_accuracy_score(y, oof_avg.argmax(1))
print(f"\nSeed-averaged OOF: {avg_bal:.6f}")

# ============ Bias Tuning ============

print("\nRunning coordinate descent bias tuning...")
best_bias, best_bias_score = tune_bias(oof_avg, y)
print(f"Bias tuning: {best_bias.tolist()} -> OOF bal_acc = {best_bias_score:.6f} (was {avg_bal:.6f})")

test_logits = np.log(test_avg + 1e-15) + best_bias
test_final_preds = test_logits.argmax(1)

# ============ Save ============

np.save(OUT_DIR / "oof_preds.npy", oof_avg)
np.save(OUT_DIR / "test_preds.npy", test_avg)

sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
sub["Irrigation_Need"] = [target_inv[p] for p in test_final_preds]
sub.to_csv(OUT_DIR / "submission.csv", index=False)

print(f"\nPrediction distribution: {pd.Series([target_inv[p] for p in test_final_preds]).value_counts().to_dict()}")

results = {
    "trial": "trial_016_digit_magic_logit",
    "metric": "balanced_accuracy",
    "seeds": SEEDS,
    "n_folds": N_FOLDS,
    "n_total_models": len(SEEDS) * N_FOLDS,
    "new_features": {
        "digit_features": len(digit_cols),
        "magic_score": len(magic_cols),
        "logit_features": len(logit_cols),
    },
    "oof_raw": round(avg_bal, 6),
    "bias": [round(float(b), 4) for b in best_bias],
    "oof_with_bias": round(best_bias_score, 6),
    "n_base_features": len(base_features),
    "n_manual_te": 24,
    "n_sklearn_pairwise_te": 513,
    "n_total_features": len(base_features) + 24 + 513,
}
with open(OUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone. Files saved to {OUT_DIR}")
print(f"OOF balanced_accuracy (final): {best_bias_score:.6f}")
