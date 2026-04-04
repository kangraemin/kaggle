"""
Trial 007: Bias-Tuned Stacked Ensemble
- Original data APPEND with weight=0.35
- Manual multiclass TE on CATS within fold (fast)
- Stacking meta-learner (Ridge/LGB on 9 OOF prob columns)
- Bias tuning on log-probabilities (coordinate descent)
- Updated model params: CatBoost depth=8, LGB leaves=127, XGB lr=0.03
- Balanced class weights throughout
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.linear_model import RidgeClassifier
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

combo_source = [(col, f"{col}_enc") for col in CATS] + [(col, col) for col in NUMS_BINNED]

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

# Concat all at once to avoid fragmentation
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

# ============ Label Encode Original CATS ============
label_encoders = {}
for col in CATS:
    le = LabelEncoder()
    all_vals = pd.concat([train[col], test[col], original[col]]).astype(str)
    le.fit(all_vals)
    for df in [train, test, original]:
        df[col] = le.transform(df[col].astype(str))
    label_encoders[col] = le

# Final feature list
drop_cols = {"id", "Irrigation_Need"}
features = [c for c in train.columns if c not in drop_cols]
# Make sure original has the same features (minus any missing)
orig_features = [c for c in features if c in original.columns]
assert len(orig_features) == len(features), f"Original missing features: {set(features) - set(orig_features)}"

print(f"Total features (before fold TE): {len(features)}")

X = train[features].copy()
X_test = test[features].copy()
X_orig = original[features].copy()

# ============ Manual Multiclass Target Encoding (within fold) ============

def fold_multiclass_te(X_tr, X_val, X_test_fold, te_cols, y_tr, n_classes=3, smoothing=10):
    """Fast manual multiclass TE within fold."""
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

    X_tr = pd.concat([X_tr.copy(), pd.DataFrame(te_train_dict, index=X_tr.index)], axis=1)
    X_val = pd.concat([X_val.copy(), pd.DataFrame(te_val_dict, index=X_val.index)], axis=1)
    X_test_fold = pd.concat([X_test_fold.copy(), pd.DataFrame(te_test_dict, index=X_test_fold.index)], axis=1)
    return X_tr, X_val, X_test_fold, new_col_names


# ============ Custom Eval Metrics ============

def xgb_balanced_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    if y_pred.ndim == 1:
        preds = y_pred.reshape(len(y_true), 3)
    else:
        preds = y_pred
    y_hat = preds.argmax(axis=1)
    return balanced_accuracy_score(y_true, y_hat)


def lgbm_balanced_accuracy(y_true, y_pred):
    n_samples = len(y_true)
    y_pred_reshaped = y_pred.reshape(n_samples, 3)
    y_hat = y_pred_reshaped.argmax(axis=1)
    score = balanced_accuracy_score(y_true.astype(int), y_hat)
    return "bal_acc", score, True


# ============ Bias Tuning ============

def bias_tune(proba, y_true, n_classes=3, steps=None):
    """Bias tuning on log-probabilities. argmax(log(p) + bias) with coordinate descent."""
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


# ============ Model Parameters ============

lgbm_params = {
    "objective": "multiclass",
    "num_class": 3,
    "verbosity": -1,
    "n_estimators": 5000,
    "learning_rate": 0.01,
    "num_leaves": 127,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "is_unbalance": True,
}

xgb_params = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "verbosity": 0,
    "n_estimators": 5000,
    "learning_rate": 0.03,
    "max_depth": 6,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "tree_method": "hist",
    "max_bin": 1024,
    "early_stopping_rounds": 300,
}

cat_params = {
    "iterations": 1500,
    "learning_rate": 0.04,
    "depth": 8,
    "l2_leaf_reg": 4.0,
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

# TE columns to use (CATS only for speed; pairwise used as-is)
te_target_cols = CATS

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n{'='*50} Fold {fold} {'='*50}")
    y_tr, y_val = y[train_idx], y[val_idx]

    # --- Original data append ---
    X_tr = pd.concat([X.iloc[train_idx], X_orig], ignore_index=True)
    y_tr_full = np.concatenate([y_tr, y_orig])
    X_val = X.iloc[val_idx].copy()
    X_test_fold = X_test.copy()

    # Sample weights: balanced + original data at 0.35
    sw_train = compute_sample_weight("balanced", y_tr)
    sw_orig = compute_sample_weight("balanced", y_orig) * 0.35
    sample_w = np.concatenate([sw_train, sw_orig])

    # --- Manual multiclass TE on CATS ---
    X_tr, X_val, X_test_fold, te_new_cols = fold_multiclass_te(
        X_tr, X_val, X_test_fold, te_target_cols, y_tr_full
    )
    fold_features = list(X_tr.columns)
    print(f"  Fold features: {len(fold_features)} (TE: {len(te_new_cols)})")

    # --- LightGBM ---
    model_lgb = lgb.LGBMClassifier(**lgbm_params)
    model_lgb.fit(
        X_tr[fold_features], y_tr_full,
        eval_set=[(X_val[fold_features], y_val)],
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(500)],
    )
    oof_lgbm[val_idx] = model_lgb.predict_proba(X_val[fold_features])
    test_lgbm += model_lgb.predict_proba(X_test_fold[fold_features]) / N_FOLDS
    lgb_bal = balanced_accuracy_score(y_val, oof_lgbm[val_idx].argmax(1))
    print(f"  LGBM bal_acc: {lgb_bal:.6f}")

    # --- XGBoost ---
    model_xgb = xgb.XGBClassifier(**xgb_params)
    model_xgb.fit(
        X_tr[fold_features], y_tr_full,
        eval_set=[(X_val[fold_features], y_val)],
        sample_weight=sample_w,
        verbose=500,
    )
    oof_xgb[val_idx] = model_xgb.predict_proba(X_val[fold_features])
    test_xgb += model_xgb.predict_proba(X_test_fold[fold_features]) / N_FOLDS
    xgb_bal = balanced_accuracy_score(y_val, oof_xgb[val_idx].argmax(1))
    print(f"  XGB  bal_acc: {xgb_bal:.6f}")

    # --- CatBoost ---
    model_cb = CatBoostClassifier(**cat_params)
    model_cb.fit(
        X_tr[fold_features], y_tr_full,
        eval_set=(X_val[fold_features], y_val),
        early_stopping_rounds=200,
        sample_weight=sample_w,
    )
    oof_cat[val_idx] = model_cb.predict_proba(X_val[fold_features])
    test_cat += model_cb.predict_proba(X_test_fold[fold_features]) / N_FOLDS
    cat_bal = balanced_accuracy_score(y_val, oof_cat[val_idx].argmax(1))
    print(f"  CAT  bal_acc: {cat_bal:.6f}")


# ============ Individual Scores ============

indiv_scores = {}
for name, oof in [("LGBM", oof_lgbm), ("XGB", oof_xgb), ("CAT", oof_cat)]:
    bal = balanced_accuracy_score(y, oof.argmax(1))
    acc = accuracy_score(y, oof.argmax(1))
    indiv_scores[name.lower()] = round(bal, 6)
    print(f"\n{name} OOF: bal_acc={bal:.6f}, acc={acc:.6f}")

# ============ Strategy 1: Simple weight grid search ============

best_grid_bal = 0
best_grid_w = (1, 1, 1)
for w1 in range(0, 11):
    for w2 in range(0, 11):
        for w3 in range(0, 11):
            if w1 + w2 + w3 == 0:
                continue
            total = w1 + w2 + w3
            oof_ens = (w1 * oof_lgbm + w2 * oof_xgb + w3 * oof_cat) / total
            bal = balanced_accuracy_score(y, oof_ens.argmax(1))
            if bal > best_grid_bal:
                best_grid_bal = bal
                best_grid_w = (w1, w2, w3)

print(f"\nGrid search weights (lgbm:xgb:cat): {best_grid_w}, bal_acc={best_grid_bal:.6f}")

# ============ Strategy 2: Stacking meta-learner ============

# Build meta-features from OOF probabilities
meta_oof = np.hstack([oof_lgbm, oof_xgb, oof_cat])  # (n_samples, 9)
meta_test = np.hstack([test_lgbm, test_xgb, test_cat])

# Ridge meta-learner with internal CV
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_val_predict

print("\nStacking meta-learner (Ridge)...")
ridge_meta = RidgeClassifier(alpha=1.0, class_weight="balanced")
ridge_oof_preds = cross_val_predict(ridge_meta, meta_oof, y, cv=5, method="decision_function")
ridge_oof_labels = ridge_oof_preds.argmax(1)
ridge_bal = balanced_accuracy_score(y, ridge_oof_labels)
print(f"  Ridge meta bal_acc: {ridge_bal:.6f}")

# Fit full Ridge for test predictions
ridge_meta.fit(meta_oof, y)
ridge_test_decision = ridge_meta.decision_function(meta_test)

# LGB meta-learner
print("Stacking meta-learner (LGB)...")
lgb_meta_params = {
    "objective": "multiclass",
    "num_class": 3,
    "verbosity": -1,
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "is_unbalance": True,
}

lgb_meta_oof = np.zeros((len(y), 3))
lgb_meta_test = np.zeros((len(X_test), 3))
meta_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

for fold, (tr_idx, va_idx) in enumerate(meta_skf.split(meta_oof, y)):
    model_meta = lgb.LGBMClassifier(**lgb_meta_params)
    model_meta.fit(
        meta_oof[tr_idx], y[tr_idx],
        eval_set=[(meta_oof[va_idx], y[va_idx])],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)],
    )
    lgb_meta_oof[va_idx] = model_meta.predict_proba(meta_oof[va_idx])
    lgb_meta_test += model_meta.predict_proba(meta_test) / 5

lgb_meta_bal = balanced_accuracy_score(y, lgb_meta_oof.argmax(1))
print(f"  LGB meta bal_acc: {lgb_meta_bal:.6f}")

# ============ Strategy 3: Blend + meta mixtures ============

# Best simple blend probabilities
total_w = sum(best_grid_w)
oof_blend = (best_grid_w[0] * oof_lgbm + best_grid_w[1] * oof_xgb + best_grid_w[2] * oof_cat) / total_w
test_blend = (best_grid_w[0] * test_lgbm + best_grid_w[1] * test_xgb + best_grid_w[2] * test_cat) / total_w

# Try blend + meta (LGB) mixtures
best_mix_bal = 0
best_mix_alpha = 0
best_mix_name = ""
for alpha in np.arange(0, 1.01, 0.05):
    mix = alpha * oof_blend + (1 - alpha) * lgb_meta_oof
    bal = balanced_accuracy_score(y, mix.argmax(1))
    if bal > best_mix_bal:
        best_mix_bal = bal
        best_mix_alpha = alpha
        best_mix_name = "blend+lgb_meta"

# Also try blend + ridge (using softmax of decision function)
from scipy.special import softmax as sp_softmax
ridge_oof_proba = sp_softmax(ridge_oof_preds, axis=1)
ridge_test_proba = sp_softmax(ridge_test_decision, axis=1)

for alpha in np.arange(0, 1.01, 0.05):
    mix = alpha * oof_blend + (1 - alpha) * ridge_oof_proba
    bal = balanced_accuracy_score(y, mix.argmax(1))
    if bal > best_mix_bal:
        best_mix_bal = bal
        best_mix_alpha = alpha
        best_mix_name = "blend+ridge_meta"

print(f"\nBest mixture: {best_mix_name}, alpha={best_mix_alpha:.2f}, bal_acc={best_mix_bal:.6f}")

# ============ Pick best ensemble strategy ============

strategies = {
    "grid_blend": (best_grid_bal, oof_blend, test_blend),
    "ridge_meta": (ridge_bal, ridge_oof_proba, ridge_test_proba),
    "lgb_meta": (lgb_meta_bal, lgb_meta_oof, lgb_meta_test),
}

# Add best mixture
if best_mix_name == "blend+lgb_meta":
    mix_oof = best_mix_alpha * oof_blend + (1 - best_mix_alpha) * lgb_meta_oof
    mix_test = best_mix_alpha * test_blend + (1 - best_mix_alpha) * lgb_meta_test
elif best_mix_name == "blend+ridge_meta":
    mix_oof = best_mix_alpha * oof_blend + (1 - best_mix_alpha) * ridge_oof_proba
    mix_test = best_mix_alpha * test_blend + (1 - best_mix_alpha) * ridge_test_proba
else:
    mix_oof = oof_blend
    mix_test = test_blend
strategies["best_mix"] = (best_mix_bal, mix_oof, mix_test)

best_strat_name = max(strategies, key=lambda k: strategies[k][0])
best_strat_bal, best_oof_proba, best_test_proba = strategies[best_strat_name]
print(f"\nBest strategy: {best_strat_name}, bal_acc={best_strat_bal:.6f}")

# ============ Bias Tuning ============

print("\nBias tuning...")
bias, biased_bal = bias_tune(best_oof_proba, y)
print(f"  Bias: {bias}")
print(f"  After bias: bal_acc={biased_bal:.6f} (before: {best_strat_bal:.6f})")

# Apply bias to test
log_test_proba = np.log(np.clip(best_test_proba, 1e-15, 1.0))
test_biased = log_test_proba + bias

# Also try bias on the simple grid blend
bias_grid, biased_grid_bal = bias_tune(oof_blend, y)
print(f"  Grid blend after bias: bal_acc={biased_grid_bal:.6f}")

# Pick overall best
if biased_grid_bal > biased_bal:
    print("  -> Grid blend + bias is better")
    final_oof_preds = (np.log(np.clip(oof_blend, 1e-15, 1.0)) + bias_grid).argmax(1)
    final_test_preds = (np.log(np.clip(test_blend, 1e-15, 1.0)) + bias_grid).argmax(1)
    final_bal = biased_grid_bal
    final_method = f"grid_blend+bias"
    final_bias = bias_grid.tolist()
elif biased_bal > best_strat_bal:
    print(f"  -> {best_strat_name} + bias is best")
    final_oof_preds = (np.log(np.clip(best_oof_proba, 1e-15, 1.0)) + bias).argmax(1)
    final_test_preds = test_biased.argmax(1)
    final_bal = biased_bal
    final_method = f"{best_strat_name}+bias"
    final_bias = bias.tolist()
else:
    print(f"  -> No bias improvement, using {best_strat_name} raw")
    final_oof_preds = best_oof_proba.argmax(1)
    final_test_preds = best_test_proba.argmax(1)
    final_bal = best_strat_bal
    final_method = best_strat_name
    final_bias = None

final_acc = accuracy_score(y, final_oof_preds)
print(f"\nFinal: method={final_method}, bal_acc={final_bal:.6f}, acc={final_acc:.6f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y, final_oof_preds, target_names=["Low", "Medium", "High"]))

# ============ Save ============

np.save(OUT_DIR / "oof_preds.npy", np.stack([oof_lgbm, oof_xgb, oof_cat]))
np.save(OUT_DIR / "test_preds.npy", np.stack([test_lgbm, test_xgb, test_cat]))

sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
sub["Irrigation_Need"] = [target_inv[p] for p in final_test_preds]
sub.to_csv(OUT_DIR / "submission.csv", index=False)

results = {
    "trial": "trial_007_bias_tuned_stacking",
    "oof_balanced_accuracy": round(final_bal, 6),
    "oof_accuracy": round(final_acc, 6),
    "oof_accuracy_ensemble": round(final_bal, 6),
    "method": final_method,
    "bias": final_bias,
    "grid_weights": best_grid_w,
    "strategy_scores": {k: round(v[0], 6) for k, v in strategies.items()},
    "individual_bal_acc": indiv_scores,
    "n_features_base": len(features),
    "n_pairwise": len(pairwise_cols),
    "n_te_cols_per_fold": len(te_target_cols) * 3,
}
with open(OUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone. Files saved to {OUT_DIR}")
