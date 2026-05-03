"""
Trial 003: Multi-Seed Stacking Ensemble
- 3 seeds x 3 models (LGBM, XGB, CAT) = 9 base learners
- 10-fold CV for stable OOF
- Level-2: LogisticRegression + LightGBM meta-model
- Same FE as trial_002
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
OUT_DIR = Path(__file__).resolve().parent

SEEDS = [42, 123, 777]
N_FOLDS = 10
N_CLASSES = 3

# Load
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

target_map = {"Low": 0, "Medium": 1, "High": 2}
target_inv = {v: k for k, v in target_map.items()}
y = train["Irrigation_Need"].map(target_map).values


def add_features(df):
    df["ET_proxy"] = df["Temperature_C"] * df["Wind_Speed_kmh"] / (df["Humidity"] + 1)
    df["water_balance"] = df["Rainfall_mm"] - df["ET_proxy"] * 100
    df["SM_x_Temp"] = df["Soil_Moisture"] * df["Temperature_C"]
    df["SM_x_Humidity"] = df["Soil_Moisture"] * df["Humidity"]
    df["Temp_x_Humidity"] = df["Temperature_C"] * df["Humidity"]
    df["Rainfall_x_Temp"] = df["Rainfall_mm"] * df["Temperature_C"]
    df["is_active_growth"] = (df["Crop_Growth_Stage"].isin(["Vegetative", "Flowering"])).astype(int)
    df["is_dry_hot"] = ((df["Soil_Moisture"] < 25) & (df["Temperature_C"] > 30)).astype(int)
    df["is_low_rain"] = (df["Rainfall_mm"] < 500).astype(int)
    df["is_mulched"] = (df["Mulching_Used"] == "Yes").astype(int)
    df["rain_per_area"] = df["Rainfall_mm"] / (df["Field_Area_hectare"] + 0.1)
    df["prev_irr_per_area"] = df["Previous_Irrigation_mm"] / (df["Field_Area_hectare"] + 0.1)
    df["moisture_deficit"] = 50 - df["Soil_Moisture"]
    return df


train = add_features(train)
test = add_features(test)

drop_cols = ["id", "Irrigation_Need"]
features = [c for c in train.columns if c not in drop_cols]
cat_cols = ["Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
            "Irrigation_Type", "Water_Source", "Mulching_Used", "Region"]

# Label encode for LGBM/XGB
train_enc = train.copy()
test_enc = test.copy()
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    train_enc[col] = le.fit_transform(train_enc[col])
    test_enc[col] = le.transform(test_enc[col])
    label_encoders[col] = le

X_lgbm = train_enc[features]
X_test_lgbm = test_enc[features]

# CatBoost: keep strings
X_cat = train[features].copy()
X_test_cat = test[features].copy()
cat_indices = [features.index(c) for c in cat_cols]

# Top-5 important features for meta-model
meta_raw_features = ["Soil_Moisture", "Temperature_C", "Humidity", "Rainfall_mm", "Previous_Irrigation_mm"]

# ============ Base Model Params ============

def get_lgbm_params(seed):
    return {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "verbosity": -1,
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "num_leaves": 127,
        "min_child_samples": 30,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
    }

def get_xgb_params(seed):
    return {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "verbosity": 0,
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "max_depth": 8,
        "min_child_weight": 30,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "tree_method": "hist",
        "enable_categorical": True,
    }

def get_cat_params(seed):
    return {
        "iterations": 2000,
        "learning_rate": 0.03,
        "depth": 8,
        "l2_leaf_reg": 3,
        "random_seed": seed,
        "verbose": 0,
        "cat_features": cat_indices,
        "auto_class_weights": "Balanced",
    }

# ============ Level-1: Multi-Seed Base Models ============

n_train = len(X_lgbm)
n_test = len(X_test_lgbm)

# Store OOF and test preds per (model_type, seed)
# Shape: {key: (oof_proba, test_proba)}
all_oof = {}
all_test = {}

model_types = ["lgbm", "xgb", "cat"]

for seed in SEEDS:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    oof_lgbm = np.zeros((n_train, N_CLASSES))
    oof_xgb = np.zeros((n_train, N_CLASSES))
    oof_cat = np.zeros((n_train, N_CLASSES))
    test_lgbm = np.zeros((n_test, N_CLASSES))
    test_xgb = np.zeros((n_test, N_CLASSES))
    test_cat = np.zeros((n_test, N_CLASSES))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_lgbm, y)):
        print(f"\n{'='*30} Seed {seed} | Fold {fold} {'='*30}")
        y_tr, y_val = y[train_idx], y[val_idx]

        # LightGBM
        model = lgb.LGBMClassifier(**get_lgbm_params(seed))
        model.fit(
            X_lgbm.iloc[train_idx], y_tr,
            eval_set=[(X_lgbm.iloc[val_idx], y_val)],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)],
        )
        oof_lgbm[val_idx] = model.predict_proba(X_lgbm.iloc[val_idx])
        test_lgbm += model.predict_proba(X_test_lgbm) / N_FOLDS

        # XGBoost
        model = xgb.XGBClassifier(**get_xgb_params(seed))
        model.fit(
            X_lgbm.iloc[train_idx], y_tr,
            eval_set=[(X_lgbm.iloc[val_idx], y_val)],
            verbose=0,
        )
        oof_xgb[val_idx] = model.predict_proba(X_lgbm.iloc[val_idx])
        test_xgb += model.predict_proba(X_test_lgbm) / N_FOLDS

        # CatBoost
        model = CatBoostClassifier(**get_cat_params(seed))
        model.fit(
            X_cat.iloc[train_idx], y_tr,
            eval_set=(X_cat.iloc[val_idx], y_val),
            early_stopping_rounds=100,
        )
        oof_cat[val_idx] = model.predict_proba(X_cat.iloc[val_idx])
        test_cat += model.predict_proba(X_test_cat) / N_FOLDS

    # Print per-seed scores
    for name, oof in [("LGBM", oof_lgbm), ("XGB", oof_xgb), ("CAT", oof_cat)]:
        acc = accuracy_score(y, oof.argmax(1))
        print(f"  Seed {seed} {name} OOF acc: {acc:.6f}")

    all_oof[f"lgbm_{seed}"] = oof_lgbm
    all_oof[f"xgb_{seed}"] = oof_xgb
    all_oof[f"cat_{seed}"] = oof_cat
    all_test[f"lgbm_{seed}"] = test_lgbm
    all_test[f"xgb_{seed}"] = test_xgb
    all_test[f"cat_{seed}"] = test_cat

# ============ Average per model type ============

avg_oof = {}
avg_test = {}
for mt in model_types:
    keys = [f"{mt}_{s}" for s in SEEDS]
    avg_oof[mt] = np.mean([all_oof[k] for k in keys], axis=0)
    avg_test[mt] = np.mean([all_test[k] for k in keys], axis=0)
    acc = accuracy_score(y, avg_oof[mt].argmax(1))
    print(f"\n{mt.upper()} multi-seed avg OOF acc: {acc:.6f}")

# Simple weighted ensemble (like trial_002 but with multi-seed avg)
best_acc_simple = 0
best_w_simple = (1, 1, 1)
for w1 in range(1, 6):
    for w2 in range(1, 6):
        for w3 in range(1, 6):
            total = w1 + w2 + w3
            oof_ens = (w1 * avg_oof["lgbm"] + w2 * avg_oof["xgb"] + w3 * avg_oof["cat"]) / total
            acc = accuracy_score(y, oof_ens.argmax(1))
            if acc > best_acc_simple:
                best_acc_simple = acc
                best_w_simple = (w1, w2, w3)

print(f"\nSimple multi-seed ensemble: weights={best_w_simple}, OOF acc={best_acc_simple:.6f}")

# ============ Level-2: Stacking ============

# Build meta-features: 9 base models x 3 classes = 27 OOF features + 5 raw features
print("\n\n========== STACKING LEVEL 2 ==========")

meta_keys = sorted(all_oof.keys())  # consistent ordering
meta_train = np.hstack([all_oof[k] for k in meta_keys])  # (n_train, 27)
meta_test = np.hstack([all_test[k] for k in meta_keys])  # (n_test, 27)

# Add top-5 raw features
raw_train = train_enc[meta_raw_features].values
raw_test = test_enc[meta_raw_features].values
meta_train = np.hstack([meta_train, raw_train])  # (n_train, 32)
meta_test = np.hstack([meta_test, raw_test])  # (n_test, 32)

print(f"Meta features shape: {meta_train.shape}")

# Meta-model 1: LogisticRegression
skf_meta = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
oof_meta_lr = np.zeros((n_train, N_CLASSES))
test_meta_lr = np.zeros((n_test, N_CLASSES))

for fold, (train_idx, val_idx) in enumerate(skf_meta.split(meta_train, y)):
    lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", multi_class="multinomial")
    lr.fit(meta_train[train_idx], y[train_idx])
    oof_meta_lr[val_idx] = lr.predict_proba(meta_train[val_idx])
    test_meta_lr += lr.predict_proba(meta_test) / N_FOLDS

acc_lr = accuracy_score(y, oof_meta_lr.argmax(1))
print(f"Meta LR OOF acc: {acc_lr:.6f}")

# Meta-model 2: LightGBM (light)
oof_meta_lgbm = np.zeros((n_train, N_CLASSES))
test_meta_lgbm = np.zeros((n_test, N_CLASSES))

meta_lgbm_params = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "verbosity": -1,
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 50,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "random_state": 42,
}

for fold, (train_idx, val_idx) in enumerate(skf_meta.split(meta_train, y)):
    model = lgb.LGBMClassifier(**meta_lgbm_params)
    model.fit(
        meta_train[train_idx], y[train_idx],
        eval_set=[(meta_train[val_idx], y[val_idx])],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(500)],
    )
    oof_meta_lgbm[val_idx] = model.predict_proba(meta_train[val_idx])
    test_meta_lgbm += model.predict_proba(meta_test) / N_FOLDS

acc_lgbm_meta = accuracy_score(y, oof_meta_lgbm.argmax(1))
print(f"Meta LGBM OOF acc: {acc_lgbm_meta:.6f}")

# Final blend: 50/50 LR + LGBM meta
oof_stack = 0.5 * oof_meta_lr + 0.5 * oof_meta_lgbm
test_stack = 0.5 * test_meta_lr + 0.5 * test_meta_lgbm
acc_stack = accuracy_score(y, oof_stack.argmax(1))
print(f"\nStacking (LR+LGBM 50/50) OOF acc: {acc_stack:.6f}")

# ============ Choose Best ============

# Compare simple multi-seed ensemble vs stacking
if acc_stack > best_acc_simple:
    final_oof = oof_stack
    final_test = test_stack
    final_acc = acc_stack
    final_method = "stacking"
    print(f"\n>>> Using STACKING: {acc_stack:.6f} > simple {best_acc_simple:.6f}")
else:
    total = sum(best_w_simple)
    final_oof = (best_w_simple[0] * avg_oof["lgbm"] + best_w_simple[1] * avg_oof["xgb"] + best_w_simple[2] * avg_oof["cat"]) / total
    final_test = (best_w_simple[0] * avg_test["lgbm"] + best_w_simple[1] * avg_test["xgb"] + best_w_simple[2] * avg_test["cat"]) / total
    final_acc = best_acc_simple
    final_method = "simple_multiseed"
    print(f"\n>>> Using SIMPLE MULTI-SEED: {best_acc_simple:.6f} >= stacking {acc_stack:.6f}")

# ============ Save ============

np.save(OUT_DIR / "oof_preds.npy", final_oof)
np.save(OUT_DIR / "test_preds.npy", final_test)

sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
sub["Irrigation_Need"] = [target_inv[p] for p in final_test.argmax(1)]
sub.to_csv(OUT_DIR / "submission.csv", index=False)

results = {
    "trial": "trial_003_multiseed_stacking",
    "method": final_method,
    "n_folds": N_FOLDS,
    "seeds": SEEDS,
    "oof_accuracy_lgbm_avg": round(accuracy_score(y, avg_oof["lgbm"].argmax(1)), 6),
    "oof_accuracy_xgb_avg": round(accuracy_score(y, avg_oof["xgb"].argmax(1)), 6),
    "oof_accuracy_cat_avg": round(accuracy_score(y, avg_oof["cat"].argmax(1)), 6),
    "oof_accuracy_simple_ensemble": round(best_acc_simple, 6),
    "simple_ensemble_weights": best_w_simple,
    "oof_accuracy_meta_lr": round(acc_lr, 6),
    "oof_accuracy_meta_lgbm": round(acc_lgbm_meta, 6),
    "oof_accuracy_stacking": round(acc_stack, 6),
    "oof_accuracy_ensemble": round(final_acc, 6),
}
with open(OUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nFinal OOF accuracy: {final_acc:.6f}")
print("Done. Files saved to", OUT_DIR)
