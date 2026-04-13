"""
Trial 017: RealMLP for Kaggle GPU Notebook
- RealMLP_TD_Classifier with periodic embeddings
- Same FE as trial_016 (digit, magic score, logit features)
- 3-seed x 5-fold = 15 models
- Goal: diverse predictions for ensemble with XGB (trial_016)
- Run on Kaggle GPU (T4/P100)

Usage: Upload to Kaggle notebook with GPU enabled.
       Attach competition data as input.
"""

import gc
import json
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, TargetEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
import sys

# ============ Path Setup (Kaggle vs Local) ============

KAGGLE = Path("/kaggle/input").exists()
if KAGGLE:
    DATA_DIR = Path("/kaggle/input/playground-series-s6e4")
    ORIG_DIR = Path("/kaggle/input/irrigation-prediction")  # attach original dataset
    OUT_DIR = Path("/kaggle/working")
    print("Running on Kaggle")
else:
    DATA_DIR = Path(__file__).resolve().parents[3] / "data"
    ORIG_DIR = DATA_DIR / "original"
    OUT_DIR = Path(__file__).resolve().parent
    print("Running locally")

# ============ Install RealMLP ============

if KAGGLE:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytabkit"], check=True)

from pytabkit.models.sklearn.sklearn_interfaces import RealMLP_TD_Classifier

# ============ Load ============

train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

if KAGGLE:
    orig = pd.read_csv(ORIG_DIR / "irrigation_prediction.csv")
else:
    orig = pd.read_csv(ORIG_DIR / "irrigation_prediction.csv")

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

# ============ Magic Score ============

for df in [train, test, orig]:
    df["is_harvest"] = (df["Crop_Growth_Stage"] == "Harvest").astype(int)
    df["is_sowing"] = (df["Crop_Growth_Stage"] == "Sowing").astype(int)
    df["mulching_yes"] = (df["Mulching_Used"] == "Yes").astype(int)
    high_score = 2 * df["soil_lt_25"] + 2 * df["rain_lt_300"] + df["temp_gt_30"] + df["wind_gt_10"]
    low_score = 2 * df["is_harvest"] + 2 * df["is_sowing"] + df["mulching_yes"]
    df["magic_score"] = high_score - low_score

magic_cols = ["magic_score", "is_harvest", "is_sowing", "mulching_yes"]

# ============ Deotte Logit Features ============

deotte_features = ["soil_lt_25", "temp_gt_30", "rain_lt_300", "wind_gt_10"]
for stage in orig["Crop_Growth_Stage"].unique():
    col = f"stage_{stage}"
    for df in [train, test, orig]:
        df[col] = (df["Crop_Growth_Stage"] == stage).astype(int)

for mul in orig["Mulching_Used"].unique():
    col = f"mulch_{mul}"
    for df in [train, test, orig]:
        df[col] = (df["Mulching_Used"] == mul).astype(int)

stage_dummies = [f"stage_{s}" for s in orig["Crop_Growth_Stage"].unique()]
mulch_dummies = [f"mulch_{m}" for m in orig["Mulching_Used"].unique()]
lr_features = deotte_features + stage_dummies[:-1] + mulch_dummies[:-1]

lr_model = LogisticRegression(max_iter=1000, C=1e6, solver="lbfgs")
lr_model.fit(orig[lr_features].values, orig_y)

logit_cols = []
for df in [train, test, orig]:
    logits = lr_model.decision_function(df[lr_features].values)
    for ci in range(3):
        col = f"deotte_logit_{ci}"
        df[col] = logits[:, ci]
logit_cols = [f"deotte_logit_{ci}" for ci in range(3)]

print(f"LR accuracy on orig: {lr_model.score(orig[lr_features].values, orig_y):.4f}")

# ============ Digit Features ============

digit_cols = []
for col in num_cols:
    for k in range(-2, 3):
        digit_col = f"{col}_digit_{k}"
        for df in [train, test, orig]:
            df[digit_col] = ((df[col].abs() * (10 ** (-k))).astype(np.int64) % 10).astype(np.int8)
        digit_cols.append(digit_col)

print(f"Digit features: {len(digit_cols)}")

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

# ============ Features for RealMLP (no pairwise — too many for NN) ============
# Use: num_cols + le_cols + binary + te_orig + magic + logit + digit

mlp_features = num_cols + le_cols + binary_cols + te_orig_cols + magic_cols + logit_cols + digit_cols
print(f"MLP features: {len(mlp_features)}")

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

SEEDS = [42, 123, 456]
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
        print(f"\n--- Seed {seed} Fold {fold} ---")
        y_tr = y[train_idx]
        y_val = y[val_idx]

        # Prepare data
        X_tr = train[mlp_features].iloc[train_idx].values.astype(np.float32)
        X_val = train[mlp_features].iloc[val_idx].values.astype(np.float32)
        X_te = test[mlp_features].values.astype(np.float32)

        # Append original data
        X_orig_mlp = orig[mlp_features].values.astype(np.float32)
        X_tr_combined = np.vstack([X_tr, X_orig_mlp])
        y_combined = np.concatenate([y_tr, orig_y])

        # Sample weights
        sw_synth = compute_sample_weight("balanced", y_tr)
        sw_orig = np.full(len(orig), 0.35)
        sw_combined = np.concatenate([sw_synth, sw_orig])

        # RealMLP
        model = RealMLP_TD_Classifier(
            n_epochs=256,
            batch_size=256,
            lr=0.04,
            hidden_sizes=[512, 64, 128],
            use_ls=True,
            ls_eps=0.04,
            n_ensemble=1,  # we do our own seed averaging
            random_state=seed + fold,
            verbosity=1,
        )

        model.fit(X_tr_combined, y_combined, sample_weight=sw_combined)

        oof[val_idx] = model.predict_proba(X_val)
        test_preds += model.predict_proba(X_te) / N_FOLDS

        fold_bal = balanced_accuracy_score(y_val, oof[val_idx].argmax(1))
        print(f"  RealMLP fold {fold}: bal_acc={fold_bal:.6f}")
        del model; gc.collect()

    oof_seeds[si] = oof
    test_seeds[si] = test_preds

    seed_bal = balanced_accuracy_score(y, oof.argmax(1))
    print(f"\nSeed {seed} OOF: {seed_bal:.6f}")

# ============ Average ============

oof_avg = oof_seeds.mean(0)
test_avg = test_seeds.mean(0)

avg_bal = balanced_accuracy_score(y, oof_avg.argmax(1))
print(f"\nSeed-averaged OOF: {avg_bal:.6f}")

# ============ Bias Tuning ============

best_bias, best_bias_score = tune_bias(oof_avg, y)
print(f"Bias: {best_bias.tolist()} -> {best_bias_score:.6f}")

test_logits = np.log(test_avg + 1e-15) + best_bias
test_final_preds = test_logits.argmax(1)

# ============ Save ============

np.save(OUT_DIR / "oof_preds_realmlp.npy", oof_avg)
np.save(OUT_DIR / "test_preds_realmlp.npy", test_avg)

sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
sub["Irrigation_Need"] = [target_inv[p] for p in test_final_preds]
sub.to_csv(OUT_DIR / "submission.csv", index=False)

print(f"\nDistribution: {pd.Series([target_inv[p] for p in test_final_preds]).value_counts().to_dict()}")

results = {
    "trial": "trial_017_realmlp",
    "metric": "balanced_accuracy",
    "seeds": SEEDS,
    "n_folds": N_FOLDS,
    "n_total_models": len(SEEDS) * N_FOLDS,
    "model": "RealMLP_TD_Classifier",
    "hidden_sizes": [512, 64, 128],
    "n_epochs": 256,
    "lr": 0.04,
    "label_smoothing": 0.04,
    "oof_raw": round(avg_bal, 6),
    "bias": [round(float(b), 4) for b in best_bias],
    "oof_with_bias": round(best_bias_score, 6),
    "n_features": len(mlp_features),
}
with open(OUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone. OOF: {best_bias_score:.6f}")
