"""
Trial 020: OrderedTE + Top Params + Digit Features as Categorical
- OrderedTE with 4-shuffle augmentation (yunsuxiaozi method)
- Digit features treated as categorical + OrderedTE
- XGB: max_depth=4, lr=0.1, alpha=5, lambda=5 (top notebook params)
- Freq encoding on all categoricals
- Orig data TE (mean + std)
- 5-seed averaging
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

# ============ Digit Features (as categorical) ============

digit_cols = []
for col in num_cols:
    for k in range(-2, 3):
        digit_col = f"{col}_digit_{k}"
        for df in [train, test, orig]:
            df[digit_col] = ((df[col].abs() * (10 ** (-k))).astype(np.int64) % 10).astype(np.int8)
        digit_cols.append(digit_col)

print(f"Digit features: {len(digit_cols)}")

# ============ Pairwise factorize (cat x cat only, 28 pairs) ============

pair_cols = []
for c1, c2 in combinations(cat_cols, 2):
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

# Also add cat x binary and cat x digit key pairs
for c1 in cat_cols:
    for c2 in binary_cols:
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

# ============ Frequency encoding ============

freq_cols = []
all_cat_features = cat_cols + pair_cols + digit_cols + binary_cols
for col in all_cat_features:
    freq = pd.concat([train[col], test[col], orig[col]]).value_counts(normalize=True)
    col_name = f"freq_{col}"
    train[col_name] = train[col].map(freq).fillna(0).astype(np.float32)
    test[col_name] = test[col].map(freq).fillna(0).astype(np.float32)
    orig[col_name] = orig[col].map(freq).fillna(0).astype(np.float32)
    freq_cols.append(col_name)

print(f"Frequency features: {len(freq_cols)}")

# ============ Original Data TE (mean + std) ============

orig["target_num"] = orig["Irrigation_Need"].map(target_map)
te_orig_cols = []
for col in all_raw_cols:
    te_map_mean = orig.groupby(col)["target_num"].mean()
    te_map_std = orig.groupby(col)["target_num"].std().fillna(0)
    col_mean = f"{col}_te_orig_mean"
    col_std = f"{col}_te_orig_std"
    train[col_mean] = train[col].map(te_map_mean).fillna(te_map_mean.mean())
    test[col_mean] = test[col].map(te_map_mean).fillna(te_map_mean.mean())
    orig[col_mean] = orig[col].map(te_map_mean).fillna(te_map_mean.mean())
    train[col_std] = train[col].map(te_map_std).fillna(0)
    test[col_std] = test[col].map(te_map_std).fillna(0)
    orig[col_std] = orig[col].map(te_map_std).fillna(0)
    te_orig_cols.extend([col_mean, col_std])

print(f"Orig TE features (mean+std): {len(te_orig_cols)}")

# ============ Ordered Target Encoding ============

class OrderedTE:
    """Cumulative leave-one-out TE with Bayesian smoothing, multiclass."""
    def __init__(self, a=10, n_shuffles=4):
        self.a = a
        self.n_shuffles = n_shuffles
        self.global_priors = None
        self.n_classes = None

    def fit_transform_train(self, X_col, y, seed=42):
        """Returns (n_samples, n_classes) array for training data with shuffle augmentation."""
        self.n_classes = len(np.unique(y))
        self.global_priors = np.array([(y == c).mean() for c in range(self.n_classes)])

        all_te = np.zeros((len(X_col), self.n_classes))
        for shuffle_i in range(self.n_shuffles):
            rng = np.random.RandomState(seed + shuffle_i)
            idx = rng.permutation(len(X_col))
            te_shuffle = self._ordered_encode(X_col[idx], y[idx])
            # Un-shuffle back to original order
            te_unshuffle = np.zeros_like(te_shuffle)
            te_unshuffle[idx] = te_shuffle
            all_te += te_unshuffle
        return all_te / self.n_shuffles

    def _ordered_encode(self, X_col, y):
        """Cumulative TE: each row uses only preceding rows in the same category."""
        n = len(X_col)
        te = np.zeros((n, self.n_classes), dtype=np.float32)

        # Group by category value, process each group with vectorized cumsum
        df = pd.DataFrame({"val": X_col, "y": y, "orig_idx": np.arange(n)})
        for c in range(self.n_classes):
            df[f"is_c{c}"] = (df["y"] == c).astype(np.float32)

        for val, grp in df.groupby("val"):
            idx = grp["orig_idx"].values
            total = len(grp)
            cum_total = np.arange(total, dtype=np.float32)  # 0, 1, 2, ...
            for c in range(self.n_classes):
                cum_count = grp[f"is_c{c}"].values.cumsum()
                # Shift: use stats BEFORE current row
                cum_count_before = np.concatenate([[0], cum_count[:-1]])
                te[idx, c] = (cum_count_before + self.a * self.global_priors[c]) / (cum_total + self.a)

        return te

    def transform(self, X_col, X_train_col, y_train):
        """Transform test/val data using full training data stats."""
        n = len(X_col)
        te = np.zeros((n, self.n_classes), dtype=np.float32)

        # Compute full stats from training data
        df_tr = pd.DataFrame({"val": X_train_col, "y": y_train})
        for c in range(self.n_classes):
            df_tr[f"is_c{c}"] = (df_tr["y"] == c).astype(np.float32)

        stats = df_tr.groupby("val").agg(
            total=("y", "count"),
            **{f"cnt_c{c}": (f"is_c{c}", "sum") for c in range(self.n_classes)}
        )

        for c in range(self.n_classes):
            mapping = (stats[f"cnt_c{c}"] + self.a * self.global_priors[c]) / (stats["total"] + self.a)
            te[:, c] = pd.Series(X_col).map(mapping).fillna(self.global_priors[c]).values

        return te

# ============ Base features (without TE — TE computed per fold) ============

base_features = num_cols + le_cols + binary_cols + te_orig_cols + pair_cols + digit_cols + freq_cols
print(f"Base features: {len(base_features)}")

# Columns to apply OrderedTE on
ote_source_cols = cat_cols + pair_cols + digit_cols + binary_cols
print(f"OrderedTE source columns: {len(ote_source_cols)}")

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
        print(f"\n--- Seed {seed} Fold {fold} ---")
        y_tr = y[train_idx]
        y_val = y[val_idx]

        # Combine train fold + orig
        y_combined = np.concatenate([y_tr, orig_y])

        # ---- OrderedTE on all categorical-like columns ----
        ote = OrderedTE(a=10, n_shuffles=4)
        ote_names = []

        # Build combined arrays for TE
        ote_train_features = []
        ote_val_features = []
        ote_test_features = []
        ote_orig_features = []

        for col in ote_source_cols:
            tr_vals = train[col].iloc[train_idx].values
            val_vals = train[col].iloc[val_idx].values
            te_vals = test[col].values
            orig_vals = orig[col].values

            # Combined: train fold + orig
            combined_vals = np.concatenate([tr_vals, orig_vals])

            # Fit on combined
            ote_enc = OrderedTE(a=10, n_shuffles=4)
            tr_te = ote_enc.fit_transform_train(combined_vals, y_combined, seed=seed+fold)

            # Split back
            tr_te_synth = tr_te[:len(train_idx)]
            tr_te_orig = tr_te[len(train_idx):]

            # Transform val and test
            val_te = ote_enc.transform(val_vals, combined_vals, y_combined)
            test_te = ote_enc.transform(te_vals, combined_vals, y_combined)

            for c in range(3):
                col_name = f"ote_{col}_c{c}"
                ote_names.append(col_name)

            ote_train_features.append(tr_te_synth)
            ote_val_features.append(val_te)
            ote_test_features.append(test_te)
            ote_orig_features.append(tr_te_orig)

        ote_train_arr = np.hstack(ote_train_features)
        ote_val_arr = np.hstack(ote_val_features)
        ote_test_arr = np.hstack(ote_test_features)
        ote_orig_arr = np.hstack(ote_orig_features)

        if fold == 0 and si == 0:
            print(f"  OrderedTE features: {ote_train_arr.shape[1]}")

        # ---- Build full feature matrices ----
        X_tr_base = train[base_features].iloc[train_idx].values
        X_val_base = train[base_features].iloc[val_idx].values
        X_te_base = test[base_features].values
        X_orig_base = orig[base_features].values

        X_tr_np = np.hstack([X_tr_base, ote_train_arr])
        X_orig_np = np.hstack([X_orig_base, ote_orig_arr])
        X_tr_full = np.vstack([X_tr_np, X_orig_np])

        X_val_np = np.hstack([X_val_base, ote_val_arr])
        X_te_np = np.hstack([X_te_base, ote_test_arr])

        if fold == 0 and si == 0:
            print(f"  Total features: {X_tr_full.shape[1]}")
            print(f"  X_tr shape (with orig): {X_tr_full.shape}")

        # Sample weights
        sw_synth = compute_sample_weight("balanced", y_tr)
        sw_orig = np.full(N_ORIG, 0.35)
        sw_combined = np.concatenate([sw_synth, sw_orig])

        # ---- XGB with top params ----
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
            X_tr_full, y_combined,
            sample_weight=sw_combined,
            eval_set=[(X_val_np, y_val)],
            verbose=500,
        )

        best_rounds = model.best_iteration
        oof[val_idx] = model.predict_proba(X_val_np)
        test_preds += model.predict_proba(X_te_np) / N_FOLDS

        fold_bal = balanced_accuracy_score(y_val, oof[val_idx].argmax(1))
        print(f"  XGB fold {fold}: bal_acc={fold_bal:.6f}, rounds={best_rounds}")
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
    "trial": "trial_020_ordered_te",
    "metric": "balanced_accuracy",
    "seeds": SEEDS,
    "n_folds": N_FOLDS,
    "n_total_models": len(SEEDS) * N_FOLDS,
    "features": {
        "base": len(base_features),
        "ordered_te_sources": len(ote_source_cols),
        "ordered_te_cols": len(ote_source_cols) * 3,
    },
    "oof_raw": round(avg_bal, 6),
    "bias": [round(float(b), 4) for b in best_bias],
    "oof_with_bias": round(best_bias_score, 6),
}
with open(OUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone. OOF: {best_bias_score:.6f}")
