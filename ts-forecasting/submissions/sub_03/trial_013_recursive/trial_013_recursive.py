"""Trial 013: recursive prediction (vectorized) — lag features를 step-by-step으로 채움"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
import numpy as np
import pandas as pd
from utils import *

LAG_COLS  = [1, 2, 3, 5, 10, 20]
ROLL_WINS = [5, 10, 20]
MAX_HIST  = 20


def add_target_encoding(combined):
    train_mask = combined['weight'].notna()
    stats = combined[train_mask].groupby(KEY)['y_target'].agg(
        series_mean='mean', series_std='std', series_median='median'
    ).reset_index()
    return combined.merge(stats, on=KEY, how='left')


def build_history(df, key_cols):
    """시리즈별 최근 MAX_HIST y_target → dict {key_tuple: [oldest..newest]}"""
    hist = {}
    for key, grp in df.groupby(key_cols, sort=False):
        vals = grp.sort_values('ts_index')['y_target'].values
        hist[key] = list(vals[-MAX_HIST:])
    return hist


def compute_lag_features_vectorized(df, hist, key_cols):
    keys = [tuple(row) for row in df[key_cols].values]

    for lag in LAG_COLS:
        df[f'lag_{lag}'] = [
            hist[k][-lag] if k in hist and len(hist[k]) >= lag else np.nan
            for k in keys
        ]

    for w in ROLL_WINS:
        means, stds = [], []
        for k in keys:
            window = hist[k][-w:] if k in hist else []
            means.append(np.mean(window) if window else np.nan)
            stds.append(np.std(window) if len(window) > 1 else np.nan)
        df[f'roll_mean_{w}'] = means
        df[f'roll_std_{w}']  = stds

    l1  = df['lag_1']
    l5  = df['lag_5']
    l10 = df['lag_10']
    df['trend_1_5']  = (l1 - l5)  / 5.0
    df['trend_1_10'] = (l1 - l10) / 10.0
    return df


def recursive_predict(model, df, hist_init, key_cols):
    hist = {k: list(v) for k, v in hist_init.items()}
    df_sorted = df.sort_values('ts_index').reset_index(drop=True)
    all_preds = {}

    for ts_idx, grp in df_sorted.groupby('ts_index', sort=True):
        grp = grp.copy()
        grp = compute_lag_features_vectorized(grp, hist, key_cols)
        preds = model.predict(prepare_X(grp))

        for i, (_, row) in enumerate(grp.iterrows()):
            k = tuple(row[c] for c in key_cols)
            all_preds[row['id']] = preds[i]
            if k not in hist:
                hist[k] = []
            hist[k].append(preds[i])
            if len(hist[k]) > MAX_HIST:
                hist[k].pop(0)

        if ts_idx % 200 == 0:
            print(f"  ts_index {ts_idx}...", flush=True)

    df['prediction'] = df['id'].map(all_preds)
    return df


def main():
    train, test = load_data()
    combined = combine_train_test(train, test)

    print("Engineering features...")
    combined = add_base_lags(combined, lags=LAG_COLS)
    combined = add_rolling(combined, windows=ROLL_WINS)
    combined = add_trend(combined)
    combined = add_target_encoding(combined)

    train_f = combined[combined['weight'].notna()].copy()
    test_f  = combined[combined['weight'].isna()].copy()

    tr  = train_f[train_f.ts_index <= CUTOFF]
    val = train_f[train_f.ts_index >  CUTOFF]
    print(f"Train: {len(tr)}, Val: {len(val)}")

    model = train_lgbm(prepare_X(tr), tr.y_target.values, tr.weight.values,
                       prepare_X(val), val.y_target.values, val.weight.values)

    score_naive = weighted_rmse_score(val.y_target.values, model.predict(prepare_X(val)), val.weight.values)
    print(f"\n[Trial 013] Val score (naive): {score_naive:.6f}")

    print("\nRecursive val prediction...")
    hist_val = build_history(tr, KEY)
    val_result = recursive_predict(model, val.copy(), hist_val, KEY)
    score_rec = weighted_rmse_score(
        val.y_target.values,
        val_result.set_index('id').loc[val['id'], 'prediction'].values,
        val.weight.values,
    )
    print(f"[Trial 013] Val score (recursive): {score_rec:.6f}")

    print("\nFull retrain...")
    model_full = retrain_full(prepare_X(train_f), train_f.y_target.values,
                              train_f.weight.values, model.best_iteration)

    print("Recursive test prediction...")
    hist_test = build_history(train_f, KEY)
    test_result = recursive_predict(model_full, test_f.copy(), hist_test, KEY)

    print(f"Prediction NaN: {test_result['prediction'].isna().sum()}")
    print(test_result['prediction'].describe())

    out = Path(__file__).parent / 'trial_013_recursive.csv'
    pd.DataFrame({'id': test_result['id'], 'prediction': test_result['prediction']}).to_csv(out, index=False)
    print(f"Saved → {out}")


if __name__ == '__main__':
    main()
