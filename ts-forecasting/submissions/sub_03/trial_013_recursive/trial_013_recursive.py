"""Trial 013: recursive prediction (vectorized) — lag features를 step-by-step으로 채움"""
import gc
import sys
import psutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
import numpy as np
import pandas as pd
from utils import *

LAG_COLS  = [1, 2, 3, 5, 10, 20]
ROLL_WINS = [5, 10, 20]
MAX_HIST  = 20
MEM_LIMIT_GB = 1.0  # 남은 RAM이 이 이하면 종료


def check_memory(label=''):
    avail_gb = psutil.virtual_memory().available / 1024**3
    if label:
        print(f"[MEM] {label}: {avail_gb:.1f} GB free", flush=True)
    if avail_gb < MEM_LIMIT_GB:
        print(f"[MEM] 위험 ({avail_gb:.1f} GB) — 강제 종료", flush=True)
        sys.exit(1)
    return avail_gb


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
            avail = psutil.virtual_memory().available / 1024**3
            print(f"  ts_index {ts_idx}... ({avail:.1f} GB free)", flush=True)
            if avail < MEM_LIMIT_GB:
                print(f"[MEM] 위험 — 강제 종료", flush=True)
                sys.exit(1)

    df['prediction'] = df['id'].map(all_preds)
    return df


def main():
    train, test = load_data()
    combined = combine_train_test(train, test)
    del train, test
    gc.collect()

    print("Engineering features...")
    combined = add_base_lags(combined, lags=LAG_COLS)
    combined = add_rolling(combined, windows=ROLL_WINS)
    combined = add_trend(combined)
    combined = add_target_encoding(combined)

    train_f = combined[combined['weight'].notna()].copy()
    test_f  = combined[combined['weight'].isna()].copy()
    del combined
    gc.collect()
    check_memory('after feature engineering')

    tr  = train_f[train_f.ts_index <= CUTOFF]
    val = train_f[train_f.ts_index >  CUTOFF]
    print(f"Train: {len(tr)}, Val: {len(val)}")

    X_tr, X_val = prepare_X(tr), prepare_X(val)
    model = train_lgbm(X_tr, tr.y_target.values, tr.weight.values,
                       X_val, val.y_target.values, val.weight.values)

    score_naive = weighted_rmse_score(val.y_target.values, model.predict(X_val), val.weight.values)
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

    # Free val/tr before retrain
    best_iter = model.best_iteration
    del tr, val, X_tr, X_val, model, val_result, hist_val
    gc.collect()
    check_memory('after val, before retrain')

    # Build test history before freeing train_f
    print("\nBuilding test history...")
    hist_test = build_history(train_f, KEY)

    print("Full retrain...")
    y_full = train_f.y_target.values
    w_full = train_f.weight.values
    X_full = prepare_X(train_f)
    del train_f
    gc.collect()
    model_full = retrain_full(X_full, y_full, w_full, best_iter)
    del X_full, y_full, w_full
    gc.collect()
    check_memory('after retrain')

    print("Recursive test prediction...")
    test_result = recursive_predict(model_full, test_f, hist_test, KEY)

    print(f"Prediction NaN: {test_result['prediction'].isna().sum()}")
    print(test_result['prediction'].describe())

    out = Path(__file__).parent / 'trial_013_recursive.csv'
    pd.DataFrame({'id': test_result['id'], 'prediction': test_result['prediction']}).to_csv(out, index=False)
    print(f"Saved → {out}")


if __name__ == '__main__':
    main()
