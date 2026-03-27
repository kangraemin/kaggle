"""Trial 016: trial_015 개선
- sub_code, sub_code×horizon 레벨 encoding 추가 (sub_code가 top feature)
- feature_w/x/y/z 제외 (test 38% NaN)
- num_leaves=127, lr=0.03 (regularization)
- cold-start val (시리즈 20% holdout)
"""
import gc
import sys
import psutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
import numpy as np
import pandas as pd
from utils import *

MEM_LIMIT_GB = 2.0
# test에서 38% NaN인 feature 제외
EXCLUDE_FEATURES = {'feature_w', 'feature_x', 'feature_y', 'feature_z'}

PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 127,
    'learning_rate': 0.03,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'verbose': -1,
    'n_jobs': -1,
}


def check_memory(label=''):
    avail = psutil.virtual_memory().available / 1024**3
    print(f"[MEM] {label}: {avail:.1f} GB free", flush=True)
    if avail < MEM_LIMIT_GB:
        print(f"[MEM] 위험 — 강제 종료", flush=True)
        sys.exit(1)


def compute_group_stats(df):
    s_subcode_h = df.groupby(['sub_code', 'horizon'])['y_target'].agg(
        subcode_h_mean='mean', subcode_h_std='std'
    ).reset_index()
    s_subcode = df.groupby(['sub_code'])['y_target'].agg(
        subcode_mean='mean', subcode_std='std'
    ).reset_index()
    s_subcat_h = df.groupby(['sub_category', 'horizon'])['y_target'].agg(
        subcat_mean='mean', subcat_std='std'
    ).reset_index()
    s_horizon = df.groupby(['horizon'])['y_target'].agg(
        horizon_mean='mean', horizon_std='std'
    ).reset_index()
    return s_subcode_h, s_subcode, s_subcat_h, s_horizon


def apply_group_stats(df, s_subcode_h, s_subcode, s_subcat_h, s_horizon):
    df = df.merge(s_subcode_h, on=['sub_code', 'horizon'], how='left')
    df = df.merge(s_subcode,   on=['sub_code'],             how='left')
    df = df.merge(s_subcat_h,  on=['sub_category', 'horizon'], how='left')
    df = df.merge(s_horizon,   on=['horizon'],               how='left')
    return df


def get_feature_cols(df):
    raw = [c for c in df.columns if c.startswith('feature_') and c not in EXCLUDE_FEATURES]
    enc = [c for c in df.columns if c.endswith('_mean') or c.endswith('_std')]
    return raw + enc


def prepare_X(df):
    feat_cols = get_feature_cols(df)
    cat_cols = ['code', 'sub_code', 'sub_category']
    X = df[feat_cols + cat_cols + ['horizon', 'ts_index']].copy()
    for c in cat_cols:
        X[c] = X[c].astype('category')
    return X


def run_train_val(tr, val, stats):
    tr  = apply_group_stats(tr,  *stats)
    val = apply_group_stats(val, *stats)

    X_tr  = prepare_X(tr)
    X_val = prepare_X(val)
    model = train_lgbm(X_tr, tr.y_target.values, tr.weight.values,
                       X_val, val.y_target.values, val.weight.values,
                       params=PARAMS)

    val_pred = model.predict(X_val)
    score = weighted_rmse_score(val.y_target.values, val_pred, val.weight.values)
    print(f"\n[Trial 016] Cold-start val score: {score:.6f}")

    imp = pd.Series(model.feature_importance('gain'), index=X_tr.columns).sort_values(ascending=False)
    print("\nTop 15 feature importance:")
    print(imp.head(15))

    return model.best_iteration, X_tr.columns.tolist()


def main():
    train = load_train()
    check_memory('after load')

    # cold-start val split
    all_series = train[KEY].drop_duplicates()
    np.random.seed(42)
    val_idx = np.random.choice(len(all_series), size=int(len(all_series) * 0.2), replace=False)
    val_keys = set(map(tuple, all_series.iloc[val_idx].values))
    is_val = train[KEY].apply(tuple, axis=1).isin(val_keys)
    tr  = train[~is_val].copy()
    val = train[is_val].copy()
    del train, is_val, all_series
    gc.collect()

    print(f"Train series: {tr[KEY].drop_duplicates().shape[0]}, Val series: {val[KEY].drop_duplicates().shape[0]}")
    print(f"Train rows: {len(tr)}, Val rows: {len(val)}")
    check_memory('after split')

    # stats from tr only (val은 cold-start)
    stats = compute_group_stats(tr)
    best_iter, feat_cols = run_train_val(tr, val, stats)

    del tr, val
    gc.collect()
    check_memory('after val, before retrain')

    # full retrain
    train_full = load_train()
    stats_full = compute_group_stats(train_full)
    train_full = apply_group_stats(train_full, *stats_full)
    y_full = train_full.y_target.values
    w_full = train_full.weight.values
    X_full = prepare_X(train_full)
    del train_full
    gc.collect()

    model_full = retrain_full(X_full, y_full, w_full, best_iter, params=PARAMS)
    del X_full, y_full, w_full
    gc.collect()
    check_memory('after retrain')

    test = load_test()
    test = apply_group_stats(test, *stats_full)
    X_te = prepare_X(test)

    for col in X_te.columns:
        nan_pct = X_te[col].isna().mean()
        if nan_pct > 0.05:
            print(f"WARNING test: {col} NaN {nan_pct*100:.1f}%")

    save_submission(test, model_full.predict(X_te), 'trial_016_more_enc',
                    output_dir=Path(__file__).parent)


if __name__ == '__main__':
    main()
