"""Trial 021: num_boost_round 2000→3000
- 020이 2000라운드 한도에서 걸림 (아직 개선 중)
- 나머지는 trial_021과 동일
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
EXCLUDE_FEATURES = {'feature_w', 'feature_x', 'feature_y', 'feature_z'}
PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 255,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
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
    s2 = df.groupby(['sub_category', 'horizon'])['y_target'].agg(
        subcat_mean='mean', subcat_std='std'
    ).reset_index()
    s3 = df.groupby(['horizon'])['y_target'].agg(
        horizon_mean='mean', horizon_std='std'
    ).reset_index()
    return s2, s3


def apply_group_stats(df, s2, s3):
    df = df.merge(s2, on=['sub_category', 'horizon'], how='left')
    df = df.merge(s3, on=['horizon'], how='left')
    return df


def get_feature_cols(df):
    raw = [c for c in df.columns if c.startswith('feature_') and c not in EXCLUDE_FEATURES]
    enc = [c for c in ['subcat_mean', 'subcat_std', 'horizon_mean', 'horizon_std'] if c in df.columns]
    return raw + enc


def prepare_X(df):
    feat_cols = get_feature_cols(df)
    cat_cols = ['code', 'sub_code', 'sub_category']
    X = df[feat_cols + cat_cols + ['horizon', 'ts_index']].copy()
    for c in cat_cols:
        X[c] = X[c].astype('category')
    return X


def main():
    train = load_train()
    check_memory('after load')

    # cold-start val split: 시리즈 20% holdout
    all_series = train[KEY].drop_duplicates()
    np.random.seed(42)
    val_idx = np.random.choice(len(all_series), size=int(len(all_series) * 0.1), replace=False)
    val_keys = set(map(tuple, all_series.iloc[val_idx].values))
    is_val = train[KEY].apply(tuple, axis=1).isin(val_keys)
    tr  = train[~is_val].copy()
    val = train[is_val].copy()
    del train, is_val, all_series
    gc.collect()

    print(f"Train series: {tr[KEY].drop_duplicates().shape[0]}, Val series: {val[KEY].drop_duplicates().shape[0]}")
    print(f"Train rows: {len(tr)}, Val rows: {len(val)}")

    s2, s3 = compute_group_stats(tr)
    tr  = apply_group_stats(tr,  s2, s3)
    val = apply_group_stats(val, s2, s3)
    check_memory('after encoding')

    X_tr  = prepare_X(tr)
    X_val = prepare_X(val)

    import lightgbm as lgb
    cat = ['code', 'sub_code', 'sub_category']
    dtrain = lgb.Dataset(X_tr, label=tr.y_target.values, weight=tr.weight.values,
                         categorical_feature=cat, free_raw_data=True)
    dval   = lgb.Dataset(X_val, label=val.y_target.values, weight=val.weight.values,
                         categorical_feature=cat, reference=dtrain, free_raw_data=True)
    model = lgb.train(
        PARAMS, dtrain, num_boost_round=3000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
    )

    val_pred = model.predict(X_val)
    score = weighted_rmse_score(val.y_target.values, val_pred, val.weight.values)
    print(f"\n[Trial 017] Cold-start val score: {score:.6f}")

    imp = pd.Series(model.feature_importance('gain'), index=X_tr.columns).sort_values(ascending=False)
    print("\nTop 15 feature importance:")
    print(imp.head(15))

    best_iter = model.best_iteration
    del tr, val, X_tr, X_val, model
    gc.collect()
    check_memory('after val, before retrain')

    # full retrain
    train_full = load_train()
    s2_f, s3_f = compute_group_stats(train_full)
    train_full = apply_group_stats(train_full, s2_f, s3_f)
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
    test = apply_group_stats(test, s2_f, s3_f)
    X_te = prepare_X(test)

    for col in X_te.columns:
        nan_pct = X_te[col].isna().mean()
        if nan_pct > 0.05:
            print(f"WARNING test: {col} NaN {nan_pct*100:.1f}%")

    save_submission(test, model_full.predict(X_te), 'trial_021_tuned',
                    output_dir=Path(__file__).parent)


if __name__ == '__main__':
    main()
