"""Trial 024: Target Normalization (z-score per series)
- y_norm = (y_target - series_mean) / series_std 로 학습
- 역변환: y_pred = denorm_mean + y_norm_pred × denorm_std
- known series → 정확한 series_mean/std
- unknown series → subcat×horizon mean/std fallback
- 시리즈간 스케일 차이 제거 → 패턴 학습에 집중
- cold-start val (10% holdout)
"""
import gc
import sys
import psutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
import lightgbm as lgb
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


def compute_norm_stats(df):
    """시리즈별 mean/std (normalization 용)."""
    stats = df.groupby(KEY)['y_target'].agg(
        series_mean='mean', series_std='std'
    ).reset_index()
    stats['series_std'] = stats['series_std'].fillna(1.0).clip(lower=1e-6)
    return stats


def compute_group_stats(df):
    """subcat×horizon mean/std (unknown series fallback)."""
    s2 = df.groupby(['sub_category', 'horizon'])['y_target'].agg(
        subcat_mean='mean', subcat_std='std'
    ).reset_index()
    s2['subcat_std'] = s2['subcat_std'].fillna(1.0).clip(lower=1e-6)
    s3 = df.groupby(['horizon'])['y_target'].agg(
        horizon_mean='mean', horizon_std='std'
    ).reset_index()
    s3['horizon_std'] = s3['horizon_std'].fillna(1.0).clip(lower=1e-6)
    return s2, s3


def apply_norm_stats(df, norm_stats, s2, s3):
    df = df.merge(norm_stats, on=KEY, how='left')
    df = df.merge(s2, on=['sub_category', 'horizon'], how='left')
    df = df.merge(s3, on=['horizon'], how='left')

    # fallback: series → subcat → horizon
    df['denorm_mean'] = df['series_mean'].fillna(df['subcat_mean']).fillna(df['horizon_mean'])
    df['denorm_std']  = df['series_std'].fillna(df['subcat_std']).fillna(df['horizon_std'])
    return df


def normalize_target(df):
    """y_target → y_norm."""
    df['y_norm'] = (df['y_target'] - df['denorm_mean']) / df['denorm_std']
    return df


def get_feature_cols(df):
    raw = [c for c in df.columns if c.startswith('feature_') and c not in EXCLUDE_FEATURES]
    enc = ['subcat_mean', 'subcat_std', 'horizon_mean', 'horizon_std', 'denorm_mean', 'denorm_std']
    enc = [c for c in enc if c in df.columns]
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

    # cold-start val split (10% holdout)
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
    check_memory('after split')

    # tr 기준으로 normalization stats 계산
    norm_stats = compute_norm_stats(tr)
    s2, s3 = compute_group_stats(tr)

    tr  = apply_norm_stats(tr,  norm_stats, s2, s3)
    val = apply_norm_stats(val, norm_stats, s2, s3)

    tr  = normalize_target(tr)
    val = normalize_target(val)

    print(f"y_norm range: {tr.y_norm.min():.2f} ~ {tr.y_norm.max():.2f}")
    print(f"val denorm_mean NaN: {val.denorm_mean.isna().mean()*100:.1f}%")
    check_memory('after norm')

    # 정규화된 target으로 학습
    X_tr  = prepare_X(tr)
    X_val = prepare_X(val)
    cat = ['code', 'sub_code', 'sub_category']
    dtrain = lgb.Dataset(X_tr, label=tr.y_norm.values, weight=tr.weight.values,
                         categorical_feature=cat, free_raw_data=True)
    dval   = lgb.Dataset(X_val, label=val.y_norm.values, weight=val.weight.values,
                         categorical_feature=cat, reference=dtrain, free_raw_data=True)
    model = lgb.train(PARAMS, dtrain, num_boost_round=3000,
                      valid_sets=[dval],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])

    # 역변환 후 평가
    val_norm_pred = model.predict(X_val)
    val_pred = val['denorm_mean'].values + val_norm_pred * val['denorm_std'].values
    score = weighted_rmse_score(val.y_target.values, val_pred, val.weight.values)
    print(f"\n[Trial 024] Cold-start val score: {score:.6f}")

    imp = pd.Series(model.feature_importance('gain'), index=X_tr.columns).sort_values(ascending=False)
    print("\nTop 15 feature importance:")
    print(imp.head(15))

    best_iter = model.best_iteration
    del tr, val, X_tr, X_val, model
    gc.collect()
    check_memory('after val, before retrain')

    # full retrain
    train_full = load_train()
    norm_stats_full = compute_norm_stats(train_full)
    s2_full, s3_full = compute_group_stats(train_full)
    train_full = apply_norm_stats(train_full, norm_stats_full, s2_full, s3_full)
    train_full = normalize_target(train_full)

    y_full = train_full.y_norm.values
    w_full = train_full.weight.values
    X_full = prepare_X(train_full)
    del train_full
    gc.collect()

    model_full = retrain_full(X_full, y_full, w_full, best_iter, params=PARAMS)
    del X_full, y_full, w_full
    gc.collect()
    check_memory('after retrain')

    # test 예측
    test = load_test()
    test = apply_norm_stats(test, norm_stats_full, s2_full, s3_full)

    print(f"\nTest denorm_mean NaN: {test.denorm_mean.isna().mean()*100:.1f}%")

    X_te = prepare_X(test)
    norm_preds = model_full.predict(X_te)
    final_preds = test['denorm_mean'].values + norm_preds * test['denorm_std'].values

    print(f"Final pred: mean={final_preds.mean():.4f}, std={final_preds.std():.4f}")

    save_submission(test, final_preds, 'trial_024_target_norm',
                    output_dir=Path(__file__).parent)


if __name__ == '__main__':
    main()
