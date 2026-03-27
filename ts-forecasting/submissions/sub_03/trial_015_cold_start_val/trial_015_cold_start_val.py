"""Trial 015: cold-start val split
- val = 시리즈 20% 완전 holdout (모델이 본 적 없는 시리즈)
- test 구조 재현: 89% 신규 시리즈 → cold-start 평가가 진짜 성능
- feature: raw + sub_category×horizon / horizon encoding (series-level 제외)
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


def check_memory(label=''):
    avail = psutil.virtual_memory().available / 1024**3
    print(f"[MEM] {label}: {avail:.1f} GB free", flush=True)
    if avail < MEM_LIMIT_GB:
        print(f"[MEM] 위험 — 강제 종료", flush=True)
        sys.exit(1)


def compute_group_stats(df):
    """train df로 sub_category×horizon, horizon 통계 계산."""
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
    raw = [c for c in df.columns if c.startswith('feature_')]
    enc = [c for c in ['subcat_mean', 'subcat_std', 'horizon_mean', 'horizon_std'] if c in df.columns]
    return raw + enc


def prepare_X_015(df):
    feat_cols = get_feature_cols(df)
    cat_cols = ['code', 'sub_code', 'sub_category']
    X = df[feat_cols + cat_cols + ['horizon', 'ts_index']].copy()
    for c in cat_cols:
        X[c] = X[c].astype('category')
    return X


def main():
    train = load_train()
    check_memory('after load train')

    # 1) cold-start val split: 시리즈 20% 완전 holdout
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

    # 2) group stats (tr 기준으로만 계산 — val은 cold-start)
    s2, s3 = compute_group_stats(tr)
    tr  = apply_group_stats(tr,  s2, s3)
    val = apply_group_stats(val, s2, s3)
    check_memory('after encoding')

    # 3) train
    X_tr  = prepare_X_015(tr)
    X_val = prepare_X_015(val)
    model = train_lgbm(X_tr, tr.y_target.values, tr.weight.values,
                       X_val, val.y_target.values, val.weight.values)

    val_pred = model.predict(X_val)
    score = weighted_rmse_score(val.y_target.values, val_pred, val.weight.values)
    print(f"\n[Trial 015] Cold-start val score: {score:.6f}")

    imp = pd.Series(model.feature_importance('gain'), index=X_tr.columns).sort_values(ascending=False)
    print("\nTop 15 feature importance:")
    print(imp.head(15))

    for col in X_val.columns:
        nan_pct = X_val[col].isna().mean()
        if nan_pct > 0.05:
            print(f"WARNING val: {col} NaN {nan_pct*100:.1f}%")

    best_iter = model.best_iteration
    del tr, val, X_tr, X_val, model
    gc.collect()
    check_memory('after val, before retrain')

    # 4) full retrain — 전체 train으로 stats 재계산
    train_full = load_train()
    s2_full, s3_full = compute_group_stats(train_full)
    train_full = apply_group_stats(train_full, s2_full, s3_full)

    y_full = train_full.y_target.values
    w_full = train_full.weight.values
    X_full = prepare_X_015(train_full)
    del train_full
    gc.collect()

    model_full = retrain_full(X_full, y_full, w_full, best_iter)
    del X_full, y_full, w_full
    gc.collect()
    check_memory('after retrain')

    # 5) test prediction
    test = load_test()
    test = apply_group_stats(test, s2_full, s3_full)

    X_te = prepare_X_015(test)
    for col in X_te.columns:
        nan_pct = X_te[col].isna().mean()
        if nan_pct > 0.05:
            print(f"WARNING test: {col} NaN {nan_pct*100:.1f}%")

    save_submission(test, model_full.predict(X_te), 'trial_015_cold_start_val',
                    output_dir=Path(__file__).parent)


if __name__ == '__main__':
    main()
