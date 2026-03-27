"""Trial 022: known/unknown 시리즈 분리 모델
- known series (test 10.9%): series stats 포함한 model_A (ts_index val)
- unknown series (test 89.1%): trial_021 예측값 그대로 사용
- series_mean이 known series에서 압도적 signal → 따로 최적화
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
TRIAL_021_CSV = Path(__file__).parent.parent / 'trial_021_more_rounds' / 'trial_021_tuned.csv'


def check_memory(label=''):
    avail = psutil.virtual_memory().available / 1024**3
    print(f"[MEM] {label}: {avail:.1f} GB free", flush=True)
    if avail < MEM_LIMIT_GB:
        print(f"[MEM] 위험 — 강제 종료", flush=True)
        sys.exit(1)


def compute_series_stats(df):
    return df.groupby(KEY)['y_target'].agg(
        series_mean='mean', series_std='std', series_median='median'
    ).reset_index()


def compute_group_stats(df):
    s2 = df.groupby(['sub_category', 'horizon'])['y_target'].agg(
        subcat_mean='mean', subcat_std='std'
    ).reset_index()
    s3 = df.groupby(['horizon'])['y_target'].agg(
        horizon_mean='mean', horizon_std='std'
    ).reset_index()
    return s2, s3


def apply_stats(df, series_stats, s2, s3):
    df = df.merge(series_stats, on=KEY, how='left')
    df = df.merge(s2, on=['sub_category', 'horizon'], how='left')
    df = df.merge(s3, on=['horizon'], how='left')
    return df


def get_feature_cols(df):
    raw = [c for c in df.columns if c.startswith('feature_') and c not in EXCLUDE_FEATURES]
    enc = [c for c in df.columns if c.endswith('_mean') or c.endswith('_std') or c.endswith('_median')]
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

    train_keys = set(map(tuple, train[KEY].drop_duplicates().values))

    # series stats는 tr(<=CUTOFF) 기준으로만 계산 (val leak 방지)
    tr_raw = train[train.ts_index <= CUTOFF]
    val_raw = train[train.ts_index > CUTOFF]

    series_stats_tr = compute_series_stats(tr_raw)
    s2_tr, s3_tr = compute_group_stats(tr_raw)

    tr  = apply_stats(tr_raw.copy(),  series_stats_tr, s2_tr, s3_tr)
    val = apply_stats(val_raw.copy(), series_stats_tr, s2_tr, s3_tr)
    del tr_raw, val_raw
    gc.collect()

    print(f"Model A — Train: {len(tr):,}, Val: {len(val):,}")
    check_memory('after encoding')

    # Model A 학습
    X_tr  = prepare_X(tr)
    X_val = prepare_X(val)
    cat = ['code', 'sub_code', 'sub_category']
    dtrain = lgb.Dataset(X_tr, label=tr.y_target.values, weight=tr.weight.values,
                         categorical_feature=cat, free_raw_data=True)
    dval   = lgb.Dataset(X_val, label=val.y_target.values, weight=val.weight.values,
                         categorical_feature=cat, reference=dtrain, free_raw_data=True)
    model_A = lgb.train(PARAMS, dtrain, num_boost_round=3000,
                        valid_sets=[dval],
                        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])

    val_pred = model_A.predict(X_val)
    score = weighted_rmse_score(val.y_target.values, val_pred, val.weight.values)
    print(f"\n[Trial 022] Model A val score (ts_index split): {score:.6f}")

    imp = pd.Series(model_A.feature_importance('gain'), index=X_tr.columns).sort_values(ascending=False)
    print("\nTop 10 feature importance:")
    print(imp.head(10))

    best_iter = model_A.best_iteration
    del tr, val, X_tr, X_val
    gc.collect()
    check_memory('after val, before retrain')

    # full retrain (전체 train 기준 stats)
    series_stats_full = compute_series_stats(train)
    s2_full, s3_full = compute_group_stats(train)
    train_full = apply_stats(train, series_stats_full, s2_full, s3_full)
    y_full = train_full.y_target.values
    w_full = train_full.weight.values
    X_full = prepare_X(train_full)
    del train, train_full
    gc.collect()

    model_A_full = retrain_full(X_full, y_full, w_full, best_iter, params=PARAMS)
    del X_full, y_full, w_full
    gc.collect()
    check_memory('after retrain')

    # Test 예측 조합
    test = load_test()
    test['is_known'] = test[KEY].apply(tuple, axis=1).isin(train_keys)
    print(f"\nTest known: {test.is_known.sum():,}, unknown: {(~test.is_known).sum():,}")

    # trial_021 예측 로드 (unknown용)
    preds_021 = pd.read_csv(TRIAL_021_CSV).set_index('id')['prediction']

    # known series → model_A 예측
    test_known = test[test.is_known].copy()
    test_known = apply_stats(test_known, series_stats_full, s2_full, s3_full)
    X_known = prepare_X(test_known)
    preds_known = pd.Series(model_A_full.predict(X_known), index=test_known['id'].values)

    # 합치기
    final_preds = preds_021.copy()
    final_preds.update(preds_known)
    final_preds = final_preds.reindex(test['id'].values)

    print(f"NaN in final: {final_preds.isna().sum()}")
    out = Path(__file__).parent / 'trial_022_split_model.csv'
    pd.DataFrame({'id': test['id'], 'prediction': final_preds.values}).to_csv(out, index=False)
    print(f"Saved → {out}")


if __name__ == '__main__':
    main()
