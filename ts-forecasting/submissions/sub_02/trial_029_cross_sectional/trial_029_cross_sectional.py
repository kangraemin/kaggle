"""Trial 029: Cross-sectional raw feature stats
- 같은 ts_index에서 전체 엔티티의 raw feature 평균/표준편차
- Anatoly Karpov (9위) 방식 확인: cross-sectional stats 허용
- ts_index가 2위 feature인 이유 = regime 정보 → 직접 피처화
- train: groupby ts_index → mean/std
- test: 동일하게 test 데이터에서 계산 (sequential, 합법)
- cold-start val 10% holdout
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
# top MI features + top importance features
CS_FEATURES = [
    'feature_bs', 'feature_am', 'feature_bx', 'feature_bt', 'feature_bw',
    'feature_u',  'feature_v',  'feature_a',  'feature_cg', 'feature_cf',
    'feature_aq', 'feature_ce', 'feature_as', 'feature_h',  'feature_ai',
]
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


def compute_cross_sectional(df):
    """ts_index별 cross-sectional feature 평균/표준편차."""
    cs_cols = [c for c in CS_FEATURES if c in df.columns]
    agg = df.groupby('ts_index')[cs_cols].agg(['mean', 'std'])
    agg.columns = [f'cs_{c}_{s}' for c, s in agg.columns]
    agg = agg.reset_index()
    return agg


def compute_group_stats(df):
    s2 = df.groupby(['sub_category', 'horizon'])['y_target'].agg(
        subcat_mean='mean', subcat_std='std'
    ).reset_index()
    s3 = df.groupby(['horizon'])['y_target'].agg(
        horizon_mean='mean', horizon_std='std'
    ).reset_index()
    return s2, s3


def apply_stats(df, cs_stats, s2, s3):
    df = df.merge(cs_stats, on='ts_index', how='left')
    df = df.merge(s2, on=['sub_category', 'horizon'], how='left')
    df = df.merge(s3, on=['horizon'], how='left')
    return df


def get_feature_cols(df):
    raw = [c for c in df.columns if c.startswith('feature_') and c not in EXCLUDE_FEATURES]
    enc = [c for c in ['subcat_mean', 'subcat_std', 'horizon_mean', 'horizon_std'] if c in df.columns]
    cs  = [c for c in df.columns if c.startswith('cs_')]
    return raw + enc + cs


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

    # cold-start val split (10%)
    all_series = train[KEY].drop_duplicates()
    np.random.seed(42)
    val_idx = np.random.choice(len(all_series), size=int(len(all_series) * 0.1), replace=False)
    val_keys = set(map(tuple, all_series.iloc[val_idx].values))
    is_val = train[KEY].apply(tuple, axis=1).isin(val_keys)
    tr  = train[~is_val].copy()
    val = train[is_val].copy()
    del train, is_val, all_series
    gc.collect()

    print(f"Train: {len(tr):,}, Val: {len(val):,}")

    # cross-sectional stats (tr 기준)
    print("Computing cross-sectional stats...")
    cs_stats = compute_cross_sectional(tr)
    print(f"Cross-sectional features: {[c for c in cs_stats.columns if c != 'ts_index'][:5]}...")

    s2, s3 = compute_group_stats(tr)
    tr  = apply_stats(tr,  cs_stats, s2, s3)
    val = apply_stats(val, cs_stats, s2, s3)

    # val NaN 체크
    cs_cols = [c for c in tr.columns if c.startswith('cs_')]
    print(f"Val cs_feature NaN: {val[cs_cols[0]].isna().mean()*100:.1f}%")
    check_memory('after encoding')

    X_tr  = prepare_X(tr)
    X_val = prepare_X(val)
    cat = ['code', 'sub_code', 'sub_category']
    dtrain = lgb.Dataset(X_tr, label=tr.y_target.values, weight=tr.weight.values,
                         categorical_feature=cat, free_raw_data=True)
    dval   = lgb.Dataset(X_val, label=val.y_target.values, weight=val.weight.values,
                         categorical_feature=cat, reference=dtrain, free_raw_data=True)
    model = lgb.train(PARAMS, dtrain, num_boost_round=3000,
                      valid_sets=[dval],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])

    val_pred = model.predict(X_val)
    score = weighted_rmse_score(val.y_target.values, val_pred, val.weight.values)
    print(f"\n[Trial 029] Cold-start val score: {score:.6f}")

    imp = pd.Series(model.feature_importance('gain'), index=X_tr.columns).sort_values(ascending=False)
    print("\nTop 20 feature importance:")
    print(imp.head(20))

    best_iter = model.best_iteration
    del tr, val, X_tr, X_val, model
    gc.collect()
    check_memory('after val, before retrain')

    # full retrain
    train_full = load_train()
    cs_stats_full = compute_cross_sectional(train_full)
    s2_f, s3_f = compute_group_stats(train_full)
    train_full = apply_stats(train_full, cs_stats_full, s2_f, s3_f)
    y_full = train_full.y_target.values
    w_full = train_full.weight.values
    X_full = prepare_X(train_full)
    del train_full
    gc.collect()

    model_full = retrain_full(X_full, y_full, w_full, best_iter, params=PARAMS)
    del X_full, y_full, w_full
    gc.collect()
    check_memory('after retrain')

    # test: test 데이터에서 cross-sectional stats 계산 (합법)
    test = load_test()
    cs_stats_test = compute_cross_sectional(test)
    test = apply_stats(test, cs_stats_test, s2_f, s3_f)

    X_te = prepare_X(test)
    for col in X_te.columns:
        nan_pct = X_te[col].isna().mean()
        if nan_pct > 0.05:
            print(f"WARNING test: {col} NaN {nan_pct*100:.1f}%")

    save_submission(test, model_full.predict(X_te), 'trial_029_cross_sectional',
                    output_dir=Path(__file__).parent)


if __name__ == '__main__':
    main()
