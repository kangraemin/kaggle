"""Trial 050: weight clipping
- weight 최대 13조 → 학습이 극소수 시리즈에 쏠림
- 99.9th percentile로 클리핑 → 더 균형있는 학습
- trial_051(0.4887) 기반
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
PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 511,
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


CLIP_BOUNDS = {}  # train에서 계산한 feature 범위


def compute_clip_bounds(df):
    """train 기준 feature 클리핑 범위 계산."""
    feat_cols = [c for c in df.columns if c.startswith('feature_')]
    bounds = {}
    for col in feat_cols:
        bounds[col] = (float(df[col].quantile(0.001)), float(df[col].quantile(0.99)))
    return bounds


def clip_features(df, bounds):
    """test feature를 train 범위로 클리핑."""
    df = df.copy()
    for col, (lo, hi) in bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lo, upper=hi)
    return df


def add_series_features(df, train_df):
    """known series에 시계열 features 추가: lag 1~3, mean, std."""
    lo = float(train_df['y_target'].quantile(0.001))
    hi = float(train_df['y_target'].quantile(0.99))

    sorted_tr = train_df.sort_values(KEY + ['ts_index'])
    grp = sorted_tr.groupby(KEY)['y_target']

    stats = grp.agg(
        series_mean='mean',
        series_std='std',
    ).reset_index()
    stats['series_mean'] = stats['series_mean'].clip(lower=lo, upper=hi)
    stats['series_std']  = stats['series_std'].clip(lower=0, upper=(hi - lo))

    # lag 1~3
    last20 = sorted_tr.groupby(KEY)['y_target'].apply(
        lambda x: (list(np.full(max(0, 20-len(x)), np.nan)) + list(x.values))[-20:]
    )
    vals20 = pd.DataFrame(last20.tolist(), index=last20.index)
    # lag 1~5, 10, 20
    lag_df = pd.DataFrame({
        'last_y_1':  vals20.iloc[:, -1],
        'last_y_2':  vals20.iloc[:, -2],
        'last_y_3':  vals20.iloc[:, -3],
        'last_y_4':  vals20.iloc[:, -4],
        'last_y_5':  vals20.iloc[:, -5],
        'last_y_10': vals20.iloc[:, -10],
        'last_y_20': vals20.iloc[:, 0],
    }, index=last20.index)
    for col in lag_df.columns:
        lag_df[col] = lag_df[col].clip(lower=lo, upper=hi)

    # rolling stats + trend + ewm
    rolling_df = pd.DataFrame({
        'roll_mean_5':  vals20.apply(lambda x: np.nanmean(x.iloc[-5:]),  axis=1),
        'roll_mean_10': vals20.apply(lambda x: np.nanmean(x.iloc[-10:]), axis=1),
        'roll_std_5':   vals20.apply(lambda x: np.nanstd(x.iloc[-5:]),   axis=1),
        'roll_std_10':  vals20.apply(lambda x: np.nanstd(x.iloc[-10:]),  axis=1),
        'ewm_5':        vals20.apply(lambda x: pd.Series(x.dropna()).ewm(span=5).mean().iloc[-1] if x.notna().sum() > 0 else np.nan, axis=1),
        'trend_1_5':    (vals20.iloc[:, -1] - vals20.iloc[:, -5]) / 5.0,
        'trend_1_10':   (vals20.iloc[:, -1] - vals20.iloc[:, -10]) / 10.0,
    }, index=last20.index)
    for col in rolling_df.columns:
        rolling_df[col] = rolling_df[col].clip(lower=lo if 'std' not in col and 'trend' not in col else None,
                                                upper=hi if 'std' not in col and 'trend' not in col else None)
    lag_df = lag_df.join(rolling_df)
    lag_df = lag_df.reset_index()

    result = stats.merge(lag_df, on=KEY, how='left')
    df = df.merge(result, on=KEY, how='left')  # unknown → NaN
    return df


def prepare_X(df):
    feat_cols = [c for c in df.columns if c.startswith('feature_')]
    cat_cols = ['code', 'sub_code', 'sub_category']
    series_feats = [c for c in [
        'last_y_1', 'last_y_2', 'last_y_3', 'last_y_4', 'last_y_5', 'last_y_10', 'last_y_20',
        'roll_mean_5', 'roll_mean_10', 'roll_std_5', 'roll_std_10', 'ewm_5',
        'trend_1_5', 'trend_1_10', 'series_mean', 'series_std',
        'code_mean', 'code_std', 'subcode_mean', 'subcode_std',
        'subcat_mean', 'subcat_std', 'horizon_mean', 'horizon_std'
    ] if c in df.columns]
    X = df[feat_cols + cat_cols + ['horizon', 'ts_index'] + series_feats].copy()
    for c in cat_cols:
        X[c] = X[c].astype('category')
    return X


def main():
    train = load_train()
    check_memory('after load')

    # train feature 클리핑 범위 계산 후 train도 클리핑
    print("Computing feature clip bounds...")
    bounds = compute_clip_bounds(train)
    train = clip_features(train, bounds)
    print("Train features clipped.")

    # group stats (unknown series용)
    print("Adding group stats...")
    code_stats    = train.groupby('code')['y_target'].agg(code_mean='mean', code_std='std').reset_index()
    subcode_stats = train.groupby(['sub_code','horizon'])['y_target'].agg(subcode_mean='mean', subcode_std='std').reset_index()
    subcat_stats  = train.groupby(['sub_category','horizon'])['y_target'].agg(subcat_mean='mean', subcat_std='std').reset_index()
    h_stats       = train.groupby('horizon')['y_target'].agg(horizon_mean='mean', horizon_std='std').reset_index()
    train = (train.merge(code_stats, on='code', how='left')
                  .merge(subcode_stats, on=['sub_code','horizon'], how='left')
                  .merge(subcat_stats, on=['sub_category','horizon'], how='left')
                  .merge(h_stats, on='horizon', how='left'))
    # 시계열 features 추가 (known series만)
    print("Adding series features...")
    train = add_series_features(train, train)

    # Weight clipping (99.9th percentile)
    w_cap = float(train['weight'].quantile(0.99))
    train['weight'] = train['weight'].clip(upper=w_cap)
    print(f"Weight clipped at: {w_cap:.2e}")

    # Mixed val: ts_index>2880 (known) + cold-start 5% holdout (unknown)
    all_series = train[KEY].drop_duplicates()
    np.random.seed(42)
    holdout_idx = np.random.choice(len(all_series), size=int(len(all_series) * 0.15), replace=False)
    holdout_keys = set(map(tuple, all_series.iloc[holdout_idx].values))
    is_holdout = train[KEY].apply(tuple, axis=1).isin(holdout_keys)

    # tr = ts_index<=CUTOFF AND not holdout
    # val = (ts_index>CUTOFF AND not holdout) OR holdout
    tr  = train[~is_holdout & (train.ts_index <= CUTOFF)]
    val = pd.concat([
        train[~is_holdout & (train.ts_index > CUTOFF)],  # time-split known
        train[is_holdout]                                  # cold-start unknown
    ], ignore_index=True)
    print(f"Mixed val: time-split {(~is_holdout & (train.ts_index>CUTOFF)).sum():,} + cold-start {is_holdout.sum():,}")
    print(f"Train: {len(tr):,}, Val: {len(val):,}")

    # 고가중치 시리즈가 training에 포함됐는지 확인
    hw_codes = ['83EG83KQ', '1HEMHZK2', 'VFWIFJPS', 'K8I5QG74', 'SJZP0OVU']
    for code in hw_codes:
        in_tr = (tr['code'] == code).sum()
        in_val = (val['code'] == code).sum()
        print(f"  {code}: tr={in_tr:,}, val={in_val:,}")

    check_memory('after split')

    X_tr  = prepare_X(tr)
    X_val = prepare_X(val)
    cat = ['code', 'sub_code', 'sub_category']
    dtrain = lgb.Dataset(X_tr, label=tr.y_target.values, weight=tr.weight.values,
                         categorical_feature=cat, free_raw_data=True)
    dval   = lgb.Dataset(X_val, label=val.y_target.values, weight=val.weight.values,
                         categorical_feature=cat, reference=dtrain, free_raw_data=True)
    model = lgb.train(PARAMS, dtrain, num_boost_round=5000,
                      valid_sets=[dval],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])

    val_pred = model.predict(X_val)
    score = weighted_rmse_score(val.y_target.values, val_pred, val.weight.values)
    print(f"\n[Trial 031] Val score: {score:.6f}")

    # 고가중치 시리즈 예측 확인
    for code in hw_codes:
        mask = val['code'] == code
        if mask.sum() > 0:
            preds = val_pred[mask.values]
            print(f"  {code} val pred: mean={preds.mean():.6f}, max_abs={np.abs(preds).max():.6f}")

    imp = pd.Series(model.feature_importance('gain'), index=X_tr.columns).sort_values(ascending=False)
    print("\nTop 15 feature importance:")
    print(imp.head(15))

    best_iter = model.best_iteration
    del tr, val, X_tr, X_val, model
    gc.collect()
    check_memory('after val, before retrain')

    # full retrain (이미 클리핑된 train 사용)
    y_full = train.y_target.values
    w_full = train.weight.values
    X_full = prepare_X(train)
    del train
    gc.collect()

    model_full = retrain_full(X_full, y_full, w_full, best_iter, params=PARAMS)
    del X_full, y_full, w_full
    gc.collect()
    check_memory('after retrain')

    test = load_test()
    test = clip_features(test, bounds)
    train_full_for_feats = load_train()
    # group stats (unknown series용)
    cs_f  = train_full_for_feats.groupby('code')['y_target'].agg(code_mean='mean', code_std='std').reset_index()
    scs_f = train_full_for_feats.groupby(['sub_code','horizon'])['y_target'].agg(subcode_mean='mean', subcode_std='std').reset_index()
    ss_f  = train_full_for_feats.groupby(['sub_category','horizon'])['y_target'].agg(subcat_mean='mean', subcat_std='std').reset_index()
    hs_f  = train_full_for_feats.groupby('horizon')['y_target'].agg(horizon_mean='mean', horizon_std='std').reset_index()
    test = (test.merge(cs_f, on='code', how='left')
                .merge(scs_f, on=['sub_code','horizon'], how='left')
                .merge(ss_f, on=['sub_category','horizon'], how='left')
                .merge(hs_f, on='horizon', how='left'))
    # series features (known series용)
    test = add_series_features(test, train_full_for_feats)
    known_cnt = test['last_y_1'].notna().sum()
    print(f"Test known series: {known_cnt:,} ({known_cnt/len(test)*100:.1f}%)")
    del train_full_for_feats
    gc.collect()
    X_te = prepare_X(test)

    preds = model_full.predict(X_te)

    # 고가중치 코드 예측 확인
    for code in hw_codes:
        mask = test['code'] == code
        if mask.sum() > 0:
            p = preds[mask.values]
            print(f"  {code} test pred: mean={p.mean():.6f}, max_abs={np.abs(p).max():.6f}")

    save_submission(test, preds, 'trial_051_ts_val_improved',
                    output_dir=Path(__file__).parent)


if __name__ == '__main__':
    main()
