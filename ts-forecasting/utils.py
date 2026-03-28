"""Shared utilities for all trials."""
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
import pyarrow.parquet as pq
from pathlib import Path

DATA_DIR = Path(__file__).parent
KEY = ['code', 'sub_code', 'sub_category', 'horizon']
CUTOFF = 2880  # val split


def weighted_rmse_score(y_target, y_pred, w):
    denom = np.sum(w * y_target ** 2)
    ratio = np.sum(w * (y_target - y_pred) ** 2) / denom
    clipped = float(np.minimum(np.maximum(ratio, 0.0), 1.0))
    return float(np.sqrt(1.0 - clipped))


def _downcast(df):
    for col in df.select_dtypes('float64').columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes('int64').columns:
        df[col] = df[col].astype('int32')
    for col in ['code', 'sub_code', 'sub_category']:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].astype('category')
    return df


def read_parquet_lean(path, columns=None):
    """Row group 단위로 읽어서 즉시 downcast → 메모리 절약.
    self_destruct=True로 arrow 메모리 즉시 해제."""
    pf = pq.ParquetFile(path)
    chunks = []
    for i in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(i, columns=columns)
        chunk = table.to_pandas(self_destruct=True)
        del table
        _downcast(chunk)
        chunks.append(chunk)
    del pf
    gc.collect()
    result = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    return result


def load_data(columns=None):
    """Load train/test parquet."""
    print("Loading data...")
    train = read_parquet_lean(DATA_DIR / 'train.parquet', columns=columns)
    test_cols = [c for c in (columns or []) if c not in ('y_target', 'weight')] or None
    test = read_parquet_lean(DATA_DIR / 'test.parquet', columns=test_cols)
    print(f"  train: {train.shape}, {train.memory_usage(deep=True).sum()/1024**2:.0f} MB")
    print(f"  test:  {test.shape}, {test.memory_usage(deep=True).sum()/1024**2:.0f} MB")
    return train, test


def load_train(columns=None):
    """Load only train parquet (for memory-constrained envs)."""
    print("Loading train...")
    train = read_parquet_lean(DATA_DIR / 'train.parquet', columns=columns)
    print(f"  train: {train.shape}, {train.memory_usage(deep=True).sum()/1024**2:.0f} MB")
    return train


def load_test(columns=None):
    """Load only test parquet (for memory-constrained envs)."""
    print("Loading test...")
    test = read_parquet_lean(DATA_DIR / 'test.parquet', columns=columns)
    print(f"  test: {test.shape}, {test.memory_usage(deep=True).sum()/1024**2:.0f} MB")
    return test


def combine_train_test(train, test):
    """Combine train and test."""
    test['y_target'] = np.float32(np.nan)
    test['weight']   = np.float32(np.nan)
    combined = pd.concat([train, test], ignore_index=True, copy=False)
    del train, test
    gc.collect()
    combined.sort_values(KEY + ['ts_index'], inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


def add_base_lags(df, lags=(1, 2, 3, 5, 10, 20)):
    g = df.groupby(KEY, sort=False)['y_target']
    for lag in lags:
        df[f'lag_{lag}'] = g.shift(lag)
    return df


def add_rolling(df, windows=(5, 10, 20, 50)):
    g = df.groupby(KEY, sort=False)['y_target']
    for w in windows:
        df[f'roll_mean_{w}'] = g.shift(1).transform(
            lambda x: x.rolling(w, min_periods=1).mean())
        df[f'roll_std_{w}']  = g.shift(1).transform(
            lambda x: x.rolling(w, min_periods=1).std())
    return df


def add_ewm(df, spans=(5, 20, 50)):
    g = df.groupby(KEY, sort=False)['y_target']
    for s in spans:
        df[f'ewm_{s}'] = g.shift(1).transform(
            lambda x: x.ewm(span=s, min_periods=1).mean())
    return df


def add_trend(df):
    if 'lag_1' in df.columns and 'lag_5' in df.columns:
        df['trend_1_5'] = (df['lag_1'] - df['lag_5']) / 5.0
    if 'lag_1' in df.columns and 'lag_10' in df.columns:
        df['trend_1_10'] = (df['lag_1'] - df['lag_10']) / 10.0
    return df


def add_cross_horizon(df):
    """Add other horizons' lag_1 as features at same (code, sub_code, sub_category, ts_index)."""
    base = df[['code','sub_code','sub_category','horizon','ts_index','lag_1']].copy()
    for h in [1, 3, 10, 25]:
        src = base[base.horizon == h][['code','sub_code','sub_category','ts_index','lag_1']]
        src = src.rename(columns={'lag_1': f'cross_h{h}_lag1'})
        df = df.merge(src, on=['code','sub_code','sub_category','ts_index'], how='left')
    return df


def get_feature_cols(df):
    raw    = [c for c in df.columns if c.startswith('feature_')]
    engineered = [c for c in df.columns if
                  c.startswith('lag_') or c.startswith('roll_') or
                  c.startswith('ewm_') or c.startswith('trend_') or
                  c.startswith('cross_') or c.startswith('series_')]
    return raw + engineered


def prepare_X(df):
    feat_cols = get_feature_cols(df)
    cat_cols  = ['code', 'sub_code', 'sub_category']
    X = df[feat_cols + cat_cols + ['horizon', 'ts_index']].copy()
    for c in cat_cols:
        X[c] = X[c].astype('category')
    return X


def train_lgbm(X_tr, y_tr, w_tr, X_val, y_val, w_val,
               params=None, num_boost_round=1000):
    cat = ['code', 'sub_code', 'sub_category']
    if params is None:
        params = {
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
    dtrain = lgb.Dataset(X_tr,  label=y_tr,  weight=w_tr,  categorical_feature=cat,
                         free_raw_data=True)
    dval   = lgb.Dataset(X_val, label=y_val, weight=w_val, categorical_feature=cat,
                         reference=dtrain, free_raw_data=True)
    model = lgb.train(
        params, dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )
    return model


def retrain_full(X_all, y_all, w_all, best_iter, params=None):
    cat = ['code', 'sub_code', 'sub_category']
    if params is None:
        params = {
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
    dall = lgb.Dataset(X_all, label=y_all, weight=w_all, categorical_feature=cat,
                       free_raw_data=True)
    return lgb.train(params, dall, num_boost_round=best_iter)


def validate_and_patch(test_df, preds, train_df=None):
    """제출 전 필수 검증. danger_ratio > 0.1인 코드는 자동 패치.
    검증 없이 save_submission 불가."""
    import numpy as np
    preds = np.array(preds, dtype=float)

    print("\n=== 제출 전 검증 ===")
    print(f"예측값 분포: mean={preds.mean():.4f}, std={preds.std():.4f}, "
          f"min={preds.min():.4f}, max={preds.max():.4f}")
    print(f"abs>10: {(np.abs(preds)>10).sum()}건, abs>100: {(np.abs(preds)>100).sum()}건")

    if train_df is None:
        train_df = load_train()

    denom = float((train_df['weight'].values * train_df['y_target'].values**2).sum())
    code_max_w = train_df.groupby('code')['weight'].max().sort_values(ascending=False)
    code_mean_y = train_df.groupby('code')['y_target'].mean()

    pred_series = pd.Series(preds, index=test_df.index)
    test_df = test_df.copy()
    test_df['_pred'] = pred_series.values

    patched = False
    for code in code_max_w.head(15).index:
        mask = test_df['code'] == code
        if mask.sum() == 0:
            continue
        max_pred = test_df.loc[mask, '_pred'].abs().max()
        max_w = float(code_max_w[code])
        ratio = max_w * max_pred**2 / denom
        flag = '⚠️  PATCH' if ratio > 0.1 else '✅'
        print(f"  {code}: max_pred={max_pred:.4f}, danger_ratio={ratio:.4f} {flag}")
        if ratio > 0.1:
            mean_y = float(code_mean_y.get(code, 0.0))
            test_df.loc[mask, '_pred'] = mean_y
            print(f"    → {code} 패치 완료: {mean_y:.8f}")
            patched = True

    if patched:
        print("⚠️  패치된 예측값으로 저장합니다.")

    print("=== 검증 완료 ===\n")
    return test_df['_pred'].values


def save_submission(test_df, preds, trial_name, output_dir=None, train_df=None):
    if output_dir is None:
        output_dir = DATA_DIR
    preds = validate_and_patch(test_df, preds, train_df=train_df)
    out = Path(output_dir) / f'{trial_name}.csv'
    pd.DataFrame({'id': test_df['id'], 'prediction': preds}).to_csv(out, index=False)
    print(f"Saved → {out}")
