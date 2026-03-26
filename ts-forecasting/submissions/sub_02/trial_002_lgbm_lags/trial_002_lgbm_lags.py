"""
LightGBM + lag features (previous y_target per series)
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path

DATA_DIR = Path(__file__).parent
KEY = ['code', 'sub_code', 'sub_category', 'horizon']


def weighted_rmse_score(y_target, y_pred, w):
    denom = np.sum(w * y_target ** 2)
    ratio = np.sum(w * (y_target - y_pred) ** 2) / denom
    clipped = float(np.minimum(np.maximum(ratio, 0.0), 1.0))
    return float(np.sqrt(1.0 - clipped))


def add_lag_features(df):
    """Add lag & rolling features per series. df must be sorted by KEY + ts_index."""
    g = df.groupby(KEY, sort=False)['y_target']

    lags = [1, 2, 3, 5, 10, 20]
    for lag in lags:
        df[f'lag_{lag}'] = g.shift(lag)

    windows = [5, 10, 20, 50]
    for w in windows:
        rolled = g.shift(1).transform(lambda x: x.rolling(w, min_periods=1).mean())
        df[f'roll_mean_{w}'] = rolled
        rolled_std = g.shift(1).transform(lambda x: x.rolling(w, min_periods=1).std())
        df[f'roll_std_{w}'] = rolled_std

    # trend: (lag1 - lag5) / 5
    if 'lag_1' in df.columns and 'lag_5' in df.columns:
        df['trend_1_5'] = (df['lag_1'] - df['lag_5']) / 5.0

    return df


def prepare_features(df):
    feat_cols = [c for c in df.columns if c.startswith('feature_')]
    lag_cols  = [c for c in df.columns if c.startswith('lag_') or
                 c.startswith('roll_') or c == 'trend_1_5']
    cat_cols  = ['code', 'sub_code', 'sub_category']

    X = df[feat_cols + lag_cols + cat_cols + ['horizon', 'ts_index']].copy()
    for c in cat_cols:
        X[c] = X[c].astype('category')
    return X


def main():
    print("Loading data...")
    train = pd.read_parquet(DATA_DIR / 'train.parquet')
    test  = pd.read_parquet(DATA_DIR / 'test.parquet')

    # Combine for lag computation (y_target = NaN in test)
    test['y_target'] = np.nan
    test['weight']   = np.nan
    combined = pd.concat([train, test], ignore_index=True)

    print("Computing lag features...")
    combined = combined.sort_values(KEY + ['ts_index'])
    combined = add_lag_features(combined)

    train_feat = combined[combined['weight'].notna()].copy()
    test_feat  = combined[combined['weight'].isna()].copy()

    # Time-based validation split
    cutoff = 2880
    tr  = train_feat[train_feat.ts_index <= cutoff]
    val = train_feat[train_feat.ts_index >  cutoff]
    print(f"Train: {len(tr)}, Val: {len(val)}, Test: {len(test_feat)}")

    X_tr  = prepare_features(tr)
    X_val = prepare_features(val)
    X_te  = prepare_features(test_feat)

    y_tr  = tr['y_target'].values
    y_val = val['y_target'].values
    w_tr  = tr['weight'].values
    w_val = val['weight'].values

    cat_feat = ['code', 'sub_code', 'sub_category']
    dtrain = lgb.Dataset(X_tr,  label=y_tr,  weight=w_tr,  categorical_feature=cat_feat)
    dval   = lgb.Dataset(X_val, label=y_val, weight=w_val, categorical_feature=cat_feat,
                         reference=dtrain)

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

    print("Training LightGBM (with lags)...")
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
    )

    val_pred = model.predict(X_val)
    score = weighted_rmse_score(y_val, val_pred, w_val)
    print(f"\nValidation weighted_rmse_score: {score:.6f}")

    # Feature importance top 20
    imp = pd.Series(model.feature_importance('gain'),
                    index=X_tr.columns).sort_values(ascending=False)
    print("\nFeature importance top 20:")
    print(imp.head(20))

    # Retrain on full train
    print("\nRetraining on full train...")
    X_all = prepare_features(train_feat)
    y_all = train_feat['y_target'].values
    w_all = train_feat['weight'].values
    dall  = lgb.Dataset(X_all, label=y_all, weight=w_all, categorical_feature=cat_feat)
    model_full = lgb.train(params, dall, num_boost_round=model.best_iteration)

    test_pred = model_full.predict(X_te)
    submission = pd.DataFrame({'id': test_feat['id'], 'prediction': test_pred})
    out_path = DATA_DIR / 'submission_lgbm_lags.csv'
    submission.to_csv(out_path, index=False)
    print(f"\nSaved {len(submission)} rows → {out_path}")


if __name__ == '__main__':
    main()
