"""
Baseline: LightGBM with raw features only (no lag features)
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path

DATA_DIR = Path(__file__).parent

def weighted_rmse_score(y_target, y_pred, w):
    denom = np.sum(w * y_target ** 2)
    ratio = np.sum(w * (y_target - y_pred) ** 2) / denom
    clipped = float(np.minimum(np.maximum(ratio, 0.0), 1.0))
    return float(np.sqrt(1.0 - clipped))


def prepare_features(df):
    feat_cols = [c for c in df.columns if c.startswith('feature_')]
    cat_cols = ['code', 'sub_code', 'sub_category']

    X = df[feat_cols + cat_cols + ['horizon', 'ts_index']].copy()
    for c in cat_cols:
        X[c] = X[c].astype('category')
    return X


def main():
    print("Loading data...")
    train = pd.read_parquet(DATA_DIR / 'train.parquet')
    test  = pd.read_parquet(DATA_DIR / 'test.parquet')

    # Time-based split (last ~20% for validation)
    cutoff = 2880
    tr  = train[train.ts_index <= cutoff]
    val = train[train.ts_index >  cutoff]
    print(f"Train: {len(tr)}, Val: {len(val)}")

    X_tr  = prepare_features(tr)
    X_val = prepare_features(val)
    X_te  = prepare_features(test)

    y_tr  = tr['y_target'].values
    y_val = val['y_target'].values
    w_tr  = tr['weight'].values
    w_val = val['weight'].values

    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr,
                         categorical_feature=['code','sub_code','sub_category'])
    dval   = lgb.Dataset(X_val, label=y_val, weight=w_val,
                         categorical_feature=['code','sub_code','sub_category'],
                         reference=dtrain)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 127,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_jobs': -1,
    }

    print("Training LightGBM...")
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=500,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
    )

    val_pred = model.predict(X_val)
    score = weighted_rmse_score(y_val, val_pred, w_val)
    print(f"\nValidation weighted_rmse_score: {score:.6f}")

    # Full train → predict test
    print("\nRetraining on full train...")
    X_all = prepare_features(train)
    y_all = train['y_target'].values
    w_all = train['weight'].values
    dall  = lgb.Dataset(X_all, label=y_all, weight=w_all,
                        categorical_feature=['code','sub_code','sub_category'])
    model_full = lgb.train(params, dall, num_boost_round=model.best_iteration)

    test_pred = model_full.predict(X_te)

    submission = pd.DataFrame({'id': test['id'], 'prediction': test_pred})
    out_path = DATA_DIR / 'submission_lgbm_baseline.csv'
    submission.to_csv(out_path, index=False)
    print(f"Saved {len(submission)} rows → {out_path}")
    print(submission.head())


if __name__ == '__main__':
    main()
