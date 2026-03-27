"""Trial 014: 계층적 target encoding + ts_index×horizon 상호작용
- 계층적 encoding: series → sub_category×horizon → horizon (NaN fallback)
- ts_index × horizon 상호작용 feature 추가
- feature_w/x/y/z NaN은 LightGBM에 위임 (ts_index 4071~4360 구조적 결측)
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


def add_hierarchical_target_encoding(combined):
    """series → sub_category×horizon → horizon 순으로 fallback target encoding."""
    train_mask = combined['weight'].notna()
    train = combined[train_mask]

    # Level 1: 시리즈 레벨 (code, sub_code, sub_category, horizon)
    s1 = train.groupby(KEY)['y_target'].agg(
        series_mean='mean', series_std='std', series_median='median'
    ).reset_index()

    # Level 2: sub_category × horizon
    s2 = train.groupby(['sub_category', 'horizon'])['y_target'].agg(
        subcat_mean='mean', subcat_std='std', subcat_median='median'
    ).reset_index()

    # Level 3: horizon
    s3 = train.groupby(['horizon'])['y_target'].agg(
        horizon_mean='mean', horizon_std='std', horizon_median='median'
    ).reset_index()

    combined = combined.merge(s1, on=KEY, how='left')
    combined = combined.merge(s2, on=['sub_category', 'horizon'], how='left')
    combined = combined.merge(s3, on=['horizon'], how='left')

    # fallback: series NaN이면 subcat으로, subcat NaN이면 horizon으로
    for stat in ['mean', 'std', 'median']:
        combined[f'enc_{stat}'] = (
            combined[f'series_{stat}']
            .fillna(combined[f'subcat_{stat}'])
            .fillna(combined[f'horizon_{stat}'])
        )

    # 원본 컬럼 정리
    drop_cols = [f'series_{s}' for s in ['mean','std','median']] + \
                [f'subcat_{s}'  for s in ['mean','std','median']] + \
                [f'horizon_{s}' for s in ['mean','std','median']]
    combined = combined.drop(columns=drop_cols)

    return combined


def add_ts_horizon_interactions(df):
    """ts_index × horizon 상호작용."""
    df['ts_x_horizon'] = df['ts_index'] * df['horizon']
    df['ts_mod_horizon'] = df['ts_index'] % df['horizon'].replace(0, 1)
    return df


def get_feature_cols_014(df):
    raw = [c for c in df.columns if c.startswith('feature_')]
    enc = [c for c in df.columns if c.startswith('enc_')]
    inter = ['ts_x_horizon', 'ts_mod_horizon']
    return raw + enc + inter


def prepare_X_014(df):
    feat_cols = get_feature_cols_014(df)
    cat_cols = ['code', 'sub_code', 'sub_category']
    X = df[feat_cols + cat_cols + ['horizon', 'ts_index']].copy()
    for c in cat_cols:
        X[c] = X[c].astype('category')
    return X


def main():
    train, test = load_data()
    combined = combine_train_test(train, test)
    del train, test
    gc.collect()
    check_memory('after load')

    print("Engineering features...")
    combined = add_hierarchical_target_encoding(combined)
    combined = add_ts_horizon_interactions(combined)

    train_f = combined[combined['weight'].notna()].copy()
    test_f  = combined[combined['weight'].isna()].copy()
    del combined
    gc.collect()
    check_memory('after feature engineering')

    # NaN check on test
    X_te = prepare_X_014(test_f)
    for col in X_te.columns:
        nan_pct = X_te[col].isna().mean()
        if nan_pct > 0.05:
            print(f"WARNING: {col} NaN {nan_pct*100:.1f}%")

    tr  = train_f[train_f.ts_index <= CUTOFF]
    val = train_f[train_f.ts_index >  CUTOFF]
    print(f"Train: {len(tr)}, Val: {len(val)}")

    X_tr  = prepare_X_014(tr)
    X_val = prepare_X_014(val)
    model = train_lgbm(X_tr, tr.y_target.values, tr.weight.values,
                       X_val, val.y_target.values, val.weight.values)

    val_pred = model.predict(X_val)
    score = weighted_rmse_score(val.y_target.values, val_pred, val.weight.values)
    print(f"\n[Trial 014] Val score: {score:.6f}")

    imp = pd.Series(model.feature_importance('gain'), index=X_tr.columns).sort_values(ascending=False)
    print("\nTop 15 feature importance:")
    print(imp.head(15))

    best_iter = model.best_iteration
    del tr, val, X_tr, X_val, model
    gc.collect()
    check_memory('after val, before retrain')

    y_full = train_f.y_target.values
    w_full = train_f.weight.values
    X_full = prepare_X_014(train_f)
    del train_f
    gc.collect()
    model_full = retrain_full(X_full, y_full, w_full, best_iter)
    del X_full, y_full, w_full
    gc.collect()
    check_memory('after retrain')

    save_submission(test_f, model_full.predict(X_te), 'trial_014_hier_enc',
                    output_dir=Path(__file__).parent)


if __name__ == '__main__':
    main()
