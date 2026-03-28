"""Trial 036: known series recursive prediction
- 지금은 모든 test step에 동일한 last_y_1/2/3 사용
- recursive: step 1 예측 → step 2의 last_y_1로 업데이트
- AR(1)=0.53 구조를 제대로 활용
- val은 여전히 ts_index split (검증용)
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


CLIP_BOUNDS = {}  # train에서 계산한 feature 범위


def compute_clip_bounds(df):
    """train 기준 feature 클리핑 범위 계산."""
    feat_cols = [c for c in df.columns if c.startswith('feature_')]
    bounds = {}
    for col in feat_cols:
        bounds[col] = (float(df[col].quantile(0.001)), float(df[col].quantile(0.999)))
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
    hi = float(train_df['y_target'].quantile(0.999))

    sorted_tr = train_df.sort_values(KEY + ['ts_index'])
    grp = sorted_tr.groupby(KEY)['y_target']

    stats = grp.agg(
        series_mean='mean',
        series_std='std',
    ).reset_index()
    stats['series_mean'] = stats['series_mean'].clip(lower=lo, upper=hi)
    stats['series_std']  = stats['series_std'].clip(lower=0, upper=(hi - lo))

    # lag 1~3
    last3 = sorted_tr.groupby(KEY)['y_target'].apply(
        lambda x: x.iloc[-3:].values if len(x) >= 3 else [np.nan]*(3-len(x)) + list(x.values)
    )
    lag_df = pd.DataFrame(last3.tolist(), index=last3.index,
                          columns=['last_y_3', 'last_y_2', 'last_y_1'])
    for col in ['last_y_1', 'last_y_2', 'last_y_3']:
        lag_df[col] = lag_df[col].clip(lower=lo, upper=hi)
    lag_df = lag_df.reset_index()

    result = stats.merge(lag_df, on=KEY, how='left')
    df = df.merge(result, on=KEY, how='left')  # unknown → NaN
    return df


def prepare_X(df):
    feat_cols = [c for c in df.columns if c.startswith('feature_')]
    cat_cols = ['code', 'sub_code', 'sub_category']
    series_feats = [c for c in ['last_y_1', 'last_y_2', 'last_y_3', 'series_mean', 'series_std']
                    if c in df.columns]
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

    # 시계열 features 추가 (known series만)
    print("Adding series features...")
    train = add_series_features(train, train)

    # ts_index-based val (sub_01 방식)
    tr  = train[train.ts_index <= CUTOFF]
    val = train[train.ts_index >  CUTOFF]
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
    model = lgb.train(PARAMS, dtrain, num_boost_round=3000,
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
    test = add_series_features(test, train_full_for_feats)
    lo = float(train_full_for_feats['y_target'].quantile(0.001))
    hi = float(train_full_for_feats['y_target'].quantile(0.999))
    known_cnt = test['last_y_1'].notna().sum()
    print(f"Test known series: {known_cnt:,} ({known_cnt/len(test)*100:.1f}%)")
    del train_full_for_feats
    gc.collect()

    # Recursive prediction for known series, batch for unknown
    print("Predicting (recursive for known, batch for unknown)...")
    test_sorted = test.sort_values(KEY + ['ts_index']).reset_index(drop=True)
    all_preds = np.zeros(len(test_sorted))

    unknown_idx = test_sorted['last_y_1'].isna()
    # unknown: batch
    X_unk = prepare_X(test_sorted[unknown_idx])
    all_preds[unknown_idx.values] = model_full.predict(X_unk)

    # known: recursive
    known_df = test_sorted[~unknown_idx].copy()
    for key_val, grp in known_df.groupby(KEY, sort=False):
        idx = grp.index.tolist()
        buf = [grp['last_y_3'].iloc[0], grp['last_y_2'].iloc[0], grp['last_y_1'].iloc[0]]
        for i, (pos, row) in enumerate(grp.iterrows()):
            row_df = grp.loc[[pos]].copy()
            row_df['last_y_1'] = float(buf[-1]) if not np.isnan(buf[-1]) else np.nan
            row_df['last_y_2'] = float(buf[-2]) if len(buf)>=2 and not np.isnan(buf[-2]) else np.nan
            row_df['last_y_3'] = float(buf[-3]) if len(buf)>=3 and not np.isnan(buf[-3]) else np.nan
            pred = float(model_full.predict(prepare_X(row_df))[0])
            pred_clipped = float(np.clip(pred, lo, hi))
            all_preds[idx[i]] = pred_clipped
            buf.append(pred_clipped)

    test_sorted['_pred'] = all_preds
    preds_map = test_sorted.set_index('id')['_pred']
    preds = test['id'].map(preds_map).values

    # 고가중치 코드 예측 확인
    for code in hw_codes:
        mask = test['code'] == code
        if mask.sum() > 0:
            p = preds[mask.values]
            print(f"  {code} test pred: mean={p.mean():.6f}, max_abs={np.abs(p).max():.6f}")

    save_submission(test, preds, 'trial_036_ts_val_improved',
                    output_dir=Path(__file__).parent)


if __name__ == '__main__':
    main()
