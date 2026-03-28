"""Trial 031: sub_01 방식으로 복귀 + 개선
- ts_index-based val (cutoff=2880) → 고가중치 코드 전부 training에 포함
- raw features only (group stats 제외 — 고가중치 y≈0 시리즈 예측 망침)
- num_leaves=255, 3000라운드
- sub_01 (0.1499 public)에서 개선 시도
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


def prepare_X(df):
    feat_cols = [c for c in df.columns if c.startswith('feature_')]
    cat_cols = ['code', 'sub_code', 'sub_category']
    X = df[feat_cols + cat_cols + ['horizon', 'ts_index']].copy()
    for c in cat_cols:
        X[c] = X[c].astype('category')
    return X


def main():
    train = load_train()
    check_memory('after load')

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

    # full retrain
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
    X_te = prepare_X(test)

    preds = model_full.predict(X_te)

    # 고가중치 코드 예측 확인
    for code in hw_codes:
        mask = test['code'] == code
        if mask.sum() > 0:
            p = preds[mask.values]
            print(f"  {code} test pred: mean={p.mean():.6f}, max_abs={np.abs(p).max():.6f}")

    save_submission(test, preds, 'trial_031_ts_val_improved',
                    output_dir=Path(__file__).parent)


if __name__ == '__main__':
    main()
