"""Trial 012: raw features + target encoding only (no lag, guaranteed valid on test)
Memory-optimized: train → train+free → test 순차 처리
"""
import gc
import sys
import psutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from utils import *

MEM_LIMIT_GB = 2.0

def check_memory(label=''):
    avail = psutil.virtual_memory().available / 1024**3
    print(f"[MEM] {label}: {avail:.1f} GB free", flush=True)
    if avail < MEM_LIMIT_GB:
        print(f"[MEM] 위험 — 강제 종료", flush=True)
        sys.exit(1)


def main():
    # 1) Load train only
    train = load_train()

    # 2) Compute target encoding stats from train
    stats = train.groupby(KEY)['y_target'].agg(
        series_mean='mean', series_std='std', series_median='median',
    ).reset_index()

    # 3) Merge stats, split tr/val
    train = train.merge(stats, on=KEY, how='left')
    tr  = train[train.ts_index <= CUTOFF]
    val = train[train.ts_index >  CUTOFF]
    print(f"Train: {len(tr)}, Val: {len(val)}")

    # 4) Train
    X_tr, X_val = prepare_X(tr), prepare_X(val)
    model = train_lgbm(X_tr, tr.y_target.values, tr.weight.values,
                       X_val, val.y_target.values, val.weight.values)

    val_pred = model.predict(X_val)
    score = weighted_rmse_score(val.y_target.values, val_pred, val.weight.values)
    print(f"\n[Trial 012] Val score: {score:.6f}")

    imp = pd.Series(model.feature_importance('gain'), index=X_tr.columns).sort_values(ascending=False)
    print("\nTop 15 feature importance:")
    print(imp.head(15))

    # 5) Free val/tr before retrain to reduce peak memory
    best_iter = model.best_iteration
    del tr, val, X_tr, X_val, model
    gc.collect()
    check_memory('after val, before retrain')

    # 6) Retrain full
    y_full = train.y_target.values
    w_full = train.weight.values
    X_full = prepare_X(train)
    del train
    gc.collect()
    model_full = retrain_full(X_full, y_full, w_full, best_iter)
    del X_full, y_full, w_full
    gc.collect()
    check_memory('after retrain, before test')

    test = load_test()
    test = test.merge(stats, on=KEY, how='left')

    X_te = prepare_X(test)
    for col in X_te.columns:
        nan_pct = X_te[col].isna().mean()
        if nan_pct > 0.01:
            print(f"WARNING: {col} NaN {nan_pct*100:.1f}%")

    save_submission(test, model_full.predict(X_te), 'trial_012_no_lag',
                    output_dir=Path(__file__).parent)


if __name__ == '__main__':
    main()
