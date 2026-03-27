"""Trial 008: cross-horizon lag_1, lag_2, lag_3"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from utils import *


def add_cross_horizon_extended(df):
    """Add other horizons' lag_1, lag_2, lag_3 at same (code, sub_code, sub_category, ts_index)."""
    base = df[['code', 'sub_code', 'sub_category', 'horizon', 'ts_index', 'lag_1', 'lag_2', 'lag_3']].copy()
    for h in [1, 3, 10, 25]:
        src = base[base.horizon == h][['code', 'sub_code', 'sub_category', 'ts_index', 'lag_1', 'lag_2', 'lag_3']]
        src = src.rename(columns={
            'lag_1': f'cross_h{h}_lag1',
            'lag_2': f'cross_h{h}_lag2',
            'lag_3': f'cross_h{h}_lag3',
        })
        df = df.merge(src, on=['code', 'sub_code', 'sub_category', 'ts_index'], how='left')
    return df


def main():
    train, test = load_data()
    combined = combine_train_test(train, test)

    print("Engineering features...")
    combined = add_base_lags(combined, lags=(1, 2, 3, 5, 10, 20))
    combined = add_rolling(combined, windows=(5, 10, 20))
    combined = add_trend(combined)
    combined = add_cross_horizon_extended(combined)

    train_f = combined[combined['weight'].notna()].copy()
    test_f  = combined[combined['weight'].isna()].copy()

    tr  = train_f[train_f.ts_index <= CUTOFF]
    val = train_f[train_f.ts_index >  CUTOFF]
    print(f"Train: {len(tr)}, Val: {len(val)}")

    X_tr, X_val, X_te = prepare_X(tr), prepare_X(val), prepare_X(test_f)

    model = train_lgbm(X_tr, tr.y_target.values, tr.weight.values,
                       X_val, val.y_target.values, val.weight.values)

    val_pred = model.predict(X_val)
    score = weighted_rmse_score(val.y_target.values, val_pred, val.weight.values)
    print(f"\n[Trial 008] Val score: {score:.6f}")

    import pandas as pd
    imp = pd.Series(model.feature_importance('gain'), index=X_tr.columns).sort_values(ascending=False)
    print("\nTop 15 feature importance:")
    print(imp.head(15))

    model_full = retrain_full(prepare_X(train_f), train_f.y_target.values,
                              train_f.weight.values, model.best_iteration)
    save_submission(test_f, model_full.predict(X_te), 'trial_008_cross_h_lags',
                    output_dir=Path(__file__).parent)

if __name__ == '__main__':
    main()
