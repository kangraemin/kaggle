"""Trial 012: raw features + target encoding only (no lag, guaranteed valid on test)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from utils import *


def add_target_encoding(combined):
    train_mask = combined['weight'].notna()
    stats = combined[train_mask].groupby(KEY)['y_target'].agg(
        series_mean='mean',
        series_std='std',
        series_median='median',
    ).reset_index()
    return combined.merge(stats, on=KEY, how='left')


def main():
    train, test = load_data()
    combined = combine_train_test(train, test)

    print("Engineering features (no lag)...")
    combined = add_target_encoding(combined)

    train_f = combined[combined['weight'].notna()].copy()
    test_f  = combined[combined['weight'].isna()].copy()

    tr  = train_f[train_f.ts_index <= CUTOFF]
    val = train_f[train_f.ts_index >  CUTOFF]
    print(f"Train: {len(tr)}, Val: {len(val)}")

    # NaN check on test
    X_te_check = prepare_X(test_f)
    for col in X_te_check.columns:
        nan_pct = X_te_check[col].isna().mean()
        if nan_pct > 0.01:
            print(f"WARNING: {col} NaN {nan_pct*100:.1f}%")

    X_tr, X_val, X_te = prepare_X(tr), prepare_X(val), prepare_X(test_f)

    model = train_lgbm(X_tr, tr.y_target.values, tr.weight.values,
                       X_val, val.y_target.values, val.weight.values)

    val_pred = model.predict(X_val)
    score = weighted_rmse_score(val.y_target.values, val_pred, val.weight.values)
    print(f"\n[Trial 012] Val score: {score:.6f}")

    import pandas as pd
    imp = pd.Series(model.feature_importance('gain'), index=X_tr.columns).sort_values(ascending=False)
    print("\nTop 15 feature importance:")
    print(imp.head(15))

    model_full = retrain_full(prepare_X(train_f), train_f.y_target.values,
                              train_f.weight.values, model.best_iteration)
    save_submission(test_f, model_full.predict(X_te), 'trial_012_no_lag',
                    output_dir=Path(__file__).parent)


if __name__ == '__main__':
    main()
