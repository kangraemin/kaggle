"""Trial 007: all features combined"""
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
import numpy as np
from utils import *

def main():
    train, test = load_data()
    combined = combine_train_test(train, test)

    print("Engineering features...")
    combined = add_base_lags(combined, lags=(1, 2, 3, 5, 10, 20, 50, 100))
    combined = add_rolling(combined, windows=(5, 10, 20, 50))
    combined = add_ewm(combined, spans=(3, 5, 10, 20, 50))
    combined = add_trend(combined)
    combined = add_cross_horizon(combined)

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
    print(f"\n[Trial 007] Val score: {score:.6f}")

    imp = __import__('pandas').Series(
        model.feature_importance('gain'), index=X_tr.columns
    ).sort_values(ascending=False)
    print("\nTop 20 feature importance:")
    print(imp.head(20))

    model_full = retrain_full(prepare_X(train_f), train_f.y_target.values,
                              train_f.weight.values, model.best_iteration)
    save_submission(test_f, model_full.predict(X_te), 'trial_007_all_features')

if __name__ == '__main__':
    main()
