"""Trial 006: separate model per horizon"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
import numpy as np
import pandas as pd
from utils import *

def main():
    train, test = load_data()
    combined = combine_train_test(train, test)

    print("Engineering features...")
    combined = add_base_lags(combined, lags=(1, 2, 3, 5, 10, 20))
    combined = add_rolling(combined, windows=(5, 10, 20, 50))
    combined = add_ewm(combined, spans=(5, 20, 50))
    combined = add_trend(combined)

    train_f = combined[combined['weight'].notna()].copy()
    test_f  = combined[combined['weight'].isna()].copy()

    horizons = [1, 3, 10, 25]
    val_preds_all = []
    test_preds_all = []

    for h in horizons:
        tr_h  = train_f[(train_f.horizon == h) & (train_f.ts_index <= CUTOFF)]
        val_h = train_f[(train_f.horizon == h) & (train_f.ts_index >  CUTOFF)]
        te_h  = test_f[test_f.horizon == h]

        X_tr  = prepare_X(tr_h)
        X_val = prepare_X(val_h)
        X_te  = prepare_X(te_h)

        print(f"\n--- Horizon {h} ---")
        model = train_lgbm(X_tr, tr_h.y_target.values, tr_h.weight.values,
                           X_val, val_h.y_target.values, val_h.weight.values)

        val_pred = model.predict(X_val)
        score = weighted_rmse_score(val_h.y_target.values, val_pred, val_h.weight.values)
        print(f"Horizon {h} val score: {score:.6f}")

        # Retrain on full
        all_h = train_f[train_f.horizon == h]
        model_full = retrain_full(prepare_X(all_h), all_h.y_target.values,
                                  all_h.weight.values, model.best_iteration)

        val_preds_all.append(pd.DataFrame({'idx': val_h.index, 'pred': val_pred}))
        test_preds_all.append(pd.DataFrame({'id': te_h['id'].values,
                                            'prediction': model_full.predict(X_te)}))

    # Overall val score
    val_all = train_f[train_f.ts_index > CUTOFF].copy()
    pred_map = pd.concat(val_preds_all).set_index('idx')['pred']
    val_all['pred'] = pred_map
    score_all = weighted_rmse_score(val_all.y_target.values,
                                    val_all.pred.values, val_all.weight.values)
    print(f"\n[Trial 006] Overall val score: {score_all:.6f}")

    submission = pd.concat(test_preds_all)
    out = Path(__file__).parent / 'trial_006_per_horizon.csv'
    submission.to_csv(out, index=False)
    print(f"Saved → {out}")

if __name__ == '__main__':
    main()
