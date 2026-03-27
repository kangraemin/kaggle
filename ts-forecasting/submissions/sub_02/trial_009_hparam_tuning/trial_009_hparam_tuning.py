"""Trial 009: hyperparameter tuning (num_leaves 줄이고 lr 낮춤)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
import lightgbm as lgb
from utils import *


def train_lgbm_tuned(X_tr, y_tr, w_tr, X_val, y_val, w_val):
    cat = ['code', 'sub_code', 'sub_category']
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 127,
        'learning_rate': 0.03,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 50,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'verbose': -1,
        'n_jobs': -1,
    }
    dtrain = lgb.Dataset(X_tr,  label=y_tr,  weight=w_tr,  categorical_feature=cat)
    dval   = lgb.Dataset(X_val, label=y_val, weight=w_val, categorical_feature=cat, reference=dtrain)
    return lgb.train(
        params, dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )


def main():
    train, test = load_data()
    combined = combine_train_test(train, test)

    print("Engineering features...")
    combined = add_base_lags(combined, lags=(1, 2, 3, 5, 10, 20))
    combined = add_rolling(combined, windows=(5, 10, 20))
    combined = add_trend(combined)
    combined = add_cross_horizon(combined)

    train_f = combined[combined['weight'].notna()].copy()
    test_f  = combined[combined['weight'].isna()].copy()

    tr  = train_f[train_f.ts_index <= CUTOFF]
    val = train_f[train_f.ts_index >  CUTOFF]
    print(f"Train: {len(tr)}, Val: {len(val)}")

    X_tr, X_val, X_te = prepare_X(tr), prepare_X(val), prepare_X(test_f)

    model = train_lgbm_tuned(X_tr, tr.y_target.values, tr.weight.values,
                             X_val, val.y_target.values, val.weight.values)

    val_pred = model.predict(X_val)
    score = weighted_rmse_score(val.y_target.values, val_pred, val.weight.values)
    print(f"\n[Trial 009] Val score: {score:.6f}")

    import lightgbm as lgb2
    cat = ['code', 'sub_code', 'sub_category']
    params = {
        'objective': 'regression', 'metric': 'rmse',
        'num_leaves': 127, 'learning_rate': 0.03,
        'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
        'min_child_samples': 50, 'lambda_l1': 0.1, 'lambda_l2': 0.1,
        'verbose': -1, 'n_jobs': -1,
    }
    dall = lgb2.Dataset(prepare_X(train_f), label=train_f.y_target.values,
                        weight=train_f.weight.values, categorical_feature=cat)
    model_full = lgb2.train(params, dall, num_boost_round=model.best_iteration)
    save_submission(test_f, model_full.predict(X_te), 'trial_009_hparam_tuning',
                    output_dir=Path(__file__).parent)

if __name__ == '__main__':
    main()
