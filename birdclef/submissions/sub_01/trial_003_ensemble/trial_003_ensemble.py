"""Trial 003: LightGBM + XGBoost 앙상블
- trial_002 (AUC 0.8731) 대비
- LightGBM + XGBoost 평균 → diversity 효과
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pathlib import Path

DATA = Path(__file__).parent.parent.parent.parent / 'data'
EMB = DATA / 'perch_embeddings' / 'perch_embeddings'


def main():
    print("Loading...")
    X_audio = np.load(EMB / 'audio_embeddings.npy')
    meta_audio = pd.read_csv(EMB / 'audio_metadata.csv')
    X_ss = np.load(EMB / 'ss_embeddings.npy')
    meta_ss = pd.read_csv(EMB / 'ss_metadata.csv')
    sample = pd.read_csv(DATA / 'sample_submission.csv')
    species_cols = [c for c in sample.columns if c != 'row_id']
    label_to_idx = {s: i for i, s in enumerate(species_cols)}

    # Labels
    y_audio = np.zeros((len(X_audio), len(species_cols)), dtype=np.float32)
    for i, row in meta_audio.iterrows():
        if str(row['label']) in label_to_idx:
            y_audio[i, label_to_idx[str(row['label'])]] = 1.0

    y_ss = np.zeros((len(X_ss), len(species_cols)), dtype=np.float32)
    for i, row in meta_ss.iterrows():
        for label in str(row['label']).split(';'):
            if label.strip() in label_to_idx:
                y_ss[i, label_to_idx[label.strip()]] = 1.0

    X_all = np.vstack([X_audio, X_ss])
    y_all = np.vstack([y_audio, y_ss])
    main_labels = list(meta_audio['label'].values) + list(meta_ss['label'].apply(lambda x: str(x).split(';')[0]).values)

    print(f"Data: {X_all.shape[0]} samples, {len(species_cols)} species")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_lgb = np.zeros_like(y_all)
    oof_xgb = np.zeros_like(y_all)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, main_labels)):
        X_tr, X_va = X_all[tr_idx], X_all[va_idx]
        aucs = []

        for j, sp in enumerate(species_cols):
            y_tr, y_va = y_all[tr_idx, j], y_all[va_idx, j]
            if y_tr.sum() == 0:
                continue

            spw = max(1, (y_tr == 0).sum() / max(1, (y_tr == 1).sum()))

            m_lgb = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
                                        min_child_samples=5, random_state=42, verbose=-1, scale_pos_weight=spw)
            m_lgb.fit(X_tr, y_tr)
            oof_lgb[va_idx, j] = m_lgb.predict_proba(X_va)[:, 1]

            m_xgb = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=4,
                                       random_state=42, verbosity=0, scale_pos_weight=spw)
            m_xgb.fit(X_tr, y_tr)
            oof_xgb[va_idx, j] = m_xgb.predict_proba(X_va)[:, 1]

            if y_va.sum() > 0:
                auc_ens = roc_auc_score(y_va, 0.5 * oof_lgb[va_idx, j] + 0.5 * oof_xgb[va_idx, j])
                aucs.append(auc_ens)

        print(f"  Fold {fold}: ensemble AUC={np.mean(aucs):.4f}")

    oof_ens = 0.5 * oof_lgb + 0.5 * oof_xgb
    valid_aucs = []
    for j in range(len(species_cols)):
        if y_all[:, j].sum() > 0:
            valid_aucs.append(roc_auc_score(y_all[:, j], oof_ens[:, j]))

    # 개별 모델 vs 앙상블
    lgb_aucs = [roc_auc_score(y_all[:, j], oof_lgb[:, j]) for j in range(len(species_cols)) if y_all[:, j].sum() > 0]
    xgb_aucs = [roc_auc_score(y_all[:, j], oof_xgb[:, j]) for j in range(len(species_cols)) if y_all[:, j].sum() > 0]

    print(f"\nLightGBM only:  {np.mean(lgb_aucs):.4f}")
    print(f"XGBoost only:   {np.mean(xgb_aucs):.4f}")
    print(f"Ensemble (0.5): {np.mean(valid_aucs):.4f}")


if __name__ == '__main__':
    main()
