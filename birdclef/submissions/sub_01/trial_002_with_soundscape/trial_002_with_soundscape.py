"""Trial 002: Perch + LightGBM + Soundscape 데이터 추가
- trial_001 (audio only, AUC 0.8375) 대비
- soundscape 1,478개 추가 (multi-label)
- 학습 데이터 35,549 → 37,027
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pathlib import Path

DATA = Path(__file__).parent.parent.parent.parent / 'data'
EMB = DATA / 'perch_embeddings' / 'perch_embeddings'


def main():
    print("Loading embeddings...")
    X_audio = np.load(EMB / 'audio_embeddings.npy')
    meta_audio = pd.read_csv(EMB / 'audio_metadata.csv')
    X_ss = np.load(EMB / 'ss_embeddings.npy')
    meta_ss = pd.read_csv(EMB / 'ss_metadata.csv')

    sample = pd.read_csv(DATA / 'sample_submission.csv')
    species_cols = [c for c in sample.columns if c != 'row_id']
    label_to_idx = {s: i for i, s in enumerate(species_cols)}

    # Audio labels (single label)
    y_audio = np.zeros((len(meta_audio), len(species_cols)), dtype=np.float32)
    for i, row in meta_audio.iterrows():
        label = str(row['label'])
        if label in label_to_idx:
            y_audio[i, label_to_idx[label]] = 1.0

    # Soundscape labels (multi-label, 세미콜론 구분)
    y_ss = np.zeros((len(meta_ss), len(species_cols)), dtype=np.float32)
    for i, row in meta_ss.iterrows():
        labels = str(row['label']).split(';')
        for label in labels:
            label = label.strip()
            if label in label_to_idx:
                y_ss[i, label_to_idx[label]] = 1.0

    # 합치기
    X_all = np.vstack([X_audio, X_ss])
    y_all = np.vstack([y_audio, y_ss])
    source = np.array(['audio'] * len(X_audio) + ['soundscape'] * len(X_ss))
    main_labels = list(meta_audio['label'].values) + list(meta_ss['label'].apply(lambda x: str(x).split(';')[0]).values)

    print(f"Combined: {X_all.shape[0]} samples ({len(X_audio)} audio + {len(X_ss)} soundscape)")
    print(f"Species: {len(species_cols)}")

    # 5-Fold CV
    print("\nTraining LightGBM...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros_like(y_all)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, main_labels)):
        X_tr, X_va = X_all[tr_idx], X_all[va_idx]
        fold_aucs = []

        for j, sp in enumerate(species_cols):
            y_tr, y_va = y_all[tr_idx, j], y_all[va_idx, j]
            if y_tr.sum() == 0:
                oof_preds[va_idx, j] = 0.0
                continue

            model = lgb.LGBMClassifier(
                n_estimators=200, learning_rate=0.05, num_leaves=31,
                min_child_samples=5, random_state=42, verbose=-1,
                scale_pos_weight=max(1, (y_tr == 0).sum() / max(1, (y_tr == 1).sum()))
            )
            model.fit(X_tr, y_tr)
            pred = model.predict_proba(X_va)[:, 1]
            oof_preds[va_idx, j] = pred

            if y_va.sum() > 0:
                auc = roc_auc_score(y_va, pred)
                fold_aucs.append(auc)

        print(f"  Fold {fold}: AUC={np.mean(fold_aucs):.4f} ({len(fold_aucs)} species)")

    # Overall
    valid_aucs = []
    for j in range(len(species_cols)):
        if y_all[:, j].sum() > 0:
            auc = roc_auc_score(y_all[:, j], oof_preds[:, j])
            valid_aucs.append(auc)
    print(f"\nOverall OOF macro AUC: {np.mean(valid_aucs):.4f} ({len(valid_aucs)} species)")


if __name__ == '__main__':
    main()
