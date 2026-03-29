"""Trial 001: Perch embedding + LightGBM
- 사전 추출된 Perch v2 임베딩 (1536차원) 사용
- LightGBM으로 종별 이진 분류 (OVR)
- GPU 필요 없음, 로컬에서 실행 가능
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
    # 1) 데이터 로드
    print("Loading embeddings...")
    X_audio = np.load(EMB / 'audio_embeddings.npy')
    meta_audio = pd.read_csv(EMB / 'audio_metadata.csv')

    X_ss = np.load(EMB / 'ss_embeddings.npy')
    meta_ss = pd.read_csv(EMB / 'ss_metadata.csv')

    sample = pd.read_csv(DATA / 'sample_submission.csv')
    species_cols = [c for c in sample.columns if c != 'row_id']

    print(f"Audio: {X_audio.shape}, Soundscape: {X_ss.shape}")
    print(f"Species: {len(species_cols)}")

    # 2) 학습 데이터: audio embeddings + labels
    # multi-label: 각 종별로 있는지 없는지
    y = np.zeros((len(meta_audio), len(species_cols)), dtype=np.float32)
    label_to_idx = {s: i for i, s in enumerate(species_cols)}

    for i, row in meta_audio.iterrows():
        label = str(row['label'])
        if label in label_to_idx:
            y[i, label_to_idx[label]] = 1.0

    print(f"Positive labels per species: min={y.sum(axis=0).min():.0f}, max={y.sum(axis=0).max():.0f}, mean={y.sum(axis=0).mean():.1f}")

    # 3) 5-Fold CV로 학습
    print("\nTraining LightGBM (per-species OVR)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 가장 빈도 높은 라벨로 stratify
    main_labels = meta_audio['label'].values

    oof_preds = np.zeros_like(y)
    models = {}

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_audio, main_labels)):
        X_tr, X_va = X_audio[tr_idx], X_audio[va_idx]

        fold_aucs = []
        for j, sp in enumerate(species_cols):
            y_tr, y_va = y[tr_idx, j], y[va_idx, j]

            # 양성 샘플 없으면 스킵
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

            models[(fold, j)] = model

        mean_auc = np.mean(fold_aucs) if fold_aucs else 0
        print(f"  Fold {fold}: AUC={mean_auc:.4f} ({len(fold_aucs)} species evaluated)")

    # 전체 OOF AUC
    valid_aucs = []
    for j, sp in enumerate(species_cols):
        if y[:, j].sum() > 0:
            auc = roc_auc_score(y[:, j], oof_preds[:, j])
            valid_aucs.append(auc)
    print(f"\nOverall OOF macro AUC: {np.mean(valid_aucs):.4f} ({len(valid_aucs)} species)")

    # 4) 이건 로컬 val만. 실제 제출은 Kaggle Notebook에서 해야 함
    # (test soundscape embedding이 대회 서버에만 있음)
    print("\n⚠️ 실제 제출은 Kaggle Notebook에서 해야 합니다.")
    print("이 스크립트는 로컬 validation용입니다.")


if __name__ == '__main__':
    main()
