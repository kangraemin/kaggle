"""Trial 004: Logistic Regression + PCA 64
- Discussion 표준 접근법: Perch 임베딩 → PCA 64 → LogisticRegression
- 상위권 노트북(0.91+)이 전부 이 패턴 사용
- XGBoost 대비 빠르고 일반화 좋을 가능성
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pathlib import Path
import time
import json

DATA = Path(__file__).parent.parent.parent.parent / 'data'
EMB = DATA / 'perch_embeddings' / 'perch_embeddings'


def main():
    t0 = time.time()
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

    X_all_raw = np.vstack([X_audio, X_ss])
    y_all = np.vstack([y_audio, y_ss])
    main_labels = list(meta_audio['label'].values) + list(meta_ss['label'].apply(lambda x: str(x).split(';')[0]).values)

    # PCA 1536 → 64
    print("PCA 1536 → 64...")
    pca = PCA(n_components=64, random_state=42)
    X_all = pca.fit_transform(X_all_raw)
    print(f"Data: {X_all.shape[0]} samples, {X_all.shape[1]} features, {len(species_cols)} species")
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros_like(y_all)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, main_labels)):
        X_tr, X_va = X_all[tr_idx], X_all[va_idx]
        fold_aucs = []

        for j, sp in enumerate(species_cols):
            y_tr, y_va = y_all[tr_idx, j], y_all[va_idx, j]
            if y_tr.sum() == 0:
                continue

            m = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver='lbfgs')
            m.fit(X_tr, y_tr)
            oof[va_idx, j] = m.predict_proba(X_va)[:, 1]

            if y_va.sum() > 0:
                fold_aucs.append(roc_auc_score(y_va, oof[va_idx, j]))

        print(f"  Fold {fold}: AUC={np.mean(fold_aucs):.4f}")

    valid_aucs = []
    for j in range(len(species_cols)):
        if y_all[:, j].sum() > 0:
            valid_aucs.append(roc_auc_score(y_all[:, j], oof[:, j]))

    oof_auc = np.mean(valid_aucs)
    elapsed = time.time() - t0
    print(f"\nOOF AUC: {oof_auc:.4f} ({elapsed:.0f}s)")

    results = {
        "trial": "004_logreg_pca64",
        "val_score": round(oof_auc, 4),
        "model": "LogisticRegression",
        "pca_dim": 64,
        "pca_explained_var": round(float(pca.explained_variance_ratio_.sum()), 4),
        "n_species": len(species_cols),
        "n_samples": X_all.shape[0],
        "elapsed_sec": round(elapsed),
        "notes": "Discussion standard: Perch + PCA 64 + LR"
    }
    out_path = Path(__file__).parent / 'results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
