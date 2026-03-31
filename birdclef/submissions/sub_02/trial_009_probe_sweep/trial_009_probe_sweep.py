"""Trial 009: Probe hyperparameter sweep
- PCA dim: 16, 32, 64, 96
- Model: LR (C sweep) + MLP(128)
- 0.916 노트북은 PCA 16, 0.924는 MLP(128) 사용
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import time
import json

DATA = Path(__file__).parent.parent.parent.parent / 'data'
EMB = DATA / 'perch_embeddings' / 'perch_embeddings'


def run_cv(X_all, y_all, main_labels, species_cols, model_fn, label):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros_like(y_all)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, main_labels)):
        for j in range(len(species_cols)):
            y_tr = y_all[tr_idx, j]
            if y_tr.sum() == 0:
                continue
            m = model_fn()
            m.fit(X_all[tr_idx], y_tr)
            oof[va_idx, j] = m.predict_proba(X_all[va_idx])[:, 1]
    valid_aucs = []
    for j in range(len(species_cols)):
        if y_all[:, j].sum() > 0:
            valid_aucs.append(roc_auc_score(y_all[:, j], oof[:, j]))
    return np.mean(valid_aucs)


def main():
    print("Loading...")
    X_audio = np.load(EMB / 'audio_embeddings.npy')
    meta_audio = pd.read_csv(EMB / 'audio_metadata.csv')
    X_ss = np.load(EMB / 'ss_embeddings.npy')
    meta_ss = pd.read_csv(EMB / 'ss_metadata.csv')
    sample = pd.read_csv(DATA / 'sample_submission.csv')
    species_cols = [c for c in sample.columns if c != 'row_id']
    label_to_idx = {s: i for i, s in enumerate(species_cols)}

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

    # StandardScaler (0.912 노트북도 사용)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all_raw)

    results = {}

    # PCA dim × C sweep (LR)
    for pca_dim in [16, 32, 64, 96]:
        pca = PCA(n_components=pca_dim, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        expl = pca.explained_variance_ratio_.sum()

        for C in [0.1, 0.25, 0.5, 1.0, 2.0]:
            t0 = time.time()
            auc = run_cv(X_pca, y_all, main_labels, species_cols,
                         lambda: LogisticRegression(C=C, max_iter=1000, solver='lbfgs', random_state=42),
                         f'LR_pca{pca_dim}_C{C}')
            elapsed = time.time() - t0
            key = f'LR_pca{pca_dim}_C{C}'
            results[key] = {'auc': round(auc, 6), 'pca': pca_dim, 'C': C, 'model': 'LR',
                           'explained_var': round(expl, 4), 'elapsed': round(elapsed)}
            print(f'  {key}: AUC={auc:.6f} ({elapsed:.0f}s)')

    # MLP sweep
    for pca_dim in [32, 64]:
        pca = PCA(n_components=pca_dim, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        t0 = time.time()
        auc = run_cv(X_pca, y_all, main_labels, species_cols,
                     lambda: MLPClassifier(hidden_layer_sizes=(128,), activation='relu',
                                           max_iter=300, early_stopping=True,
                                           learning_rate_init=0.001, alpha=0.01, random_state=42),
                     f'MLP128_pca{pca_dim}')
        elapsed = time.time() - t0
        key = f'MLP128_pca{pca_dim}'
        results[key] = {'auc': round(auc, 6), 'pca': pca_dim, 'model': 'MLP(128)',
                       'elapsed': round(elapsed)}
        print(f'  {key}: AUC={auc:.6f} ({elapsed:.0f}s)')

    # Summary
    print('\n=== Top 10 ===')
    sorted_r = sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True)
    for k, v in sorted_r[:10]:
        print(f'  {k}: AUC={v["auc"]:.6f} ({v["elapsed"]}s)')

    out_path = Path(__file__).parent / 'results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
