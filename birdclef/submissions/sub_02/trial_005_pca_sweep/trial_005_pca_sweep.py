"""Trial 005: PCA 차원별 XGBoost 성능 비교
- PCA dims: 64, 128, 256, 512, 1024, None(1536)
- XGBoost 단일 모델로 통일
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pathlib import Path
import time
import json

DATA = Path(__file__).parent.parent.parent.parent / 'data'
EMB = DATA / 'perch_embeddings' / 'perch_embeddings'


def run_xgb_cv(X_all, y_all, main_labels, species_cols, pca_dim_label):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros_like(y_all)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, main_labels)):
        X_tr, X_va = X_all[tr_idx], X_all[va_idx]
        for j, sp in enumerate(species_cols):
            y_tr, y_va = y_all[tr_idx, j], y_all[va_idx, j]
            if y_tr.sum() < 3:
                continue
            spw = max(1, (y_tr == 0).sum() / max(1, (y_tr == 1).sum()))
            m = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=4,
                                   random_state=42, verbosity=0, scale_pos_weight=spw)
            m.fit(X_tr, y_tr)
            oof[va_idx, j] = m.predict_proba(X_va)[:, 1]

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

    print(f"Data: {X_all_raw.shape[0]} samples, {X_all_raw.shape[1]} features")

    pca_dims = [64, 128, 256, 512, 1024, None]
    results = {}

    for dim in pca_dims:
        t0 = time.time()
        label = str(dim) if dim else "1536(no PCA)"

        if dim:
            pca = PCA(n_components=dim, random_state=42)
            X_all = pca.fit_transform(X_all_raw)
            expl_var = pca.explained_variance_ratio_.sum()
        else:
            X_all = X_all_raw
            expl_var = 1.0

        auc = run_xgb_cv(X_all, y_all, main_labels, species_cols, label)
        elapsed = time.time() - t0
        results[label] = {"auc": round(auc, 4), "explained_var": round(expl_var, 4), "elapsed_sec": round(elapsed)}
        print(f"  PCA {label}: AUC={auc:.4f}, var={expl_var:.4f}, {elapsed:.0f}s")

    print("\n=== Summary ===")
    for k, v in sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True):
        print(f"  PCA {k}: AUC={v['auc']:.4f} ({v['elapsed_sec']}s)")

    out_path = Path(__file__).parent / 'results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
