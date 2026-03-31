"""Trial 010: Pseudo-labeling with soundscape embeddings
- 10k soundscape Perch 임베딩을 현재 모델로 예측
- 높은 confidence 예측을 pseudo-label로 사용
- probe 학습 데이터를 708 → 수천 개로 증가

이 스크립트는 로컬에서 pseudo-label을 생성하는 용도.
실제 Kaggle 제출은 메인 노트북에서 처리.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import time
import re

DATA = Path(__file__).parent.parent.parent.parent / 'data'
EMB = DATA / 'perch_embeddings' / 'perch_embeddings'

FNAME_RE = re.compile(r'BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg')


def parse_soundscape_filename(name):
    m = FNAME_RE.match(name)
    if not m:
        return {'site': None, 'hour_utc': -1}
    _, site, _, hms = m.groups()
    return {'site': site, 'hour_utc': int(hms[:2])}


def main():
    t0 = time.time()

    # === 1. 기존 데이터 로드 ===
    sample = pd.read_csv(DATA / 'sample_submission.csv')
    species_cols = [c for c in sample.columns if c != 'row_id']
    label_to_idx = {s: i for i, s in enumerate(species_cols)}

    X_audio = np.load(EMB / 'audio_embeddings.npy')
    meta_audio = pd.read_csv(EMB / 'audio_metadata.csv')
    X_ss = np.load(EMB / 'ss_embeddings.npy')
    meta_ss = pd.read_csv(EMB / 'ss_metadata.csv')

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

    print(f'Original data: {X_all.shape[0]} samples')

    # === 2. 추출된 soundscape 임베딩 로드 ===
    # Kaggle 추출 결과 또는 로컬 추출 결과
    ss_emb_path = DATA / 'all_soundscape_embeddings'
    if not ss_emb_path.exists():
        ss_emb_path = DATA / 'soundscape_extracted'

    if not ss_emb_path.exists():
        print('추출된 soundscape 임베딩이 없습니다.')
        print('Kaggle 추출 노트북 결과를 다운로드하거나 로컬에서 추출하세요.')
        return

    X_ss_all = np.load(ss_emb_path / 'all_ss_embeddings.npy')
    meta_ss_all = pd.read_parquet(ss_emb_path / 'all_ss_meta.parquet')
    print(f'Extracted soundscape: {X_ss_all.shape[0]} windows from {meta_ss_all["filename"].nunique()} files')

    # === 3. 기존 모델로 pseudo-label 생성 ===
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    pca = PCA(n_components=64, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # 전체 데이터로 모델 학습
    print('Training models for pseudo-labeling...')
    pseudo_probs = np.zeros((len(X_ss_all), len(species_cols)), dtype=np.float32)
    X_ss_scaled = scaler.transform(X_ss_all)
    X_ss_pca = pca.transform(X_ss_scaled)

    for j in range(len(species_cols)):
        y_j = y_all[:, j]
        if y_j.sum() < 3:
            continue
        m = LogisticRegression(C=0.1, max_iter=1000, solver='lbfgs', random_state=42)
        m.fit(X_pca, y_j)
        pseudo_probs[:, j] = m.predict_proba(X_ss_pca)[:, 1]

    # === 4. Confidence 기반 필터링 ===
    # 높은 confidence (>0.8) 양성 또는 매우 낮은 (<0.05) 음성만 사용
    HIGH_CONF = 0.8
    LOW_CONF = 0.05

    pseudo_labels = np.zeros_like(pseudo_probs, dtype=np.float32)
    pseudo_mask = np.zeros(len(X_ss_all), dtype=bool)

    for j in range(len(species_cols)):
        high = pseudo_probs[:, j] >= HIGH_CONF
        pseudo_labels[high, j] = 1.0
        pseudo_mask |= high

    n_pseudo = pseudo_mask.sum()
    print(f'Pseudo-labeled windows: {n_pseudo} / {len(X_ss_all)} ({100*n_pseudo/len(X_ss_all):.1f}%)')
    print(f'Pseudo positive per species (mean): {pseudo_labels[pseudo_mask].sum(axis=0).mean():.1f}')

    # === 5. Pseudo-label + 원본으로 재학습 후 OOF 평가 ===
    X_combined = np.vstack([X_all, X_ss_all[pseudo_mask]])
    y_combined = np.vstack([y_all, pseudo_labels[pseudo_mask]])

    print(f'Combined data: {X_combined.shape[0]} samples ({X_all.shape[0]} original + {n_pseudo} pseudo)')

    # OOF (원본 데이터만 평가)
    X_comb_scaled = StandardScaler().fit_transform(X_combined)
    pca2 = PCA(n_components=64, random_state=42)
    X_comb_pca = pca2.fit_transform(X_comb_scaled)

    # 원본 부분만 평가
    main_labels_orig = list(meta_audio['label'].values) + list(meta_ss['label'].apply(lambda x: str(x).split(';')[0]).values)
    n_orig = len(X_all)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros((n_orig, len(species_cols)), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_comb_pca[:n_orig], main_labels_orig)):
        # train: fold train + ALL pseudo
        tr_full = np.concatenate([tr_idx, np.arange(n_orig, len(X_combined))])
        for j in range(len(species_cols)):
            y_tr = y_combined[tr_full, j]
            if y_tr.sum() < 3:
                continue
            m = LogisticRegression(C=0.1, max_iter=1000, solver='lbfgs', random_state=42)
            m.fit(X_comb_pca[tr_full], y_tr)
            oof[va_idx, j] = m.predict_proba(X_comb_pca[va_idx])[:, 1]

    valid_aucs = [roc_auc_score(y_all[:, j], oof[:, j]) for j in range(len(species_cols)) if y_all[:, j].sum() > 0]
    pseudo_auc = np.mean(valid_aucs)

    print(f'\n=== 결과 ===')
    print(f'원본만 OOF AUC:      0.9766 (trial_009 best)')
    print(f'Pseudo-label OOF AUC: {pseudo_auc:.6f}')
    print(f'차이: {pseudo_auc - 0.9766:+.6f}')
    print(f'소요 시간: {time.time()-t0:.0f}s')

    # === 6. Site-hour 통계 개선 확인 ===
    sites = meta_ss_all['filename'].apply(lambda x: parse_soundscape_filename(x)['site'])
    hours = meta_ss_all['filename'].apply(lambda x: parse_soundscape_filename(x)['hour_utc'])

    print(f'\n=== Soundscape 통계 ===')
    print(f'Sites: {sorted(sites.dropna().unique())}')
    print(f'Hours: {sorted(hours[hours >= 0].unique())}')
    print(f'Site × Hour 조합: {len(meta_ss_all.assign(site=sites, hour=hours).groupby(["site", "hour"]).size())}')

    # 저장
    out_dir = Path(__file__).parent
    np.save(out_dir / 'pseudo_probs.npy', pseudo_probs)
    np.save(out_dir / 'pseudo_labels.npy', pseudo_labels[pseudo_mask])
    print(f'\n저장 완료: {out_dir}')


if __name__ == '__main__':
    main()
