"""BirdCLEF post-processing OOF 검증 스크립트
- 캐시된 OOF 점수에 post-processing 적용 전/후 비교
- 제출 전에 반드시 이걸로 검증할 것!
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

DATA = Path(__file__).parent.parent.parent / 'data'
EMB = DATA / 'perch_embeddings' / 'perch_embeddings'


def macro_auc(y_true, y_score):
    keep = y_true.sum(axis=0) > 0
    if keep.sum() == 0:
        return 0.0
    return roc_auc_score(y_true[:, keep], y_score[:, keep], average='macro')


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def apply_temperature(scores, species_classes, temp_event=1.10, temp_texture=0.95):
    """Per-taxon temperature scaling"""
    TEXTURE_TAXA = {'Amphibia', 'Insecta'}
    out = scores.copy()
    for ci, cn in enumerate(species_classes):
        if cn in TEXTURE_TAXA:
            out[:, ci] /= temp_texture
        else:
            out[:, ci] /= temp_event
    return out


def apply_file_confidence(scores, n_windows=12, top_k=2):
    """File-level confidence scaling (2025 Rank 1/2)"""
    file_view = scores.reshape(-1, n_windows, scores.shape[1])
    top_k_mean = np.sort(file_view, axis=1)[:, -top_k:, :].mean(axis=1, keepdims=True)
    confidence = sigmoid(top_k_mean)
    file_scaled = file_view * (0.5 + 0.5 * confidence)
    return file_scaled.reshape(-1, scores.shape[1])


def apply_rank_aware(scores, n_windows=12):
    """Rank-aware scaling (2025 Rank 3)"""
    file_view = scores.reshape(-1, n_windows, scores.shape[1])
    file_max = file_view.max(axis=1, keepdims=True)
    file_max_factor = np.clip(file_max, 0, None) ** 0.5
    rank_scaled = file_view * (0.3 + 0.7 * (file_max_factor / (file_max_factor.max() + 1e-8)))
    return rank_scaled.reshape(-1, scores.shape[1])


def apply_gauss_smooth(scores, n_windows=12, weights=np.array([0.1, 0.2, 0.4, 0.2, 0.1])):
    """Gaussian temporal smoothing"""
    from scipy.ndimage import convolve1d
    smoothed = scores.reshape(-1, n_windows, scores.shape[1]).copy()
    for i in range(smoothed.shape[0]):
        smoothed[i] = convolve1d(smoothed[i], weights, axis=0, mode='nearest')
    return smoothed.reshape(-1, scores.shape[1])


def main():
    # 데이터 로드
    sample = pd.read_csv(DATA / 'sample_submission.csv')
    species_cols = [c for c in sample.columns if c != 'row_id']
    taxonomy = pd.read_csv(DATA / 'taxonomy.csv') if (DATA / 'taxonomy.csv').exists() else None

    X_audio = np.load(EMB / 'audio_embeddings.npy')
    meta_audio = pd.read_csv(EMB / 'audio_metadata.csv')
    X_ss = np.load(EMB / 'ss_embeddings.npy')
    meta_ss = pd.read_csv(EMB / 'ss_metadata.csv')

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

    # 간단한 LR OOF로 base scores 생성
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    pca = PCA(n_components=64, random_state=42)
    X_pca = pca.fit_transform(X_all)
    main_labels = list(meta_audio['label'].values) + list(meta_ss['label'].apply(lambda x: str(x).split(';')[0]).values)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros_like(y_all)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_pca, main_labels)):
        for j in range(len(species_cols)):
            y_tr = y_all[tr_idx, j]
            if y_tr.sum() == 0:
                continue
            m = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver='lbfgs')
            m.fit(X_pca[tr_idx], y_tr)
            oof[va_idx, j] = m.predict_proba(X_pca[va_idx])[:, 1]

    base_auc = macro_auc(y_all, oof)
    print(f'Base OOF AUC: {base_auc:.6f}')

    # species class names
    if taxonomy is not None:
        class_map = taxonomy.set_index('primary_label')['class_name'].to_dict()
        species_classes = [class_map.get(s, 'Unknown') for s in species_cols]
    else:
        species_classes = ['Unknown'] * len(species_cols)

    # logit 변환 (post-processing은 logit 공간에서 작동)
    oof_logit = np.log(np.clip(oof, 1e-7, 1 - 1e-7) / (1 - np.clip(oof, 1e-7, 1 - 1e-7)))

    # 각 post-processing 단계별 효과
    # 주의: soundscape가 60초 파일이 아닌 개별 5초 클립이라 file-level은 의미 없을 수 있음
    # 여기서는 12개씩 묶어서 시뮬레이션

    # 1. Temperature만
    temp = apply_temperature(oof_logit, species_classes)
    print(f'+ Temperature:        {macro_auc(y_all, sigmoid(temp)):.6f}')

    # 2. Temperature + Gaussian
    gauss = apply_gauss_smooth(temp, n_windows=12)
    print(f'+ Temp + Gauss:       {macro_auc(y_all, sigmoid(gauss)):.6f}')

    # 참고: file-level/rank-aware는 12개씩 묶어야 해서 데이터가 12의 배수여야 함
    n = len(oof_logit)
    n_trim = (n // 12) * 12
    if n_trim > 0:
        oof_trim = oof_logit[:n_trim]
        y_trim = y_all[:n_trim]

        fc = apply_file_confidence(apply_temperature(oof_trim, species_classes))
        print(f'+ Temp + FileConf:    {macro_auc(y_trim, sigmoid(fc)):.6f} (trimmed to {n_trim})')

        ra = apply_rank_aware(apply_temperature(oof_trim, species_classes))
        print(f'+ Temp + RankAware:   {macro_auc(y_trim, sigmoid(ra)):.6f} (trimmed to {n_trim})')

        full = apply_gauss_smooth(apply_rank_aware(apply_file_confidence(apply_temperature(oof_trim, species_classes))))
        print(f'+ All combined:       {macro_auc(y_trim, sigmoid(full)):.6f} (trimmed to {n_trim})')


if __name__ == '__main__':
    main()
