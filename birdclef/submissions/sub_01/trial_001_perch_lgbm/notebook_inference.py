"""BirdCLEF 2026 - Perch v2 + LightGBM Inference Notebook
Kaggle Notebook에서 실행하는 제출용 코드.
GPU 필요 없음 (CPU only).
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from pathlib import Path
import os

# Kaggle 경로
COMP_DATA = Path('/kaggle/input/birdclef-2026')
PERCH_MODEL = Path('/kaggle/input/google-perch-embeddings')  # 또는 직접 perch 모델 로드

# 1) Perch 모델 로드
print("Loading Perch model...")
model_url = "https://tfhub.dev/google/bird-vocalization-classifier/1"
perch_model = hub.load(model_url)

# 2) 학습 데이터 임베딩 + 모델 학습
print("Loading train embeddings...")
# 사전 추출 임베딩 사용 (또는 여기서 추출)
train_csv = pd.read_csv(COMP_DATA / 'train.csv')
sample = pd.read_csv(COMP_DATA / 'sample_submission.csv')
species_cols = [c for c in sample.columns if c != 'row_id']

# TODO: 사전 추출 임베딩 로드 or 실시간 추출
# 여기서는 실시간 추출 방식

def extract_embedding(audio_path, sr=32000):
    """오디오 → Perch 임베딩 추출"""
    y, _ = librosa.load(audio_path, sr=sr, duration=10)
    # Perch는 5초 단위
    if len(y) < sr * 5:
        y = np.pad(y, (0, sr * 5 - len(y)))
    y = y[:sr * 5].astype(np.float32)
    logits, embeddings = perch_model.infer_tf(y[np.newaxis, :])
    return embeddings.numpy().mean(axis=0)

# 3) Test soundscape 예측
print("Predicting test soundscapes...")
test_dir = COMP_DATA / 'test_soundscapes'
results = []

for audio_file in sorted(test_dir.glob('*.ogg')):
    y, sr = librosa.load(audio_file, sr=32000)
    # 5초 단위로 분할
    segment_len = sr * 5
    n_segments = len(y) // segment_len

    for seg_idx in range(n_segments):
        start = seg_idx * segment_len
        segment = y[start:start + segment_len].astype(np.float32)
        if len(segment) < segment_len:
            segment = np.pad(segment, (0, segment_len - len(segment)))

        logits, embeddings = perch_model.infer_tf(segment[np.newaxis, :])
        emb = embeddings.numpy().mean(axis=0)

        row_id = f"{audio_file.stem}_{seg_idx * 5}"
        row = {'row_id': row_id}
        # TODO: LightGBM 모델로 예측
        # 임시로 logits 기반 예측
        for sp in species_cols:
            row[sp] = 0.0
        results.append(row)

result_df = pd.DataFrame(results)
# sample submission 순서에 맞추기
submission = sample[['row_id']].merge(result_df, on='row_id', how='left').fillna(0.0)
submission.to_csv('submission.csv', index=False)
print(f"Saved submission.csv ({len(submission)} rows)")
