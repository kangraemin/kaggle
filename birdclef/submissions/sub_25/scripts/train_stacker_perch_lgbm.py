"""sub_25 — Perch embedding 기반 stacker (LightGBM).

목적:
  현재 노트북은 Perch embedding을 logit으로만 변환 (per-class linear probe).
  이 스크립트는 **labeled soundscape (test 도메인)** 1478 segments × Perch 1536 embedding
  으로 LightGBM을 직접 학습 → 노트북에 추가하는 두 번째 head.

왜 soundscape으로 학습하나:
  - test = Pantanal soundscape (라벨된 1478 segments와 같은 도메인)
  - focal training과 도메인 다름 (focal=close-mic 단일 종, soundscape=multi-species ambient)
  - trial_007 (Perch+XGB on focal) = 0.912. 그 이상 가능성.

산출:
  data/v2/stacker_perch_lgbm.pkl  — 234 binary classifier (per-class)
  data/v2/stacker_perch_oof.npy   — (1478, 234) OOF preds (val 평가 + blend weight tuning용)

사용:
  ipynb 추론 시: test의 Perch embedding → stacker.predict_proba → final blend
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
DATA_V2 = DATA / "v2"
PERCH_DIR = DATA / "perch_embeddings" / "perch_embeddings"
OUT = DATA_V2

import lightgbm as lgb
import joblib


def main():
    # 1) Perch soundscape embeddings + 라벨
    emb_ss = np.load(PERCH_DIR / "ss_embeddings.npy")  # (1478, 1536)
    meta_ss = pd.read_csv(PERCH_DIR / "ss_metadata.csv")
    print(f"Perch ss embeddings: {emb_ss.shape}")
    print(f"meta ss cols: {meta_ss.columns.tolist()}")

    # 2) labels_v2에서 soundscape segment 라벨 추출
    npz = np.load(DATA_V2 / "labels_v2.npz", allow_pickle=True)
    paths = npz["paths"]
    labels = npz["labels"]
    source = npz["source"]
    seg_start = npz["seg_start"]
    classes = npz["classes"]

    # soundscape segments (1478개)
    ss_mask = source == "soundscape"
    ss_paths = paths[ss_mask]
    ss_labels = labels[ss_mask]  # (1478, 234)
    ss_starts = seg_start[ss_mask]
    print(f"soundscape labels: {ss_labels.shape}, label sum: {ss_labels.sum()}")

    # 3) emb_ss와 labels 정렬 — 같은 순서인지 검증 (둘 다 ts.csv 행 순서로 만들어졌는지)
    # meta_ss에 (filename, start) 있으므로 매칭
    ts = pd.read_csv(DATA / "train_soundscapes_labels.csv")
    print(f"ts rows: {len(ts)}, emb_ss rows: {len(emb_ss)}")
    assert len(emb_ss) == len(ts), f"mismatch: emb {len(emb_ss)} vs ts {len(ts)}"
    # ss_labels (1478개) 순서 = ts와 같다고 가정 (build_labels_v2에서 ts iterrows)
    assert len(ss_labels) == len(ts), f"label mismatch: {len(ss_labels)} vs {len(ts)}"

    X = emb_ss.astype(np.float32)
    Y = ss_labels.astype(np.float32)

    # 4) 5-fold CV per-class stacker
    n_classes = Y.shape[1]
    n_samples = len(X)

    # stratify by file (같은 파일은 같은 fold) — leakage 방지
    file_ids = pd.factorize(ts["filename"])[0]
    unique_files = np.unique(file_ids)
    rng = np.random.RandomState(42)
    rng.shuffle(unique_files)
    file_to_fold = {fid: i % 5 for i, fid in enumerate(unique_files)}
    fold_idx = np.array([file_to_fold[f] for f in file_ids])

    oof_preds = np.zeros((n_samples, n_classes), dtype=np.float32)
    models_per_fold: list[dict[int, lgb.Booster]] = []

    for fold in range(5):
        train_mask = fold_idx != fold
        val_mask = fold_idx == fold
        X_train, X_val = X[train_mask], X[val_mask]
        Y_train, Y_val = Y[train_mask], Y[val_mask]
        print(f"\nFold {fold+1}/5: train={train_mask.sum()}, val={val_mask.sum()}")

        fold_models: dict[int, lgb.Booster] = {}
        for ci in range(n_classes):
            y_tr = Y_train[:, ci]
            if y_tr.sum() < 2:
                # too few positives — predict prior
                continue
            params = dict(
                objective="binary",
                learning_rate=0.05,
                num_leaves=15,
                max_depth=4,
                min_data_in_leaf=8,
                feature_fraction=0.5,
                bagging_fraction=0.8,
                bagging_freq=5,
                verbose=-1,
            )
            train_ds = lgb.Dataset(X_train, label=y_tr)
            booster = lgb.train(params, train_ds, num_boost_round=80)
            fold_models[ci] = booster
            oof_preds[val_mask, ci] = booster.predict(X_val)
        models_per_fold.append(fold_models)
        # quick fold AUC
        aucs = [roc_auc_score(Y_val[:, ci], oof_preds[val_mask, ci])
                for ci in fold_models if Y_val[:, ci].sum() > 0]
        print(f"  fold {fold+1} mean AUC: {np.mean(aucs):.4f} ({len(aucs)} classes)")

    # overall OOF AUC
    aucs_all = [roc_auc_score(Y[:, ci], oof_preds[:, ci])
                for ci in range(n_classes) if Y[:, ci].sum() > 0]
    print(f"\nOverall OOF AUC (macro, valid classes only): {np.mean(aucs_all):.4f} ({len(aucs_all)}/{n_classes} classes)")

    # 5) refit on all data per class
    print("\nRefit on all 1478 samples per class...")
    final_models: dict[int, lgb.Booster] = {}
    for ci in range(n_classes):
        y = Y[:, ci]
        if y.sum() < 2:
            continue
        params = dict(
            objective="binary",
            learning_rate=0.05,
            num_leaves=15,
            max_depth=4,
            min_data_in_leaf=8,
            feature_fraction=0.5,
            bagging_fraction=0.8,
            bagging_freq=5,
            verbose=-1,
        )
        train_ds = lgb.Dataset(X, label=y)
        booster = lgb.train(params, train_ds, num_boost_round=80)
        final_models[ci] = booster

    out_pkl = OUT / "stacker_perch_lgbm.pkl"
    joblib.dump({
        "models": final_models,
        "classes": classes,
        "n_features": X.shape[1],
    }, out_pkl)
    np.save(OUT / "stacker_perch_oof.npy", oof_preds)
    print(f"\nSaved: {out_pkl}")
    print(f"Saved: {OUT / 'stacker_perch_oof.npy'}")
    print(f"Trained classes: {len(final_models)} / {n_classes}")


if __name__ == "__main__":
    main()
