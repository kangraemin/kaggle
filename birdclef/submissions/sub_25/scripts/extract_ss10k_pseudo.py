"""sub_25 — ss10k_pseudo confident-positive 추출 → labels_v3 (cross-model pseudo).

기존 ss10k_pseudo:
  - 127104 soundscape segments × 234 Perch logit (5초 단위)
  - meta에 site/hour 정보 (S05 108개 = test 사이트)
  - **Perch가 만든 pseudo (cross-model: ConvNeXt/EffNet의 self-pseudo와 다름)**

trial_031 self-pseudo 실패 vs 본 cross-model pseudo:
  - trial_031: EffNet → EffNet self-label → bias 증폭
  - 본 ss10k: Perch (다른 architecture, 다른 학습 데이터) → EffNet/ConvNeXt 학습

추출 정책:
  - sigmoid(logit) > 0.99 = confident positive (soft label 0.85)
  - sigmoid(logit) < 0.01 = confident negative (생략 — multi-hot에선 0이 default)
  - 25 Insect sonotype은 Perch가 못 잡음 (max prob 0.5) → ss10k에서 제외
  - subsample: 너무 많으면 focal data 비중 떨어짐. ~30k로 제한

산출:
  data/v2/ss10k_subset.npz (paths, soft_labels, seg_start, source='pseudo_ss10k', primary_strat)

이후 build_multi_window_cache.py를 ss10k_subset.npz에도 적용 → 통합 cache 만들고 train_convnext_v2.py에서 함께 학습.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
DATA_V2 = DATA / "v2"
PSEUDO_DIR = DATA / "ss10k_pseudo"
SS_AUDIO_DIR = DATA / "train_soundscapes"

LOGIT_THRESH = 4.6  # sigmoid(4.6) ≈ 0.99
SOFT_LABEL = 0.85  # confident pseudo: 학습 신호로 강함, 1.0 미만으로 노이즈 톨러런스
N_SUBSAMPLE = 30000  # focal 35549 대비 비중 균형


def main():
    scores = np.load(PSEUDO_DIR / "scores.npy", mmap_mode="r")
    meta = pd.read_csv(PSEUDO_DIR / "meta.csv")
    print(f"ss10k_pseudo: {scores.shape}, meta {len(meta)}")

    # 234 클래스 정렬 — taxonomy 기준 동일해야 함
    tax = pd.read_csv(DATA / "taxonomy.csv")
    LABELS = sorted(tax["primary_label"].dropna().astype(str).unique())
    assert scores.shape[1] == len(LABELS), f"class mismatch: scores {scores.shape[1]} vs taxonomy {len(LABELS)}"

    # 25 sonotype index — 이 클래스는 Perch 못 잡음, 제외
    sonotype_idx = np.array([i for i, c in enumerate(LABELS) if c.startswith("47158son")])
    print(f"25 sonotype indices excluded from pseudo: {len(sonotype_idx)}")

    # confident-positive segments 선택: 적어도 1개 sonotype 외 클래스에서 logit > thresh
    conf_pos_mask = scores > LOGIT_THRESH  # (127104, 234)
    if len(sonotype_idx) > 0:
        # sonotype 컬럼 0으로 마스크 (이 클래스 제외)
        conf_pos_mask_no_sono = conf_pos_mask.copy()
        conf_pos_mask_no_sono[:, sonotype_idx] = False
    else:
        conf_pos_mask_no_sono = conf_pos_mask

    has_any_conf = conf_pos_mask_no_sono.any(axis=1)
    print(f"segments with any confident pos (excluding sonotype): {has_any_conf.sum()}/{len(scores)}")

    # subsample: S05 (test 사이트) 우선, 그 외 균등
    s05_mask = (meta["site"] == "S05").values
    s05_conf = s05_mask & has_any_conf
    print(f"S05 confident: {s05_conf.sum()}")

    rng = np.random.RandomState(42)
    idx_s05 = np.where(s05_conf)[0]
    idx_others = np.where(has_any_conf & ~s05_mask)[0]
    rng.shuffle(idx_others)

    n_others = max(0, N_SUBSAMPLE - len(idx_s05))
    selected_idx = np.concatenate([idx_s05, idx_others[:n_others]])
    selected_idx.sort()
    print(f"Selected: {len(selected_idx)} (S05={len(idx_s05)}, others={n_others})")

    # 메타 + soft label 만들기
    sel_scores = np.array(scores[selected_idx], dtype=np.float32)
    sel_meta = meta.iloc[selected_idx].reset_index(drop=True)

    # soft labels: confident-positive만 0.85, 나머지 0
    soft_labels = np.zeros((len(selected_idx), len(LABELS)), dtype=np.float32)
    pos_mask = sel_scores > LOGIT_THRESH
    if len(sonotype_idx) > 0:
        pos_mask[:, sonotype_idx] = False  # sonotype은 학습 신호 X
    soft_labels[pos_mask] = SOFT_LABEL
    print(f"soft labels: total positives = {(soft_labels > 0).sum()}, mean per row = {(soft_labels > 0).sum(axis=1).mean():.2f}")

    # paths + seg_start: row_id에서 파싱
    # row_id: BC2026_Train_NNNN_SXX_YYYYMMDD_HHMMSS_<endsec>
    paths = []
    seg_starts = []
    primary_strat = []
    n_skip = 0
    for i, row in sel_meta.iterrows():
        rid = row["row_id"]
        # endsec 마지막 토큰
        parts = rid.rsplit("_", 1)
        if len(parts) != 2:
            n_skip += 1
            continue
        try:
            endsec = int(parts[1])
        except ValueError:
            n_skip += 1
            continue
        start_sec = endsec - 5
        # filename 복원
        fname = parts[0].replace("BC2026_Train_", "BC2026_Train_") + ".ogg"
        # 실제로는 row_id가 BC2026_Train_NNNN_SXX_YYYYMMDD_HHMMSS_endsec → fname는 _endsec 빼고 .ogg
        fname = parts[0] + ".ogg"
        ap = SS_AUDIO_DIR / fname
        paths.append(str(ap))
        seg_starts.append(float(start_sec))
        # primary stratify: confident class 중 가장 강한 것 (sonotype 제외)
        s = sel_scores[i].copy()
        if len(sonotype_idx) > 0:
            s[sonotype_idx] = -1e9  # 사실상 제외
        argmax_cls = int(np.argmax(s))
        primary_strat.append(LABELS[argmax_cls])

    print(f"paths skipped: {n_skip}")
    paths = np.array(paths, dtype=object)
    seg_starts = np.array(seg_starts, dtype=np.float32)
    primary_strat = np.array(primary_strat, dtype=object)

    # 파일 존재 확인 (몇 개 sample)
    exists = np.array([Path(p).exists() for p in paths[:100]])
    print(f"first 100 paths exist: {exists.sum()}/100")
    assert exists.sum() == 100, "some pseudo audio paths don't resolve"

    out_path = DATA_V2 / "ss10k_subset.npz"
    np.savez_compressed(
        out_path,
        paths=paths,
        labels=soft_labels,
        seg_start=seg_starts,
        seg_dur=np.full(len(paths), 5.0, dtype=np.float32),
        source=np.full(len(paths), "pseudo_ss10k", dtype=object),
        primary_strat=primary_strat,
        classes=np.array(LABELS, dtype=object),
    )
    print(f"Saved: {out_path}  shape={soft_labels.shape}")
    print(f"  S05 in subset: {(np.char.find(paths.astype(str), '_S05_') >= 0).sum()}")


if __name__ == "__main__":
    main()
