"""sub_25 — 현 best 노트북에 domain prior mask 후처리 추가.

Strategy:
  - 학습/모델 weight 변경 0
  - cell 63 (최종 submission 작성) 직전에 prior mask 곱하기
  - prior mask는 competition data (taxonomy + train.csv + train_soundscapes_labels)로 in-notebook 계산
    → 별도 Kaggle dataset upload 불필요

Tier 가중치:
  - A_in_soundscape (75 종)       1.0
  - B_in_pantanal_train (92 종)   1.0
  - C_other (67 종)               0.3

라이브러리 교훈 적용:
  - cell IDs 보존 (`api-notebook-edit-breaks-scoring`)
  - kernel-metadata.json 파일명 고정 (`kernels-push-ignores-non-default-metadata-filename`)
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NB_SRC = ROOT / "notebooks" / "birdclef2026-effnet-5fold-blend.ipynb"
NB_DST = ROOT / "notebooks" / "birdclef2026-effnet-5fold-blend.ipynb"  # in-place edit

PRIOR_MASK_FN = '''
# === sub_25 v1: Domain prior mask (post-processing) ===
# Pantanal box (recording_location.txt: 위도 -21.6~-16.5, 경도 -57.6~-55.9)
# Tier A (in labeled soundscape) + Tier B (in Pantanal box train) → 1.0
# Tier C (other train) → 0.3
def _compute_class_prior_mask(primary_labels, base_dir):
    import pandas as pd
    import numpy as np
    train_csv = pd.read_csv(base_dir / "train.csv")
    ss_lab = pd.read_csv(base_dir / "train_soundscapes_labels.csv")
    ss_classes = set()
    for s in ss_lab["primary_label"]:
        ss_classes.update(str(s).split(";"))
    in_box = train_csv[
        (train_csv["latitude"].between(-21.6, -16.5))
        & (train_csv["longitude"].between(-57.6, -55.9))
    ]
    pantanal_classes = set(in_box["primary_label"].astype(str).unique())
    mask = np.ones(len(primary_labels), dtype=np.float32)
    for i, cls in enumerate(primary_labels):
        if str(cls) in ss_classes or str(cls) in pantanal_classes:
            mask[i] = 1.0
        else:
            mask[i] = 0.3
    return mask

CLASS_PRIOR_MASK = _compute_class_prior_mask(PRIMARY_LABELS, BASE)
print(f"[sub_25 v1] prior mask: A+B={int((CLASS_PRIOR_MASK==1.0).sum())} C={int((CLASS_PRIOR_MASK==0.3).sum())} mean={CLASS_PRIOR_MASK.mean():.3f}")
'''

APPLY_LINE = "\n# --- sub_25 v1: Apply domain prior mask ---\nprint('[sub_25 v1] applying prior mask...')\nprobs = probs * CLASS_PRIOR_MASK[None, :]\n"


def main():
    nb = json.loads(NB_SRC.read_text())

    # 1) Cell 16 직후에 prior mask 함수 + 계산 셀 추가 (taxonomy 로드 후)
    new_cell = {
        "cell_type": "code",
        "id": "sub25-prior-mask",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": PRIOR_MASK_FN.lstrip().splitlines(keepends=True),
    }

    # 이미 추가되어 있으면 스킵
    if any(c.get("id") == "sub25-prior-mask" for c in nb["cells"]):
        print("[skip] prior mask cell already present, replacing")
        nb["cells"] = [c for c in nb["cells"] if c.get("id") != "sub25-prior-mask"]

    # cell 16 = taxonomy/sample_sub 로드. 그 직후에 삽입.
    insert_idx = 17
    nb["cells"].insert(insert_idx, new_cell)
    print(f"Inserted prior mask cell at idx {insert_idx} (id=sub25-prior-mask)")

    # 2) 최종 submission cell (이전 cell 63 = 이제 idx +1)을 찾아서 'probs = probs * ...' 추가
    target_idx = None
    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell.get("source", [])) if isinstance(cell.get("source"), list) else cell.get("source", "")
        if 'submission.to_csv("submission.csv"' in src:
            target_idx = i
            break
    if target_idx is None:
        raise RuntimeError("submission write cell not found")

    src_lines = nb["cells"][target_idx]["source"]
    src_str = "".join(src_lines) if isinstance(src_lines, list) else src_lines

    if "[sub_25 v1] applying prior mask" in src_str:
        print(f"[skip] apply line already present in cell {target_idx}")
    else:
        # `probs = apply_per_class_thresholds` 다음 줄에 prior mask apply
        marker = "probs = apply_per_class_thresholds(probs, PER_CLASS_THRESHOLDS, n_windows=N_WINDOWS)"
        if marker not in src_str:
            raise RuntimeError("apply_per_class_thresholds marker not found in submission cell")
        new_src = src_str.replace(marker, marker + APPLY_LINE)
        nb["cells"][target_idx]["source"] = new_src.splitlines(keepends=True)
        print(f"Inserted apply line in cell {target_idx} (post apply_per_class_thresholds)")

    NB_DST.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"Saved: {NB_DST}")


if __name__ == "__main__":
    main()
