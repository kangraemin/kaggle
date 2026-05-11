# Plan: pseudo_mix TRAIN_FOLDS 전체(0~4) 실행

## Context
`train_effnet_pseudo_mix.py`는 `TRAIN_FOLDS = [0]`으로 fold 0만 학습하도록 고정되어 있음.
사용자가 나머지 fold(1~4)도 모두 돌리길 원함.
`train_effnet_multiwindow.py`처럼 `--folds` argparse를 추가하거나,
상수를 `list(range(N_FOLDS))`로 변경하는 방법이 있음.
argparse 추가가 더 유연하지만, 이미 multiwindow에 있으니 pseudo_mix도 동일하게 맞춤.

## 변경 파일

### `birdclef/scripts/train_effnet_pseudo_mix.py`

**Before:**
```python
TRAIN_FOLDS = [0]
```

**After:**
```python
TRAIN_FOLDS = list(range(N_FOLDS))  # 기본: 전체 5 folds
```

그리고 main()에 argparse 추가:
```python
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, nargs='+', default=list(range(N_FOLDS)))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    dry_run = args.dry_run
    train_folds = args.folds
```

그리고 `TRAIN_FOLDS` 상수 사용 부분을 `train_folds`로 교체.

## 검증
```bash
cd birdclef && uv run python scripts/train_effnet_pseudo_mix.py --dry-run
# → [dry-run] Data loading OK. train_folds=[0, 1, 2, 3, 4]

uv run python scripts/train_effnet_pseudo_mix.py --folds 1 2 --dry-run
# → train_folds=[1, 2]
```

## E2E 테스트
```bash
cd /Users/ram/programming/vibecoding/kaggle/birdclef
uv run python scripts/train_effnet_pseudo_mix.py --dry-run 2>&1 | grep "train_folds"
# 기대: train_folds=[0, 1, 2, 3, 4]

uv run python scripts/train_effnet_pseudo_mix.py --folds 1 --dry-run 2>&1 | grep "train_folds"
# 기대: train_folds=[1]
```

## 개발 Phase 계획

### Phase 1: argparse + TRAIN_FOLDS 수정
**목표**: dry-run 포함 argparse로 folds 선택 가능하게 변경
- Step 1: argparse 추가 + TRAIN_FOLDS 상수 제거 + dry_run 변수 교체 — dry-run 정상 종료 확인
