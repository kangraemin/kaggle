## 목표
TRAIN_FOLDS = [0] 상수를 argparse로 교체해 기본값이 전체 5 fold가 되도록 변경

## 기술 접근
- `birdclef/scripts/train_effnet_pseudo_mix.py`: TRAIN_FOLDS 상수 제거 → main()에 argparse 추가, train_folds 변수로 교체

## Steps
- Step 1: argparse 추가 + TRAIN_FOLDS 제거 + dry_run 변수 교체 — dry-run 전체 fold / 선택 fold 정상 출력 확인
