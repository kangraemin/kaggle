## 구현 목표
- 변경 대상: `birdclef/scripts/train_effnet_pseudo_mix.py`
- 핵심 변경:
  - `TRAIN_FOLDS = [0]` 상수 제거
  - main()에 argparse 추가: `--folds` (기본 0~4), `--dry-run`
  - `dry_run = '--dry-run' in sys.argv` → `dry_run = args.dry_run`
  - `TRAIN_FOLDS` 참조 → `train_folds` 변수로 교체

## 테스트 기준

| TC-ID | 유형 | 시나리오 | 기대 결과 | 실제 결과 |
|-------|------|----------|-----------|-----------|
| TC-1  | happy | `--dry-run` 인자만으로 실행 시 전체 5 fold | `train_folds=[0, 1, 2, 3, 4]` 출력 후 정상 종료 | ✅ |
| TC-2  | happy | `--folds 1 --dry-run` 으로 특정 fold 선택 | `train_folds=[1]` 출력 후 정상 종료 | ✅ |
| TC-3  | regression | dry-run이 실제 학습 없이 종료됨 | `Data loading OK` 출력, 모델 저장 없음 | ✅ |

## 실행출력
```
[dry-run] Data loading OK. Device=mps WORK_DIR=.../models/effnet_pseudo_mix
train_folds=[0, 1, 2, 3, 4]

train_folds=[1]
```

검증 명령어: `cd /Users/ram/programming/vibecoding/kaggle/birdclef && uv run python scripts/train_effnet_pseudo_mix.py --dry-run 2>&1 | grep "train_folds"`
