# Sub 03 Reflection

## 결과
- trial_003_balanced_blend: Val 0.9711 (bal_acc) / Public **0.9691**
- trial_003_no_threshold: Val 0.9678 / Public 0.9652
- 이전 best (sub_02): Public 0.9609 → **+0.0082 개선**
- Gap: -0.0020 (sub_01 -0.0255 → sub_03 -0.0020으로 대폭 개선)

## 핵심 발견

### 1. 메트릭이 전부였다
- 대회 메트릭이 accuracy가 아니라 **balanced_accuracy**라는 걸 discussion에서 발견
- High 클래스가 3.3%밖에 안 되니 일반 accuracy로는 High를 무시해도 높게 나옴
- 메트릭 수정만으로 최적화 방향이 완전히 바뀜

### 2. class_weight=balanced가 CatBoost를 살림
- trial_002에서 CatBoost가 꼴찌 (0.9824, 앙상블 가중치 1/10)
- auto_class_weights=Balanced 적용 후 **개별 모델 1등** (0.9679)
- 앙상블 가중치도 CAT:XGB:LGBM = 4:3:1로 뒤집힘

### 3. threshold optimization이 public에서 먹힌다
- Val: 0.9678 → 0.9711 (+0.0033)
- Public: 0.9652 → 0.9691 (+0.0039) — public에서 더 큰 효과!
- High 클래스 확률에 2.6배, Medium에 0.75배 가중치
- minority class 보정이 일반화에도 효과적

### 4. original data blending 효과
- 10K 행 추가 (train 630K → 640K)
- target encoding의 source로도 활용
- val-public gap이 -0.025 → -0.002로 크게 줄어든 건 blending + balanced 메트릭 효과

## 아쉬운 점
- 1등(0.9797)과 0.0106 차이 — 아직 큼
- CatBoost가 폴드당 30분+ 걸려서 전체 5시간 소요
- 003_multiseed_stacking은 잘못된 메트릭(accuracy)으로 작성됨 — 실행 전 폐기

## 다른 세션 trials (004~007)
- 004: target enc + cat pairs — accuracy 메트릭 (잘못됨)
- 005: ext data + balanced acc — val 0.9692 (003보다 낮음)
- 006: full pairwise + XGB dominant — public 0.9668 (003보다 낮음)
- 007: bias tuned stacking — 진행 중

## 다음에 해볼 것
- Optuna 하이퍼파라미터 튜닝 (특히 CatBoost)
- 10-fold + multi-seed 앙상블
- pseudo-labeling
- Neural net (TabNet/MLP) 추가
- threshold grid를 더 fine-grained하게 (0.01 단위)
