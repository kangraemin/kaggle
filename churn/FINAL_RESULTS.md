# Churn Competition Final Results

대회: Playground Series S6E3 — Predict Customer Churn
기간: 2026-03-27 ~ 2026-03-31 (5일)
팀 수: 4,142

## 최종 성적

| 제출 | Description | Public | Private | 선택 |
|---|---|---|---|---|
| trial_084 Ridge v4 | 56 OOFs Ridge alpha=50 | 0.91704 | **0.91807** | |
| trial_083 Ridge v3 | 53 OOFs Ridge alpha=100 | 0.91701 | **0.91815** ← best private | |
| trial_082 RealMLP+XGB rank | RealMLP×0.79+XGB081×0.21 | 0.91689 | 0.91802 | |
| trial_080 Ridge v2 | 51 OOFs Ridge | 0.91702 | 0.91808 | |
| trial_079 HC v2 | 3-model rank blend | 0.91693 | 0.91808 | |
| trial_078 RealMLP+XGB074 | rank blend | 0.91690 | 0.91802 | |
| **Ridge ensemble 55** | **55 OOFs alpha=100** | **0.91707** ← best public | **0.91807** | |
| fine_blend | RealMLP 0.86+XGB 0.14 | 0.91686 | 0.91801 | |
| trial_067 dist_digit | distribution features | 0.91571 | 0.91697 | |
| RealMLP+XGB blend | RealMLP 0.85+XGB 0.15 | 0.91686 | 0.91801 | |
| RealMLP 단독 | 20-fold CPU | 0.91683 | 0.91793 | |
| trial_028 LGBM reg | lambda=2.0 7seeds | 0.91400 | 0.91527 | |
| trial_024 XGB multi | 7seeds | 0.91395 | 0.91540 | |
| trial_017 blend | 5모델 blend | 0.91404 | 0.91538 | |
| trial_014 blend | 5모델 blend | 0.91404 | 0.91534 | |
| trial_004 LGBM tuned | Optuna 50 trials | 0.91393 | 0.91535 | |
| trial_001 baseline | LGBM baseline | 0.91377 | 0.91497 | |

## Best Private: 0.91815 (trial_083 Ridge v3)
## Best Public: 0.91707 (Ridge 55 OOFs)

## 핵심 수치
- 총 제출: 17회
- 총 trial: 84+
- Best public: 0.91707
- Best private: 0.91815
- 1위 public: 0.91771
- Baseline → Best: 0.91377 → 0.91707 (+0.0033)

## 주요 전환점
1. **trial_001 → 004**: Optuna 튜닝 (+0.0005)
2. **trial_004 → 060**: RealMLP Kaggle fork (+0.0028) ← 가장 큰 점프
3. **trial_060 → Ridge**: 55 OOF Ridge ensemble (+0.0002)

## 배운 것
- GBDT 로컬 한계: val 0.91694가 천장
- RealMLP가 게임 체인저: 0.91938 (val +0.0024)
- Ridge ensemble이 다양한 OOF를 효과적으로 결합
- 피처 엔지니어링보다 모델 다양성이 중요
- Kaggle notebook fork 전략이 핵심
- pytabkit CUDA 호환성 문제 → CPU fallback으로 해결
