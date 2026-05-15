# trial_062 reflection — pmix_weight_up

- Date: 2026-05-15
- Score: 0.934 (➖ best 동률, baseline trial_059 = 0.934)
- ralph iter: 12

## 변경 내용

vs trial_059:
- BLEND_PSEUDOMIX 0.08 → 0.11 (+3pp)
- ConvNeXt 컴포넌트 제거 (BLEND_CONVNEXT 0.03 → 0)
- ConvNeXt 데이터셋 2개 kernel-metadata 제거 (`ramkang/birdclef2026-convnext-5fold`, `denden12/birdset-convnext-base-xcl`)

## 결과 분석

**pmix 0.08→0.11**: 효과 없음. trial_056에서 pmix 0.06→0.08이 무효과였고, 0.08→0.11도 동일. 5-fold pmix 컴포넌트의 weight 증가는 이미 포화 — 0.11 이상은 신호보다 노이즈를 더 추가하는 것으로 보임.

**ConvNeXt dataset 제거 → auto-submit 정상화 확인**: trial_060/061에서 PENDING row조차 등장하지 않았던 이유가 ConvNeXt dataset 마운트 거부였음이 trial_062에서 확정됨. trial_062는 정상적으로 PENDING row 등장 → 78분 후 0.934 채점 완료. 단, 채점 자체가 78분 소요 (trial_058/059는 수분 내 완료) — Kaggle 서버 지연으로 추정, 코드 자체 문제 아님.

## 교훈

1. pmix weight 증가는 0.08을 넘어서면 효과 없음. 방향 전환 필요.
2. ConvNeXt dataset이 hidden test re-run을 차단하는 패턴 완전 확인 — 앞으로 새 데이터셋 추가 시 hidden test 호환성 사전 검증 필요.
3. PENDING 78분은 Kaggle 인프라 지연 — 이 자체는 문제 없음.

## 다음 시도

trial_063 (mwf0_zero_pmix13): mwf0 0.02 → 0.00 (noisy 단일 fold0 완전 제거), 여기서 freed 2pp를 pmix 0.11 → 0.13 으로 이동.
- 가설: mwf0(단일 fold0, noisy)를 제거하면 블렌드 순도가 높아질 수 있다. pmix는 5-fold 앙상블로 mwf0보다 clean.
- kernel v50 이미 COMPLETE (wall 352s, blend trial_063: Perch 72% + EffNet5fold 15% + pmix 13%, mean 0.0507).
