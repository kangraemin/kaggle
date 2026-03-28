# Submissions Log — ts-forecasting

## 규칙
- sub_XX = Kaggle 제출 1회 단위
- 제출 후 반드시 `sub_XX/reflection.md` 작성 → ✅ reflected 표시 후 다음 sub 진행
- 새 trial은 항상 새 sub 폴더에 (제출 완료된 sub에 추가 금지)

## 제출 전 필수 체크 (0점 방지)
1. 예측값 분포 확인 (abs > 10 몇 건인지)
2. 고가중치 코드 top 10 예측값 확인
3. danger_ratio = max_weight × max_pred² / denominator → 0.1 넘으면 패치 후 제출

| # | Date | Best Trial | Val | Public | Private | Gap | Status |
|---|------|------------|-----|--------|---------|-----|--------|
| 01 | 2026-03-26 | trial_001 | 0.1163 | 0.1499 | - | +0.034 | ✅ reflected |
| 02 | 2026-03-27 | trial_010 | 0.8923 | 0.0000 | - | -0.892 | ✅ reflected |
| 03 | 2026-03-28 | trial_021 | 0.6420 (cold-start) | TBD | - | TBD | ⏳ pending (daily limit) |
