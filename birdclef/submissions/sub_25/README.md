# sub_25 — Domain-aware Cleanup Round

## 가설
이전 24개 submission이 거의 다 못 본 영역:
1. **labeled soundscape (66 files, 1478 segments)** 학습/val 미활용 → 25 Insect sonotype 클래스 영구 0점
2. **test = single site S05 (Pantanal)** prior 미활용
3. **secondary_labels** train.csv 컬럼 미활용
4. **첫 5초만 cache** → time-domain augmentation 0
5. **focal→soundscape 도메인 갭** = val 0.99 vs LB 0.93 본질
6. **BirdNET 외부 모델** 활용 안 됨 (self-pseudo만 trial_031에서 시도)

## 산출물 구조
```
sub_25/
  eda/             # 데이터 분석 결과 (PNG, MD, CSV)
  scripts/         # sub_25 전용 스크립트
  data/            # 산출 라벨/마스크/캐시 (작은 것만)

scripts/v2/        # 학습용 새 스크립트 (큰 코드)
data/v2/           # 큰 산출물 (multi-window cache, pseudo-label, etc.)
```

## 작업 순서 (학습 자원 경합 없는 것 → 있는 것)
- Phase A (학습 영향 0, 즉시): EDA, labeled soundscape 통합, secondary_labels, class prior, BirdNET 셋업
- Phase B (fold 5 끝나면): multi-window cache 재구축, v2 학습

## 이전 reflection 미스 정정
- sub_18 reflection 中 "hidden test (미국 자연 soundscape)" 가정 → **틀림**. test = Pantanal Brazil (`recording_location.txt`).
