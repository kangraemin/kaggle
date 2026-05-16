# Round 3: 최종 검증

## 블렌드 가중치 합산 검증

BLEND_PERCH = 1 - BLEND_PROTO - BLEND_EFFNET - BLEND_MWF0 - BLEND_PSEUDOMIX - BLEND_SED
           = 1 - 0.0 - 0.0 - 0.02 - 0.05 - 0.15
           = 0.78 (78%) ✅

## 격리 테스트 의도 확인

- ProtoSSM 0% → ProtoSSM 완전 제거 ✅
- SED 15% 단독 유지 ✅
- Perch 78% (전체에서 나머지 모두 제거 후 흡수) ✅
- trial_075 라벨 cells[67]에 반영 ✅

## 결론
통과
