# Sub 26 Reflection — trial_039_convnext_5fold

**Base**: sub_24 trial_037 ConvNeXt 3-way blend (0.930) — fold1 single
**Trial**: trial_039

## 결과
- Public: **silent reject** (COMPLETE, publicScore 없음)
- 2026-04-30 제출. 4월 30일부터 Kaggle 채점 시스템 이상 시작됨.

## 변경사항 (sub_25 대비)
- ConvNeXt: fold2 single → fold0~4 전체 5-fold (glob 자동 로드)
- 데이터셋 `ramkang/birdclef2026-convnext-5fold` 업데이트 (fold0,1,2,3,4 전부 포함)
- 블렌드 비율 유지: Perch 65% + EffNet 15% + ConvNeXt 20%

## 교훈
- 채점 시스템 자체가 이상. 코드 문제 아님.
- sub_27~33까지 전부 silent reject — 플랫폼 이상으로 결론.

## 버려야 할 것
- ONNX INT8 변환 (sub_28~30): 채점 기회 자체가 없었으므로 검증 불가. 플랫폼 회복 후 재시도.

## 유지해야 할 것
- 3-way blend 골격 (Perch 65% + EffNet 15% + ConvNeXt 20%)
- ConvNeXt XCL backbone 5-fold (CV 0.9905)

## 다음 가설
- 플랫폼 회복 대기 → blend v28 diagnostic 결과 확인 후 판단
- trial_040 effnet_multiwindow (Val 0.9794): 플랫폼 정상화 시 재제출 가치 있음
