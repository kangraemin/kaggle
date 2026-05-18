## 구현 목표
- 변경 대상: `notebooks/kernel-metadata.json`, `notebooks/birdclef2026-effnet-5fold-blend.ipynb` cells[62]+cells[69], `submissions/sub_71/trial_078_birdnet_remove/meta.json`
- 핵심 변경:
  - kernel-metadata.json dataset_sources에서 "jarturo/birdnet" 제거
  - cells[62]: BLEND_BIRDNET 0.10→0.0, 주석 trial_077→trial_078, Perch 0.13→0.23
  - cells[69]: 헤더 첫 줄 trial_077→trial_078, Perch 13%→23%, BirdNET 10% 제거
  - meta.json 신규: trial_078, sub_71, kernel v69, weights={perch:0.23, proto:0.60, sed:0.10, ...}
- 참고: plan.md Phase 1 / Step 1

## 테스트 기준

| TC-ID | 유형 | 시나리오 | 기대 결과 | 실제 결과 |
|-------|------|----------|-----------|-----------|
| TC-1 | happy | cells[62]에서 BLEND_BIRDNET = 0.0 확인 | 'BLEND_BIRDNET = 0.0' 문자열 존재 | ✅ |
| TC-2 | happy | cells[62]에서 BLEND_PROTO = 0.60 확인 | 'BLEND_PROTO = 0.60' 문자열 존재 | ✅ |
| TC-3 | happy | cells[62]에서 BLEND_SED = 0.10 확인 | 'BLEND_SED = 0.10' 문자열 존재 | ✅ |
| TC-4 | happy | cells[62] 주석에 trial_078 포함 | 'trial_078' 문자열 존재 | ✅ |
| TC-5 | happy | cells[69] 헤더에 trial_078 포함 | 'trial_078' 문자열 존재 | ✅ |
| TC-6 | regression | kernel-metadata.json에 jarturo/birdnet 없음 | dataset_sources에 해당 항목 미존재 | ✅ |

검증 명령어: `python3 /Users/ram/verify_trial078.py`

## 실행출력
```
PASS (6 TCs)
```
cells[62] BLEND_BIRDNET = 0.0 ✅, BLEND_PROTO = 0.60 ✅, trial_078 ✅
cells[69] trial_078 헤더 ✅
kernel-metadata.json jarturo/birdnet 제거 ✅
