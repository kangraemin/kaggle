## Submission 80 Reflection

**Base**: trial_080 sed_up (kernel v72, 0.938 — best)
**Trial**: trial_087 proto_tta

### 결과
- Public: **0.938** (best 동률)
- 제출: 2026-06-01 02:09 → COMPLETE

### 변경사항 (이전 sub 대비)
- ProtoSSM TTA shift 5→7 (`CFG["tta_shifts"]` [0,1,-1,2,-2]→[0,1,-1,2,-2,3,-3])
- best weight 복원: Proto 0.40→0.50, EffNet 0.10→0 (기여 한계 확정으로 제거)
- kernel v78→v79

### 교훈
- ProtoSSM TTA 7-shift도 무효과. V18에서 이미 5-shift라 추가 2개 shift의 한계효용 0. 시간축 TTA는 포화.
- **4축 전부 0.938 포화 확정**:
  - weight (trial_082/083: SED·Perch 비중 조정) → 무효/하락
  - scale (trial_084: logit z-score 정규화) → 무효 (ROC-AUC rank-based)
  - 보조 컴포넌트 (trial_085/086: EffNet 5~10%) → 무효/하락
  - 추론 (trial_087: TTA 7-shift) → 무효
- 약 16+ trial(062~087)이 0.934~0.938 천장. **blend·후처리·추론 파라미터로 짤 수 있는 점수는 0.938이 한계.**

### 버려야 할 것
- blend weight/scale/컴포넌트/추론 파라미터 추가 튜닝 전부. 4축 모두 포화 확인됨.
- "파라미터 조합으로 천장 돌파" 접근 — 더 시도 무의미.

### 유지해야 할 것
- **trial_080 = best 0.938** (SED40/Proto50/Perch10) — 최종 제출 후보 확정.
- 4개 컴포넌트 logit 진단값, TTA 5-shift 구성.

### 다음 가설
**파라미터 공간 소진 → 남은 레버는 모델/데이터 자체 (큰 작업, 사용자 판단 필요)**:
1. **ProtoSSM 재학습** — d_model/layer 확장, epoch↑, 더 강한 augmentation. base 모델 성능 자체를 올려야 천장 상승.
2. **SED 모델 교체/재학습** — 현 Distilled-SED를 더 강한 SED로.
3. **신규 1차 모델 학습** — Perch/Proto/SED 외 새 강한 백본 (BirdNET 재시도, 또는 SOTA 오디오 모델 fine-tune).

⚠️ 위 3개는 모두 학습 GPU·시간이 드는 큰 작업으로, 자율 blend 튜닝 루프 범위를 벗어남. **현 시점 자율 루프는 0.938 천장에서 수렴 — 사용자 판단(모델 재학습 착수 여부)이 필요한 지점.**
