## Submission 79 Reflection

**Base**: trial_080 sed_up (kernel v72, 0.938 — best)
**Trial**: trial_086 effnet_up

### 결과
- Public: **0.937** (best 0.938 대비 **-0.001 하락**)
- 제출: 2026-06-01 00:04 (UTC 자정 직후 자동 제출) → COMPLETE
- 참고: UTC 05-31 일일 제출 한도 5/5 소진 → submit 400 → UTC 자정 리셋 후 자동 제출 예약으로 처리

### 변경사항 (이전 sub 대비)
- BLEND_EFFNET 0.05→0.10, BLEND_PROTO 0.45→0.40 (-5pp 추가 차출)
- SED 0.40/Perch 0.10 유지, kernel v77→v78

### 교훈
- **EffNet 기여 한계 확정**: 5%(trial_085)=동률, 10%(trial_086)=하락. EffNet 비중을 늘릴수록 강한 ProtoSSM(50→40%)이 약화돼 LB 하락.
- 다른 모델 가족(EfficientNetV2-B0)이라도 단독 성능이 Proto/SED보다 약하면 blend에서 noise 역할. library `blend-ratio-weak-model-hurts-ensemble` 재확인.
- 모델 다양성 가설(row-rank 변경) 기각 — EffNet의 다른 오류 패턴이 LB 개선으로 이어지지 않음.

### 버려야 할 것
- EffNet 비중 ↑ 방향 (5%가 상한, 그나마 동률).
- "다른 모델 가족 추가하면 다양성으로 천장 돌파" 가설.
- 보조 컴포넌트(EffNet/mwf0/pmix/distill) 단독 추가로 0.938 돌파 시도 — trial_059/071/085/086 전부 무효/하락으로 패턴 확정.

### 유지해야 할 것
- **trial_080 best 0.938** (SED40/Proto50/Perch10) — 최종 제출 후보.
- blend weight space는 완전 포화. 3-컴포넌트(Perch/Proto/SED) 구성이 최적.

### 다음 가설
blend weight·scale·보조컴포넌트 전 축 포화(trial_062~086, 약 15+ trial 0.938~0.934). **남은 유효 레버는 모델/추론 자체 개선뿐**:
1. **ProtoSSM TTA shift 확대** — 최대 비중(50%) Proto의 추론 품질 직접 향상. `tta_shifts` 늘려 시간축 앙상블 강화 (코드의 CFG tta_shifts).
2. **SED 5-fold → 개별 fold weight 튜닝** — SED 내부 fold mean-pool 대신 fold별 품질 가중.
3. **(큰 작업) 모델 재학습** — 현 컴포넌트 조합이 천장이면 base 모델 자체를 개선해야 함. 자율 루프 범위 밖, 사용자 판단 필요.

⚠️ weight 미세조정·컴포넌트 추가는 더 시도하지 않음 (포화 확정). 다음은 추론 파라미터(TTA) 방향.
