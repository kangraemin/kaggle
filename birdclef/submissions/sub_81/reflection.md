## Submission 81 Reflection

**Base**: trial_080 sed_up (kernel v72, 0.938 — best)
**Trial**: trial_088 proto_dual

### 결과
- Public: **0.938** (best 동률)
- 제출: 2026-06-01 08:32 → COMPLETE

### 변경사항 (이전 sub 대비)
- 자체학습 ProtoSSM(우리 8-soundscape, v80) + 외부 pretrained(작년 데이터) 추론 평균 앙상블
- `copy.deepcopy(model)`로 외부 구조 복제 + 자체 weight load → temporal_shift_tta 추론 → proto_scores_flat 평균
- dataset `ramkang/birdclef2026-self-protossm` 마운트, blend best 유지

### 교훈
- **dual ProtoSSM 앙상블도 0.938 동률.** 자체학습·외부 두 모델이 같은 ProtoSSMv2 아키텍처 + Perch 임베딩 입력이라 예측 상관이 높아 평균해도 row-rank 거의 안 바뀜. 다른 데이터(우리 8-soundscape vs 작년) 학습이 만든 오류패턴 차이가 LB를 못 움직임.
- **이번 세션의 큰 교정**: 직전에 자체학습을 OOF 0.6513만 보고 "약하다"고 성급히 기각했으나, 실제 제출하니 v80=0.938(외부와 동률). **OOF(8-soundscape, fold당 1~2개)는 노이즈일 뿐 LB 추정에 못 쓴다** — 실측이 답. 사용자 지적("니맘대로 종료")이 정확했음.

### 버려야 할 것
- OOF noise를 LB 추정으로 쓰는 것 (8-soundscape OOF는 ±0.3 요동).
- 같은 아키텍처 모델끼리 앙상블로 다양성 기대 (상관 높아 무효).

### 유지해야 할 것
- **trial_080 best 0.938** (SED40/Proto50/Perch10) — 최종 제출 후보.
- 실측 우선 원칙: OOF/추측으로 기각 말고 제출 한도 있으면 LB로 판별.

### 다음 가설
**모든 레버를 실제 제출로 소진 — 0.938이 실측 천장 확정**:
- 파라미터 4축(weight/scale/컴포넌트/추론): 포화 (trial_062~087)
- ProtoSSM 자체 재학습(우리 데이터): 0.938 동률 (v80 실측)
- 자체+외부 dual 앙상블: 0.938 동률 (v81 실측)
- 데이터 확장 불가: train_audio 없음(soundscape 대회), ss10k_pseudo는 Perch 임베딩 없어 ProtoSSM 학습 불가

남은 건 (a) 완전히 새로운 1차 모델 학습(SOTA 오디오 백본 fine-tune, 큰 작업·이틀 빠듯) 또는 (b) 0.938 확정. 파라미터·재학습·앙상블 레버는 전부 실측으로 막힘.
