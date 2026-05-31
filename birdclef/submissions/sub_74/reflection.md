## Submission 74 Reflection

**Base**: trial_080 sed_up (kernel v72, 0.938 — 현재 best)
**Trial**: trial_081 proto_path_fix

### 결과
- Public: **0.938** (trial_080과 동률, best 동률)
- 제출: 2026-05-31 07:11 → COMPLETE

### 변경사항 (이전 sub 대비)
- ProtoSSM pretrained mount 경로 수정: `/kaggle/input/datasets/hideyukizushi/sgkfk-202604041716/` → `/kaggle/input/sgkfk-202604041716/`
- blend weights 무변경 (Proto 50% + SED 40% + Perch 10%)
- kernel v72 → v73

### 교훈
- 경로 수정이 채점 결과를 전혀 바꾸지 않음 (0.938 → 0.938). 기대치(0.945+)는 빗나감.
- 두 가지 해석 가능: (a) 경로 수정 전에도 ProtoSSM pretrained가 이미 정상 로드되고 있었거나, (b) 경로가 틀려 fallback이 돌고 있었으나 최종 blend 출력에는 영향이 없었음.
- 어느 쪽이든 "pretrained 적용 시 대폭 개선" 가설은 기각. ProtoSSM 컴포넌트는 현재 구성에서 이미 천장에 가까움.

### 버려야 할 것
- "ProtoSSM 경로/pretrained 손보면 0.945+" 가설 — 검증 결과 효과 0.
- 단순 경로/로딩 수정으로 점수 개선 기대하는 방향.

### 유지해야 할 것
- trial_080의 SED 40% + Proto 50% + Perch 10% 구성 (0.938 best).
- SED 비중이 핵심 레버였다는 사실 (trial_079 0.935 → trial_080 0.938).

### 다음 가설
1. **SED 비중 추가 상향** — trial_079(SED 18%)→trial_080(SED 40%)에서 +0.003. SED 50~60%까지 push해 포화 지점 탐색 (Proto 동반 하향). 가장 검증된 방향.
2. **SED standalone 점수 측정** — SED가 이렇게 효과적이면 SED 단독 LB가 어디인지 확인 후 blend 상한 추정.
3. **ProtoSSM 5-fold 앙상블** — 단일 ProtoSSM이 천장이면 fold 앙상블로 신호 증폭 (pmix에서 검증된 패턴).
