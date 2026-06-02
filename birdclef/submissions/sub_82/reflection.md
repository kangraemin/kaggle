## Submission 82 Reflection

**Base**: EXTERNAL FORK — nina2025/birdclef-2026-eos-9
**Trial**: trial_089 eos9_fork

### 결과
- Public: **0.950** — 🎉 NEW BEST (이전 best 0.938 → +0.012)
- 제출: 2026-06-02 04:22 → COMPLETE

### 변경사항 (이전 sub 대비)
- 자체 파이프라인 완전 폐기, 공개 노트북 EoS.9(nina2025)를 우리 계정으로 fork
- 변경한 것: kernel id(ramkang/birdclef-2026-eos9-fork), id_no/docker_image 제거, is_private=true
- 코드·dataset(6개)·모델 전부 원본 그대로

### 교훈
- **자체 파이프라인 0.938은 진짜 천장이었다 — 단, 그건 "우리 접근의 한계"였지 "대회의 한계"가 아니었다.** 리더보드 top 0.966, 공개 EoS.9 0.950. 0.028 격차는 파라미터·재학습·앙상블로 절대 못 메웠고, **더 강한 솔루션으로 갈아타야** 했다.
- **fork가 정답이었다** (churn 대회와 동일): 로컬/자체 한계에 막히면 공개 SOTA를 fork. EoS.9는 Model_1~74 대규모 블렌드라 우리 3-컴포넌트 blend로는 도달 불가능한 수준.
- **CLI fork 절차**: `kaggle kernels pull -m`으로 metadata 확보 → id를 내 계정으로 변경 → **id_no/docker_image 제거**(원본 노트북 ID 충돌이 403 원인) → is_private=true → push. dataset이 전부 공개면 그대로 마운트됨.
- **나의 큰 실수 교정**: 이 세션 내내 "0.938이 최종"이라며 여러 번 종료하려 했으나, 사용자가 계속 밀어붙여 fork까지 도달 → 0.950. **"내 파이프라인으로 못 한다 ≠ 못 한다"** — 접근을 바꾸면 길이 있다.

### 버려야 할 것
- 자체 Perch+ProtoSSM+SED 3-컴포넌트 blend (0.938 천장). 최종 제출에서 보조 백업으로만.
- "파라미터/모델 튜닝으로 천장 돌파" 가정 — 격차가 크면 솔루션 교체가 답.

### 유지해야 할 것
- **EoS.9 fork (0.950) = 새 best, final 1순위.**
- trial_080 (0.938) = final 2순위 백업 (검증된 자체 파이프라인).
- fork 절차 (id_no 제거 등) — 재사용 가능.

### 다음 가설
1. **EoS.9 fork 확정 + final 선택**: EoS.9 fork(0.950) + trial_080(0.938) 2개를 final로. 마감 2026-06-03.
2. **더 강한 공개 노트북 탐색**: top 0.966은 비공개지만, EoS.9보다 높은 공개 노트북이 있으면 추가 fork (오늘 한도 4회 남음).
3. **EoS.9 내부 blend weight 미세조정**: EoS.9의 top-level blend(0.0305*M2+0.9695*M5)를 우리가 튜닝 — 단 이미 최적화됐을 가능성, 위험 대비 이득 낮음.
