# Reflection — trial_057 (sub 49, ralph it.7)

## 결과
- **Public LB: 0.929** — best(0.934, trial_053/054/056) 대비 **-0.005**. 큰 폭 하락. trial_022~025(pmix 컴포넌트 추가 전, Perch+EffNet 시절) 수준으로 후퇴.
- Private: 미공개.
- kernel v44 COMPLETE, wall 214.0s, dry-run 20 soundscapes. submission 240×235, no NaN, range [0.000475, 0.993394], Final mean 0.2476 (trial_056 0.0425 대비 대폭 상승 — rank-avg는 percentile이 0.5 근처에 모이므로 mean 상승은 정상이고 macro ROC-AUC는 scale-invariant이라 mean 자체는 점수와 무관).

## 변경사항
- **앙상블 fusion 연산자만 교체** (모델·blend weight·후처리 전부 trial_056 동일):
  - 기존(trial_056, 0.934): `final = _perch*(1-Σw) + 0.15*effnet5fold + 0.05*mwf0 + 0.08*pmix` — 가중 logit 합
  - 신규(trial_057): 각 컴포넌트를 클래스별 percentile rank로 변환(`scipy.stats.rankdata` per column → `(rank-1)/(n-1)` ∈ [0,1]) → 같은 weight로 가중 평균 → `clip(1e-4, 1-1e-4)` → `np.log(p/(1-p))`로 logit 공간 복원 → downstream 온도/file-level/rank-aware/delta-shift/prior-mask 파이프라인은 그대로
- cell 62 헤더 코멘트만 trial_057로 갱신, kernel-metadata.json 무변경, kernel v44 push & 제출
- 가설: logit 공간에선 다이내믹 레인지가 큰 컴포넌트가 명목 weight보다 더 기여한다(trial_034 HGNet raw logit scale 불일치 전례) → rank-avg로 각 컴포넌트가 정확히 명목 weight만큼만 ordering에 기여하게 만들면 ±0~+0.001 기대. ralph reflection it.6 제안 #4. "회귀 위험 낮음"으로 판단했었음 — **틀렸다**.

## 교훈
- **logit 공간의 magnitude가 실제 신호를 담고 있다.** Perch(72%)가 어떤 윈도우/클래스에 대해 logit으로 강하게 "present"라고 말하는 것 — 그 *강도* 가 percentile rank로 변환되면 사라진다. 같은 클래스 내 ordering은 보존돼도, 다른 컴포넌트(EffNet 노이즈)와 섞일 때 "강한 confidence는 잘 안 흔들리고 약한 건 잘 흔들린다"는 비선형 효과가 없어져 전체적으로 묽어진다.
- **EffNet 보조 컴포넌트가 logit 가중합에서 명목 weight(15/5/8%)보다 큰 effective weight로 기여하던 것이 사실은 +방향이었다.** EffNet 헤드 logit이 Perch 출력보다 다이내믹 레인지가 커서 logit 합에서 과대 기여하고 있었는데, 그게 trial_046(+0.002)·trial_053(+0.001)의 이득 일부였던 것. rank-avg로 "공정하게" 명목 weight만 주자 오히려 손해 → 즉 0.934 구성은 의도치 않게 EffNet에 더 무게를 주고 있었고 그게 맞았다.
- **앙상블 *방식* 변경(logit vs rank 평균)은 닫힌 가지로 확정.** prob-space 평균(`Σw·sigmoid(logit)`)도 magnitude를 유지하지만 logit 합과 곱셈적 vs 덧셈적 차이가 있어 위험 추정 — 시도 가치 낮음. 현재의 단순 가중 logit 합이 베스트.
- **회귀 위험 평가가 틀렸다** — "코드 변경 작고 모델·weight·후처리 동일하니 위험 낮음"이 아니라, *fusion 연산자*는 0.934를 만든 핵심 메커니즘이라 바꾸면 크게 흔들린다. 다음부터 "후처리/weight는 marginal, fusion 연산자/컴포넌트 구성은 core" 구분.

## 다음 시도 (필수 → 권장)
0. **[필수] trial_058: trial_056 logit-space 블렌드 복원** — cell 65를 `2c2a948^`(= trial_056) 버전으로 되돌려 0.934 회귀 확인. ralph it.7 -0.005 청소.
1. **Perch 후처리/TTA** — rank-avg 실패는 *전체* 앙상블을 rank화한 게 원인이지 Perch 윈도우 단위 처리가 원인이 아님. Perch 12-윈도우 멀티윈도우(shift) 평균 또는 per-soundscape 분위수 정규화는 별개 미탐색 축. (단 per-class monotonic 변환 — 온도/per-class threshold/prior_mask 배수 — 은 macro ROC-AUC에 무효임이 이미 검증됨. cross-file/cross-window 변환만 의미.)
2. **새 5번째 1차 모델** — EfficientNet B1/B2 또는 SED(attention pooling) 헤드 5초 윈도우 신규 학습 → 손실/아키 diversity. 새 컴포넌트면 Perch 72→70% 격리 실험도 가능.
3. **새 임베딩 소스 probe** — Perch 외 다른 사전학습 오디오 임베딩(AudioMAE/BEATs 등)으로 probe 추가.
