# Sub 51 Reflection — trial_059 distill_add

**Base**: trial_058 (Perch 72% + epoch50-5fold 15% + mwf0-fold0 5% + pmix(fold0..4) 8%) — 0.934 best 동률
**Trial**: trial_059 kernel v46 (ralph-x iteration 9)

## 결과
- Public: **0.934** ➖ best 동률 (trial_053/054/056/058과 동일). +0 — 회귀도 개선도 아니다.
- Private: 미공개
- kernel v46 COMPLETE (wall 259.7s, +124s vs trial_058 136s — distill 5-fold 추가 추론), submission 240×235, no NaN, range [2.30e-15, 0.997], Final mean 0.0452 (trial_058 0.0425 +0.0027 — distill 별도 컴포넌트 추가 영향, 정상)

## 변경사항
- **distill_5fold(KD L2-MSE 학습)을 5번째 별도 컴포넌트로 추가** (trial_055는 epoch50 슬롯 안에 평균으로 섞었는데, 이번엔 BLEND_DISTILL=0.03 신규 weight로 분리):
  - Cell62: `_distill_models` 신규 loader (slug `birdclef2026-effnet-5fold-distill`, fold0..4, `_BirdEffNet` arch + `_EffSpec(n_fft=2048, n_mels=256)` 동일), inference loop 내 `if _distill_models:` 분기로 `distill_logits` 산출, `BLEND_MWF0 0.05 → 0.02`, `BLEND_DISTILL = 0.03` 신규
  - Cell65: fusion에 `+ BLEND_DISTILL * distill_logits` 항 추가, Perch 계수 `(1 - 0.15 - 0.02 - 0.08 - 0.03) = 0.72`로 유지
- 모델·다른 컴포넌트·prior_mask·downstream 후처리(온도/file-level/rank-aware/delta-shift) 전부 trial_058 그대로 → distill 별도 weight 효과만 격리

## 검증
- Local 로드 검증 통과: `Loaded 5 EffNet folds (global pool, epoch50 SoftAUC)`, `Loaded 5 distill folds (KD L2-MSE)`, `EffNetF0 fold0 loaded`, `EffNet Pseudo-Mix loaded 5 folds`
- blend print: `Perch 72% + EffNet5fold 15% + fold0-B0 2% + fold0-S 0% + pmix 8% + distill 3%`
- final_test_scores (logit) range [-11.924, 6.958], mean -2.936 (trial_058 대비 살짝 넓은 분포)

## 가설 (검증됨 — KD 손실 diversity는 별도 weight로도 ineffective)
SoftAUC(epoch50)와 KD L2-MSE는 손실 함수가 다르므로 logit-space에서 다른 정보를 담을 것 → 별도 weight 부여시 epoch50 0.15와 5x weight ratio로 명확히 분리되어 +0.001 정도 기대 (trial_057 통찰: logit magnitude가 effective contribution이므로 명목 0.03도 무게 있을 수 있음) → **0.934 동률 (한계효용 0)**.
- trial_055(epoch50+distill 같은 슬롯 평균, 0.075/0.075): -0.001
- trial_059(epoch50 0.15 + distill 0.03 별도): 0.000
- 둘 다 KD 손실 컴포넌트가 LB granularity(~0.001) 안에서 가치가 없음. mwf0(노이즈 큰 단일 fold)에서 3pp를 빼와 distill에 줘도, EffNet 합산 28% 안에서의 어떤 재분배도 0.934 천장을 못 깬다 — sub_45/46 reflection 진단 "EffNet 28% 예산 내부 재분배 데드 엔드, 새 모델/diversity 축 필요" 재확인.
- mwf0 weight 약화(0.05→0.02)도 LB 무영향 — mwf0이 fold0 단일이라 노이즈 크다고 봤는데 5pp든 2pp든 LB에는 보이지 않을 만큼 미미한 기여.

## 교훈
- **"별도 weight"가 "같은 슬롯 평균"보다 나을 거라는 직관이 틀렸다.** trial_055가 -0.001이고 trial_059가 0인 건 별도 weight가 약간 덜 나쁘다는 정도지, +방향이 아니다. KD 손실 함수 자체가 SoftAUC + Perch 출력과 logit-space에서 ortho 정보를 거의 안 담는다는 신호.
- **EffNet 28% 예산 내 모든 재분배·다양화가 닫혔다** — fold 앙상블 확장(trial_054 4→5fold 0, trial_056 mwf0↔pmix 2pp 재분배 0), 손실 다양화(trial_055 평균 -0.001, trial_059 별도 0), weight 미세조정(trial_047 mwf0 0.05→0.08 0, trial_052 합산 23→28% 0). 이 축에서 추가 시도는 시간 낭비.
- **0.934 천장을 깨려면 Perch 72% 또는 EffNet 28% 비율 자체를 흔드는 새 구성이 필요** — 즉 새 1차 모델 컴포넌트(다른 backbone 또는 다른 임베딩 소스). trial_046 EffNet 추가로 0.930→0.932(+0.002), trial_053 pmix 4-fold로 0.933→0.934(+0.001), 그 이후로 6연속 0.934 — 새 컴포넌트 가설이 강하게 지지된다.

## 다음 가설 (ralph it.10+, 우선순위)
1. **[필수] EffNet 28% 예산 내 재분배·KD 다양화 탐색 중단** — trial_054/055/056/057/058/059가 모두 동률 또는 회귀. 같은 풀에서 더 흔들 가치 없다.
2. **새 1차 모델 컴포넌트 추가** (가장 가능성 높음):
   - **AudioMAE 또는 BEATs probe**: Perch 외 다른 사전학습 오디오 임베딩으로 frozen probe 학습 → 별도 5%~10% blend weight. 새 임베딩 소스 = 새 diversity 축.
   - **EfficientNetV2-S Xeno-pretrain (trial_049 idle)**: 같은 EffNet 가족이지만 다른 backbone scale + 다른 pre-train (XCL Xeno). 학습 완료시 BLEND_EFFNET_S 활성화 후보.
   - **SED (attention pooling) 헤드**: 같은 EffNet backbone이라도 global mean pool → attention 갈아끼우면 다른 정보 추출 가능. 학습 비용 듦.
3. **Perch 자체 강화** (rank-avg 실패의 교훈 반대편 — Perch confidence가 핵심 신호이므로 Perch 정확도 자체를 올리는 게 더 안전):
   - **Perch 멀티윈도우 평균**: 현재 5초 1회 → 5초 ±shift 2~3회 평균. 추론 wall 약 2~3x. hidden test 증가 시 timeout 위험.
   - **Perch per-soundscape 분위수 정규화**: cross-window 정보 사용 (per-class monotonic은 macro-AUC 무효, cross-window는 유효 가능성).
4. **새 후처리 축 탐색** — 단 단일 클래스 monotonic 변환(온도, prior_mask 배수)은 검증 끝. cross-file/cross-window/cross-class만 가치.
