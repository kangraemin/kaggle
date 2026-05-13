# Reflection — trial_056 (sub 48, ralph it.6)

## 결과
- **Public LB: 0.934** — best(0.934, trial_053/054) 동률. 개선 없음.
- Private: 미공개.
- kernel v43 COMPLETE, wall 203.6s, submission mean 0.0425 (trial_054 0.0396 / trial_055 0.0400 대비 상승 — pmix weight↑ 정상).

## 변경사항
1. **회귀 복원**: 15% EffNet 슬롯을 trial_055의 epoch50+distill 10ckpt 멀티로스 앙상블 → epoch50 SoftAUC 5ckpt 단일로 되돌림 (= trial_054 0.934 구성).
2. **EffNet 내부 2pp 재배분**: BLEND_PSEUDOMIX 0.06 → 0.08, BLEND_MWF0 0.07 → 0.05. Perch는 72%로 고정 (trial_036에서 Perch↓는 역효과 확인됨).
   - 의도: EffNet 예산(28%) 안에서, 노이즈 많은 단일 fold0 컴포넌트(mwf0) → 분산이 작은 pmix 5-fold 앙상블 쪽으로 weight 이동.

## 교훈
- **EffNet 내부 weight 미세 재배분(2pp)은 LB granularity(0.001) 아래** — pmix 5-fold가 6% weight에서 trial_053 4-fold 대비 한계효용 0이었는데(trial_054), 8% weight로 올려도 mwf0 fold0와 차이 없음. EffNet 보조 컴포넌트들끼리는 서로 거의 교환 가능(interchangeable)하며, 어느 쪽에 1~3pp를 주든 0.934 평형점이 유지된다.
- **이번 라운드의 진짜 성과**: trial_055(-0.001) 회귀를 깔끔하게 복원해 0.934로 복귀. 멀티로스 앙상블 실험은 닫힌 가지로 확정.
- **trial_052/047 결과와 일관**: weight 재배분(trial_052 4종 weight 동시 조정), mwf0 0.05~0.08 sweep(trial_047) 모두 neutral이었음. 이제 EffNet 28% 예산 내부 재분배는 더 시도 가치 없음 — 데드 엔드.
- 0.934는 trial_053(pmix 2→4-fold 확장) 이후 5번 연속(trial_054, 055는 -0.001, 056) 깨지지 않음. EffNet 보조 컴포넌트 축의 탐색 공간이 사실상 소진됨.

## 다음 시도 제안 (우선순위)
1. **Perch 임계/후처리 축** — 지금까지 모든 시도가 "EffNet 보조 컴포넌트" 한 축에만 집중. Perch가 72% 비중인데 Perch 출력 자체에 손대본 적이 거의 없음. prior_mask(trial_038, Tier C ×0.3)는 이미 들어가 있으나, (a) per-soundscape 분위수 정규화, (b) self-distillation/TTA(Perch 멀티윈도우 평균), (c) Tier 경계 재튜닝(67종 → EDA 재검토) 등 후처리 쪽이 미탐색.
2. **새 1차 모델 추가** — HGNetV2(trial_034 OOF 0.9657로 약했음), ConvNeXt(trial_039~041 플랫폼 silent reject로 좌초)는 닫혔지만, 가벼운 SED 헤드(timm + attention pooling)나 EfficientNet 다른 backbone(B1/B2)을 5초 윈도우로 새로 학습해 5번째 컴포넌트로. EffNet 5-fold(epoch50)와 손실/아키 diversity 확보.
3. **Perch 비중 자체를 건드리는 실험 (조심)** — trial_036에서 Perch↓는 역효과였지만 그건 BLEND 0.25 + epoch50 동시 변경이라 교란됨. 지금 안정된 구성(Perch 72% / EffNet5fold 15% / mwf0 5% / pmix 8%)에서 Perch 72→70%, +2pp는 새 컴포넌트(제안 2)에 주는 식의 격리 실험은 가치 있음. 단, 새 컴포넌트 없이 기존 EffNet에 2pp 더 주는 건 이번에 무효 확인했으니 금지.
4. **앙상블 방식 변경** — 현재 단순 가중 평균. logit 공간 평균 vs prob 평균, 또는 rank-average 비교. 코드 변경 작고 한 번도 안 해봄.

→ **iter 7 권장**: 제안 1(Perch 후처리, per-soundscape 분위수 정규화 또는 멀티윈도우 TTA) 또는 제안 4(rank-average). 둘 다 모델 재학습 불필요, 회귀 위험 낮음, 미탐색 축.
