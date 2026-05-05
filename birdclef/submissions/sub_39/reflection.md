# Sub 39 Reflection — trial_046 effnet_mw_blend

**Base**: Perch 85% + EffNet distill 5fold 15% + prior_mask (sub_37, 0.930)
**Trial**: trial_046 kernel v34

## 결과
- Public: **0.932** ✅ **new best** (+0.002)

## 변경사항 (sub_37 대비)
- EffNetF0 fold0 standalone 모델 추가 (birdclef2026-effnet-multiwindow-fold0/best_fold0.pth)
- 3-way blend: Perch 80% + EffNet distill 5fold 15% + EffNet fold0 5%
- EffNetF0 아키텍처: stem_conv(1→3) + tf_efficientnetv2_b0(in_chans=3) — distill5fold의 _BirdEffNet과 다른 구조
- _mwf0_spec: MelSpec(n_fft=1024, hop=320, n_mels=128, fmin=50, fmax=14000) + dynamic TOP_DB=80 정규화

## 가설
distill 5fold(Perch 모방 학습)와 다른 학습 방식의 fold0 standalone(일반 CE 학습)이 다른 오류 분포를 가질 수 있음 → diversity 효과

## 교훈
- EffNet fold0 standalone(CE 학습)은 distill 5fold와 다른 오류 분포를 가짐 → 5%만 blend해도 +0.002 효과
- Perch를 80%로 낮춰도 점수 유지됨 (85% → 80% 비율 변경이 neutral 이상)
- 3-way blend가 2-way 대비 실질적 개선을 줬다는 것이 확인됨

## 버려야 할 것
- ConvNeXt-Base: 추론 시간 초과 (경쟁 eval 환경에서 운영 불가)
- Gaussian smoothing: 역효과 (0.930 → 0.928)
- Standalone EffNet without Perch: 0.836로 독립 운영 불가

## 유지해야 할 것
- Perch ONNX (backbone): 여전히 압도적 기여
- EffNet distill 5fold 15%: 안정적 기여
- EffNet fold0 standalone 5%: diversity 효과 확인됨
- prior_mask (Tier C ×0.3): 최소한 neutral, 일부 케이스에서 도움

## 다음 가설
- BLEND_MWF0 sweep (0.08, 0.10): fold0 비중 올리면 더 개선 가능성
- fold1~4 standalone 추가: 5-way blend로 다양성 극대화
- EffNet multiwindow 5fold (distill 방식 아닌 CE 방식) 전체 fold blend
