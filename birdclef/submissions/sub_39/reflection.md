# Sub 39 Reflection — trial_046 effnet_mw_blend

**Base**: Perch 85% + EffNet distill 5fold 15% + prior_mask (sub_37, 0.930)
**Trial**: trial_046 kernel v34

## 결과
- Public: **TBD** (ref 52318158, 2026-05-04 제출, PENDING)

## 변경사항 (sub_37 대비)
- EffNetF0 fold0 standalone 모델 추가 (birdclef2026-effnet-multiwindow-fold0/best_fold0.pth)
- 3-way blend: Perch 80% + EffNet distill 5fold 15% + EffNet fold0 5%
- EffNetF0 아키텍처: stem_conv(1→3) + tf_efficientnetv2_b0(in_chans=3) — distill5fold의 _BirdEffNet과 다른 구조
- _mwf0_spec: MelSpec(n_fft=1024, hop=320, n_mels=128, fmin=50, fmax=14000) + dynamic TOP_DB=80 정규화

## 가설
distill 5fold(Perch 모방 학습)와 다른 학습 방식의 fold0 standalone(일반 CE 학습)이 다른 오류 분포를 가질 수 있음 → diversity 효과

## 교훈
- (점수 확인 후 업데이트)

## 버려야 할 것
- (점수 확인 후 업데이트)

## 유지해야 할 것
- (점수 확인 후 업데이트)

## 다음 가설
- 점수가 0.930 이상이면: BLEND_MWF0 추가 스윕 (0.08, 0.10)
- 점수가 0.930 미만이면: fold0 standalone은 diversity 제공 불가 → blend에서 제거
