# Sub 21 Reflection — trial_034_hgnet_blend

## 결과
- Public: **0.927** (-0.003 vs best 0.930)
- HGNetV2 10% 추가가 오히려 약간 하락

## 변경사항 (sub_20 대비)
- HGNetV2-B0 4-fold 학습 후 blend 노트북에 10% 추가
- 기존: Perch+ProtoSSM + EffNet 15%
- 변경: Perch+ProtoSSM + EffNet 15% + HGNetV2 10%

## 무슨 일이 있었나
- **v1 (ERROR)**: 새 cell을 code cell 번호로 삽입했는데, `.ipynb`는 markdown+code 합산 full index를 씀 → meta_test 정의 이전에 HGNetV2 셀 실행 → NameError
- **v2 (0.858, -0.072)**: HGNetV2 raw logit(mean=-62, range -100~-20)을 기존 scores(range -4~+6)와 scale 맞추지 않고 blend → 전체 score 끌어내려 확률 → 0
- **v3 (0.927, -0.003)**: z-score 정규화 후 blend → 거의 회복. HGNetV2 OOF(0.9657)가 EffNet(0.9792)보다 낮아 noise로 작용

## 교훈
- LSE pooling head 모델의 raw logit은 scale이 극단적으로 다름 (mean=-62). 반드시 정규화 후 blend
- `.ipynb` cell 삽입 시 code cell 번호 ≠ full list index. 키워드 검색으로 위치 확인 필수
- OOF AUC가 낮은 모델은 blend weight를 줄여도 noise 효과 → 순수 다양성이 낮으면 도움 안 됨

## 버려야 할 것
- HGNetV2 10% blend — OOF(0.9657) 낮아서 EffNet보다 약함. 더 강한 모델 없이는 의미 없음
- 낮은 OOF 모델의 앙상블 추가 (threshold: EffNet 0.9792 이상이어야 효과 기대 가능)

## 유지해야 할 것
- HGNetV2 weights 자체 (`ramkang/birdclef2026-hgnetv2-b0-4fold`) — 더 잘 학습시키거나 다른 방식으로 활용 가능
- z-score 정규화 패턴 — 이종 모델 blend 시 필수

## 다음 가설
1. **기존 best(0.930)로 돌아가서 안정 유지** — quota 아끼기
2. **HGNetV2 더 오래 학습** (20→40 epochs) — OOF 올려서 재시도
3. **V18 ProtoSSM 직접 학습** — 공개 weight 의존 제거, 올해 데이터로 학습
