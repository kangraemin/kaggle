# Sub 41 Reflection — trial_048 distill30_swap

**Base**: Perch 80% + EffNet distill30 KD 5fold 15% + EffNet fold0 5%
**Trial**: trial_048 blend kernel v36

## 결과
- Public: **0.931** ❌ best 대비 -0.001 (trial_046/047의 0.932 미달)

## 변경사항 (sub_40 대비)
- 5fold 컴포넌트 교체: `birdclef2026-effnet-5fold-epoch50` (SoftAUC 50ep) → `birdclef2026-effnet-5fold-distill` (KD 30ep)
- 블렌드 가중치 동일 유지 (Perch 80% + 5fold 15% + fold0 5%)
- 단일 변수 실험 (모델만 교체)

## 교훈
- epoch50 SoftAUC(0.932) > distill30 KD(0.931): SoftAUC 5fold 50ep 쪽이 더 나음
- distill30 KD는 30ep 밖에 안 학습되었고, SoftAUC는 50ep → epoch 수 차이가 영향 가능성
- 혹은 SoftAUC loss 자체가 macro-AUC 최적화에 더 직접적이라 기여
- 두 효과를 분리하려면 distill30 + 50ep, KD + SoftAUC 등 추가 실험 필요

## 버려야 할 것
- distill30 KD 5fold (epoch50 SoftAUC보다 LB 낮음): 현재 blend에서 제외

## 유지해야 할 것
- epoch50 SoftAUC 5fold가 현재 5fold 컴포넌트로 최선
- 3-way blend 구조 (Perch 80% + 5fold 15% + fold0 5%)

## 다음 가설
- fold1~4 standalone 추가 (현재 fold1 CE 훈련 중, ~9h): 3-way → 7-way blend diversity 확대
- BirdCLEF 2025 pretrained checkpoint 활용 (discussion에서 0.941 달성 사례 확인): Xeno-Canto 전체 pretrained weight가 주요 차별점
- pseudo-labeling on unlabeled soundscapes: cudacoding이 0.950+ 달성한 방법 (+0.010 이상 기대)
