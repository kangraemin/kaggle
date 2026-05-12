# Sub 46 Reflection — trial_054 pmix_5fold_blend

**Base**: trial_053 (Perch 72% + epoch50-5fold 15% + mwf0-fold0 7% + pmix(fold0..3) 6%) — 0.934 best
**Trial**: trial_054 kernel v41 (ralph-x iteration 4)

## 결과
- Public: **채점중** (제출 PENDING — BirdCLEF eval 지연)

## 변경사항
- pseudo-mix 컴포넌트 앙상블 확장: `fold0..3` 4-fold → `fold0..4` 5-fold
  - 로컬 `models/effnet_pseudo_mix/best_fold4.pth` (5/12 09:49) 학습 완료 (fold0~3과 동일 arch·동일 학습 스크립트)
  - Kaggle dataset `ramkang/birdclef2026-effnet-pseudo-mix` v4 (best_fold0..4.pth 5개), `dataset-metadata.json` title 'BirdCLEF2026 EffNet Pseudo-Mix 5fold'로 갱신
- **노트북 코드 변경 0** — pmix loader가 이미 `glob('best_fold*.pth')` + `mean(axis=0)` 구조라 dataset만 갱신하면 자동 5-fold 앙상블. Cell62/63/65 코멘트만 trial_054로 갱신
- blend weight·다른 컴포넌트·prior_mask 전부 trial_053 그대로 (BLEND_EFFNET=0.15, MWF0=0.07, PSEUDOMIX=0.06, EFFNET_S=0 → Perch 72%) → fold4 추가 효과만 격리

## 검증
- kernel v41 COMPLETE (wall 216.0s, dry-run 20 soundscapes): `EffNet Pseudo-Mix loaded 5 folds: [best_fold0..4.pth]`, `blend: Perch 72% + EffNet5fold 15% + fold0-B0 7% + fold0-S 0% + pmix 6%`, submission 240×235 no NaN, range [4.6e-15, 0.998], mean 0.0396 (trial_053과 동일)

## 가설
pmix를 5-fold 앙상블로 만들면 4-fold보다 분산이 1/5로 더 줄어 컴포넌트 품질이 미세 향상 → 동일 6% weight에서 +0.000~0.001 기대.
- 근거: trial_051(fold0+fold1 2-fold)은 동률(0.933)이었으나 trial_053(fold0..3 4-fold)은 +0.001(0.934). 분산 절감이 임계치를 넘으면 LB granularity(~0.001)에 반영됨.
- 단 한계효용 체감 예상: 2→4 fold에서 분산 1/2 추가 절감했지만, 4→5는 1/5만큼만 더 줄어듦. 동률(0.934 유지)일 가능성도 높음.

## 다음 가설 (ralph it.5+)
- pmix 동률이면: mwf0(현재 fold0 standalone 1개)도 fold1+ 학습해 앙상블화 — pmix와 같은 트릭 (mwf0-train 노트북으로 다른 fold 학습)
- pmix +0.001이면: BLEND_PSEUDOMIX 0.06 → 0.08~0.10 sweep (5-fold로 더 안정해졌으니 weight 늘릴 여지)
- EffNetV2-S + Xeno pretrain (trial_049 학습) 완성 시 BLEND_EFFNET_S 활성화 — 새 backbone diversity
- 5fold 컴포넌트(epoch50, BLEND_EFFNET=0.15)도 epoch50+softauc+distill 멀티 앙상블로 강화 가능하나 추론 시간 ↑ → hidden test 타임아웃 주의 (현재 wall 216s, 여유 충분)
