# Sub 46 Reflection — trial_054 pmix_5fold_blend

**Base**: trial_053 (Perch 72% + epoch50-5fold 15% + mwf0-fold0 7% + pmix(fold0..3) 6%) — 0.934 best
**Trial**: trial_054 kernel v41 (ralph-x iteration 4)

## 결과
- Public: **0.934** ➖ best 동률 (trial_053 0.934와 동일, 회귀 없음)

## 변경사항
- pseudo-mix 컴포넌트 앙상블 확장: `fold0..3` 4-fold → `fold0..4` 5-fold
  - 로컬 `models/effnet_pseudo_mix/best_fold4.pth` (5/12 09:49) 학습 완료 (fold0~3과 동일 arch·동일 학습 스크립트)
  - Kaggle dataset `ramkang/birdclef2026-effnet-pseudo-mix` v4 (best_fold0..4.pth 5개), `dataset-metadata.json` title 'BirdCLEF2026 EffNet Pseudo-Mix 5fold'로 갱신
- **노트북 코드 변경 0** — pmix loader가 이미 `glob('best_fold*.pth')` + `mean(axis=0)` 구조라 dataset만 갱신하면 자동 5-fold 앙상블. Cell62/63/65 코멘트만 trial_054로 갱신
- blend weight·다른 컴포넌트·prior_mask 전부 trial_053 그대로 (BLEND_EFFNET=0.15, MWF0=0.07, PSEUDOMIX=0.06, EFFNET_S=0 → Perch 72%) → fold4 추가 효과만 격리

## 검증
- kernel v41 COMPLETE (wall 216.0s, dry-run 20 soundscapes): `EffNet Pseudo-Mix loaded 5 folds: [best_fold0..4.pth]`, `blend: Perch 72% + EffNet5fold 15% + fold0-B0 7% + fold0-S 0% + pmix 6%`, submission 240×235 no NaN, range [4.6e-15, 0.998], mean 0.0396 (trial_053과 동일)

## 가설 (검증됨 — 한계효용 0 쪽으로 확정)
pmix를 5-fold 앙상블로 만들면 4-fold보다 분산이 1/5로 더 줄어 컴포넌트 품질이 미세 향상 → 동일 6% weight에서 +0.000~0.001 기대했으나 → **0.934 동률 (한계효용 0)**.
- 흐름: trial_050(fold0 1개, 0.933) → trial_051(fold0+1 2-fold, 0.933 동률) → trial_053(fold0..3 4-fold, **0.934** +0.001) → trial_054(fold0..4 5-fold, 0.934 동률)
- 즉 보조 컴포넌트(6% weight) 내부 fold 앙상블의 LB 이득은 **2→4-fold 구간에서 한 번 +0.001 점프했고 4→5-fold는 0**. 분산 추가 절감(1/4→1/5)이 macro ROC-AUC granularity(~0.001) 아래로 떨어진 것. fold 4개에서 사실상 포화.
- 회귀 없음(0.934 유지) → 5-fold 구성 그대로 best 유지. fold4 학습 비용은 이미 들였으니 sunk cost, 굳이 4-fold로 되돌릴 이유 없음.

## 교훈
- **보조 컴포넌트 fold 앙상블은 4개에서 포화** — 추론 코드 변경 없이 dataset 버전만 올리는 가장 싼 개선이지만 2→4-fold에서 +0.001 한 번 먹은 뒤로는 더 늘려도 LB granularity 아래. mwf0(현재 1-fold standalone)도 같은 한계 예상되므로 4-fold까지만 만들면 충분.
- 이제 같은 트릭(같은 arch fold 더 쌓기)으로는 0.935 못 감. 다음은 **새 diversity 소스**가 필요 — 다른 backbone(EffNetV2-S+Xeno, trial_049), 또는 5fold 컴포넌트(BLEND_EFFNET=0.15) 자체를 multi-loss 앙상블로 교체.

## 다음 가설 (ralph it.5+)
- **mwf0 앙상블화**: mwf0(현재 fold0 standalone 1개)를 fold0..3 4-fold로 — pmix 트릭 재적용. mwf0-train 노트북으로 fold1~3 학습 필요 (로컬 학습 시간 듦). pmix 4-fold 효과(+0.001)와 유사 기대.
- **BLEND_PSEUDOMIX sweep**: 5-fold로 안정해졌으니 0.06 → 0.08/0.10 시도. 단 trial_047(MWF0 0.05→0.08 무효과)·trial_052(reweight 무효과) 전례 있어 기대 낮음.
- **EffNetV2-S + Xeno pretrain** (trial_049): 학습 완료 시 BLEND_EFFNET_S 활성화 — 유일하게 남은 새 backbone diversity 카드.
- **5fold 컴포넌트 강화**: epoch50 단일 → epoch50+softauc+distill 멀티 앙상블로 BLEND_EFFNET 15% 슬롯 교체. 추론 시간 2~3배(현재 wall 216s → 여유 있음, hidden test 증가 시 타임아웃 주의).
- post-processing: prior mask 임계값 변형, file-level 정규화 — EDA 발견 재활용 (지금까지 prior mask는 neutral).
