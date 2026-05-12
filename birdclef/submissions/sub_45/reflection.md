# Sub 45 Reflection — trial_053 pmix_4fold_blend

**Base**: trial_052 (Perch 72% + epoch50-5fold 15% + mwf0-fold0 7% + pmix(fold0+fold1) 6%) — 0.933 동률
**Trial**: trial_053 kernel v40 (ralph-x iteration 3, 재개)

## 결과
- Public: **0.934** ✅ **new best** (+0.001 vs trial_050/051/052 0.933)

## 변경사항
- pseudo-mix 컴포넌트 앙상블 확장: `fold0+fold1` 2-fold → `fold0..3` 4-fold
  - 로컬에서 `models/effnet_pseudo_mix/best_fold2.pth` (5/12 04:43), `best_fold3.pth` (5/12 07:04) 학습 완료 (각 val AUC ~0.98, fold0/fold1과 동일 arch·동일 학습 스크립트)
  - Kaggle dataset `ramkang/birdclef2026-effnet-pseudo-mix` v3 (best_fold0..3.pth 4개), `dataset-metadata.json` 신규 생성
- **노트북 코드 변경 0** — pmix loader가 이미 `glob('best_fold*.pth')` + `mean(axis=0)` 구조라 dataset만 갱신하면 자동 4-fold 앙상블. Cell62/63/65 코멘트만 trial_053으로 갱신
- blend weight·다른 컴포넌트·prior_mask 전부 trial_052 그대로 (BLEND_EFFNET=0.15, MWF0=0.07, PSEUDOMIX=0.06, EFFNET_S=0 → Perch 72%) → fold2+fold3 추가 효과만 격리

## 검증
- kernel v40 COMPLETE (wall 190.6s, dry-run 20 soundscapes): `EffNet Pseudo-Mix loaded 4 folds: [best_fold0..3.pth]`, `blend: Perch 72% + EffNet5fold 15% + fold0-B0 7% + fold0-S 0% + pmix 6%`, submission 240×235 no NaN, range [4.8e-15, 0.998], mean 0.0396 (trial_052 0.0395와 사실상 동일)

## 가설 (검증됨, 단 의외성 있음)
pmix를 4-fold 앙상블로 만들면 단일/2-fold보다 노이즈가 줄어 컴포넌트 품질이 소폭 올라 동일 6% weight에서 +0.000~0.001 기대 → **+0.001 확인**.
- 의외점: trial_051에서 fold0 → fold0+fold1 2-fold 확장은 **동률(0.933)** 이었음. 2-fold의 분산 절감은 LB에 안 보였는데 4-fold(분산 1/4)에서는 임계치를 넘어 macro ROC-AUC에 반영됨. blend weight가 6%로 작아도 컴포넌트 내부 앙상블 품질이 LB granularity(~0.001)를 넘으면 반영된다는 신호.

## 교훈
- **작은 weight(6%)의 보조 컴포넌트라도 fold 앙상블 수를 늘리면 LB가 움직인다** — 단 2-fold로는 부족, 3~4-fold는 되어야 함. 추론 코드 변경 없이 dataset 버전만 올리면 되는 가장 싼 개선.
- 0.930 → 0.932(mwf0 추가) → 0.933(pmix fold0 추가) → 0.934(pmix 4-fold): EffNet 보조 컴포넌트 누적·강화 전략이 계속 작동 중. Perch backbone이 압도적이지만 그 위에 다양·안정한 EffNet을 작은 weight로 쌓는 게 +0.001씩 먹힌다.
- trial_052의 weight 재배분(Perch 77→72%)은 회귀가 아니었으므로(0.933 동률) 그 위에서 컴포넌트 품질만 올린 게 깔끔하게 효과를 봄.

## 유지해야 할 것
- Perch ONNX backbone (압도적)
- epoch50-5fold 15%, mwf0-fold0 7%, pmix-fold0..3 6% (Perch 72%) — 현재 best 구성
- prior_mask (Tier C ×0.3): neutral~약간 도움

## 다음 가설 (ralph it.4+)
- pmix fold4 학습 완료 시 5-fold로 확장 (또는 5fold 학습 후 전체 CV로 더 안정화)
- mwf0(현재 fold0 standalone 1개)도 fold1+ 학습해 mwf0 앙상블화 — pmix와 같은 트릭
- BLEND_PSEUDOMIX 0.06 → 0.08~0.10 sweep (4-fold로 안정해졌으니 weight를 더 줄 여지)
- EffNetV2-S + Xeno pretrain (trial_049 학습) 완성 시 BLEND_EFFNET_S 활성화 — 새 backbone diversity
- 5fold 컴포넌트(epoch50, BLEND_EFFNET=0.15)도 epoch50+softauc+distill 멀티 앙상블로 강화 가능하나 추론 시간 2~3배 → 실제 hidden test 타임아웃 주의
