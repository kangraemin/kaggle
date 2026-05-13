# Sub 53 Reflection — trial_061 convnext_5fold

**Base**: trial_060 (Perch 72% + EffNet5fold 15% + mwf0 2% + pmix 8% + distill 0% + convnext fold0 3%) — TIMEOUT, 점수 미확정
**Trial**: trial_061 kernel v48 (ralph-x iteration 11)

## 결과
- Public: **PENDING** (kernel v48 push 완료, 채점 대기)
- Private: N/A

## 변경사항
- **ConvNeXt 컴포넌트 fold 확장만**: single fold0 → fold0..4 5-fold 평균. BLEND_CONVNEXT=0.03 동일, 다른 컴포넌트 weights·fusion 식·prior_mask 전부 trial_060 그대로.
  - Cell 65 변경: `_ck_path = .../best_fold0.pth` 한 줄 → `sorted(glob(.../best_fold*.pth))`. 단일 `_convnext_model` → 리스트 `_convnext_models`. 5번 루프로 per-fold `ConvNextForImageClassification.from_pretrained → _BirdConvNeXt → load_state_dict(strict=True)` (각 fold마다 fresh from_pretrained — backbone reference 공유 방지). 추론 `convnext_logits[...] = _convnext_model(_spec).numpy()` → `_cnx_preds = np.stack([m(_spec).numpy() for m in _convnext_models]); convnext_logits[...] = _cnx_preds.mean(axis=0)` (pmix 패턴 그대로).
  - Cell 66 변경: trial-label만 trial_060 → trial_061, blend 수식 byte-identical. print에서 `convnext` → `convnext5fold` 라벨링.
- kernel-metadata.json 무변경 (id=ramkang/birdclef2026-effnet-5fold-blend, ramkang/birdclef2026-convnext-5fold + denden12/birdset-convnext-base-xcl 이미 dataset_sources 포함).

## 가설
"ConvNeXt 컴포넌트의 single-fold noise를 5-fold avg로 제거하면 trial_060(TIMEOUT) 미확정 신호가 LB에 양수로 잡힌다."

근거:
- **pmix fold-expansion analog**: trial_051 (2-fold) → 0.933 neutral, trial_053 (4-fold) → **0.934 NEW BEST** (+0.001). 같은 weight에서 fold 수 증가가 single-fold noise를 평균으로 깎아 LB에 잡히는 패턴. ConvNeXt도 동일 메커니즘 기대.
- **ConvNeXt 자체 잠재력**: trial_060에서 fold0 추론은 정상이었고 submission.csv mean이 trial_059 대비 -0.0020(`0.0432 vs 0.0452`)로 logit 분포가 약간 다른 방향 — 즉 ortho 신호가 logit-space에 존재. trial_060 LB 점수가 없어 +방향인지 확인 불가지만, 5-fold avg로 ortho 신호를 안정화시키면 측정 가능.
- **timeout 복구도 동시 달성**: trial_060은 kernel v47이 정상 완료됐는데도 Kaggle 신규 submission이 90분 폴링 중 등장 안 함. 추정 원인은 일회성 API 400이었으므로 trial_061 새 push(v48)에서는 정상 트리거 기대. trial_061이 채점되면 trial_060 정보도 같이 회수되는 셈.

부정 시나리오:
- **distill 별도 컴포넌트의 zero-effect 재현**: trial_059 distill_5fold 별도 3% weight는 0 효과였음. distill도 5-fold avg였고, ConvNeXt 5-fold도 같은 fate일 가능성. 결과적으로 backbone-family 다양화가 0.934 천장에서 무력하다면 iter_4+는 다른 축(Perch multi-window TTA, Perch share 65~70% 축소 + EffNet share 확장, AudioMAE/BEATs probe) 필요.
- **5-fold avg의 양면성**: averaging이 noise뿐 아니라 강한 신호도 깎을 수 있음 — 특정 fold에 유효한 ortho 신호가 5-fold 평균에서 묽어질 가능성도 있음 (HGNetV2 trial_034 -0.003 학습효과: OOF 낮은 weak ensemble은 noise만 추가).

## 검증
- 노트북 syntax: cell 65 (4384 chars), cell 66 (1226 chars) 둘 다 `ast.parse` OK.
- Kaggle dataset `ramkang/birdclef2026-convnext-5fold`: best_fold0..4.pth (각 351MB) + config.json + train.log 전부 확보 (`kaggle datasets files` 확인).
- 메모리 budget: 5×350MB ckpts + 5× from_pretrained = ~2GB peak — Kaggle CPU 환경에서 안전.
- Wall budget: trial_060 = 445s (fold0 inference 1.8min 포함). 5-fold avg는 chunk forward 5x → 추가 ~7-8min 예상, 총 wall ≈ 12-13min, 9h 한도 안에서 충분.

## 다음 가설 (ralph it.12+, 우선순위)
1. **결과 +(0.001 이상)**: ConvNeXt 5-fold avg가 효과적 ortho 컴포넌트라는 확정. → it.12: BLEND_CONVNEXT 0.03 → 0.05~0.07 ramp (Perch 72% → 70%/68%로 살짝 줄여서 ConvNeXt 예산 확장). 또는 distill 3% 복구 + ConvNeXt 5% 동시 운영(Perch 67%/EffNet 28%/ConvNeXt 5%) — 3-model 가족 분리.
2. **결과 neutral (0.934 동률)**: ConvNeXt 자체는 5-fold도 LB-detectable signal 없음. 백본 가족 다양화 축 closure. → it.12: 다른 축으로 pivot:
   - **Perch multi-window TTA** (sub_51 rec #3): 5초 ±shift 2~3회 평균. Perch 72%의 자체 정확도 강화 — single-window noise 제거.
   - **Perch share 축소 + EffNet 확장**: Perch 72% → 65%, EffNet 28% → 35% (mwf0 0.02→0.05, pmix 0.08→0.12 등). trial_036 (Perch ↓ + BLEND 0.25) 학습효과 회피 위해 점진적.
   - **AudioMAE/BEATs frozen probe** (sub_51 rec #1): 학습 비용 필요하나 진짜 다른 임베딩 가족.
3. **결과 −(0.933 또는 그 이하)**: ConvNeXt 5-fold가 noise만 추가 → BLEND_CONVNEXT 0 복귀 (trial_058 0.934 구성 회귀). 백본 다양화 closure 강한 evidence.
4. **결과 TIMEOUT (trial_060 재현)**: Kaggle 플랫폼 이슈 가능성 → kaggle 채점 시스템 자체 점검. 같은 kernel을 v49로 dummy push해서 자동 트리거 작동 여부 확인 필요.
