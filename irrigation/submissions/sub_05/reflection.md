# Sub 05 Reflection — trial_005_ext_data_balanced_acc

## 결과
- Val: 0.9692 (balanced_accuracy) / Public: 미제출
- 147 features, original data TE + append(w=0.1)

## 평가
- balanced_accuracy로 메트릭 수정 — 올바른 방향
- trial_003(0.9711)보다 낮은 0.9692
- original data를 weight=0.1로 append한 건 보수적 — 003은 1.0으로 full blend해서 더 나음
- Deotte formula features 추가했지만 큰 효과 없음
- fold std 안정적 (0.9680~0.9699)
