# Submissions Log — ts-forecasting

## 규칙
- sub_XX = Kaggle 제출 1회 단위
- 제출 후 반드시 `sub_XX/reflection.md` 작성 → ✅ reflected 표시 후 다음 sub 진행
- 새 trial은 항상 새 sub 폴더에 (제출 완료된 sub에 추가 금지)

## 제출 전 필수 체크 (0점 방지)
**반드시 실행 후 결과 확인하고 제출할 것. 확인 없이 제출 금지.**

1. 예측값 분포 확인 (abs > 10 몇 건인지)
2. 고가중치 코드 top 10 예측값 확인
3. danger_ratio 계산 → 0.1 넘으면 해당 코드 train mean으로 패치
4. 위 3개 다 통과해야 제출

```bash
python3 -c "
import sys; sys.path.insert(0, 'ts-forecasting')
from utils import load_train, load_test
import pandas as pd, numpy as np

train = load_train()
test = load_test()
preds = pd.read_csv('예측파일.csv')
test_p = test[['id','code']].merge(preds, on='id')

print('=== 1. 예측값 분포 ===')
print(preds.prediction.describe())
print(f'abs>10: {(preds.prediction.abs()>10).sum()}건')

denom = (train.weight * train.y_target**2).sum()
code_max_w = train.groupby('code').weight.max().sort_values(ascending=False)

print('=== 2&3. 고가중치 코드 danger_ratio ===')
for code in code_max_w.head(10).index:
    max_pred = test_p[test_p.code==code].prediction.abs().max()
    ratio = code_max_w[code] * max_pred**2 / denom
    flag = '⚠️ 패치필요' if ratio > 0.1 else '✅'
    print(f'{code}: max_pred={max_pred:.4f}, ratio={ratio:.4f} {flag}')
"
```

| # | Date | Best Trial | Val | Public | Private | Gap | Status |
|---|------|------------|-----|--------|---------|-----|--------|
| 01 | 2026-03-26 | trial_001 | 0.1163 | 0.1499 | - | +0.034 | ✅ reflected |
| 02 | 2026-03-27 | trial_010 | 0.8923 | 0.0000 | - | -0.892 | ✅ reflected |
| 03 | 2026-03-28 | trial_021 | 0.6420 (cold-start) | TBD | - | TBD | ⏳ pending (daily limit) |
