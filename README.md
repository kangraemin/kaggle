# kaggle

Kaggle 대회 실험 및 제출 관리 레포.
대회별 폴더로 구성되며, 모든 대회는 `TRIAL_GUIDE.md`의 규칙을 따른다.

## 대회 목록

| 폴더 | 대회 | 상태 | Best Public |
|------|------|------|-------------|
| `ts-forecasting/` | Hedge fund - Time series forecasting | 진행중 | 0.1499 (sub_01) |
| `churn/` | Playground S6E3 - Binary Classification | 진행중 | TBD |

## 관리 방식

- 실험(Trial)과 제출(Submission)을 분리해서 관리
- 각 제출 단위(`sub_NN/`) 안에 해당 시도한 trial들이 포함
- Public score 수령 후 reflection 작성 + library 기록 의무
- 자세한 규칙: [`TRIAL_GUIDE.md`](./TRIAL_GUIDE.md)

## 제출 전 필수 체크 (ts-forecasting)

```python
# utils.py의 save_submission이 자동으로 실행
# danger_ratio > 0.1인 코드는 train mean으로 자동 패치
```

- `validate_and_patch()` 자동 실행 → 0점 방지
- 제출 후 SUBMISSIONS.md 업데이트 의무

## ts-forecasting 핵심 인사이트

- test 시리즈 89% 신규 → series stats NaN
- AR(1)=0.86, 하지만 test는 lag 없음
- **weight^0.05** 변환이 핵심: 균형있는 학습 → val 0.59+
- known series (10.9%): last_y + series_mean이 강력한 feature
- mixed val (ts_index split + cold-start 15%) 이 최적
