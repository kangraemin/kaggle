# sub_25 — Domain EDA Summary

## 클래스 카탈로그
- **taxonomy.csv 클래스: 234** (submission target)
- **train.csv 클래스: 206**
- **labeled soundscape 클래스: 75** (in target: 75)
- **학습 데이터 0인 클래스: 28** ← 현재 모델 영구 0점

## Prior Tier 분포 (235 클래스)
| tier | 의미 | 수 |
|---|---|---|
| A_in_soundscape | labeled soundscape 등장 | 75 |
| B_in_pantanal_train | Pantanal 박스 안 학습 데이터에 등장 | 92 |
| C_train_only | 학습 데이터 있으나 Pantanal·soundscape 미등장 | 67 |
| D_no_train_data | 학습 데이터 0 | 0 |

→ 후처리: A=1.0, B=1.0, C=0.3, D=0.5 같은 prior mask 적용 검토.
   D는 25 Insect sonotype = labeled soundscape에서만 학습 가능.

## train_soundscapes 폴더 메타 (10658 files)
- 사이트: 23개
- 사이트 분포 top: {'S01': 2341, 'S02': 2505, 'S03': 5, 'S04': 17, 'S05': 9}
- **S05 파일: 9개** (test와 같은 사이트)
- S05 시각 분포: {3: 5, 17: 4}
- S05 월 분포: {2: 4, 11: 5}
- S05 연도 분포: {2024: 5, 2025: 4}

## Pantanal 박스 안 학습 녹음
- 전체 train.csv: 35549
- Pantanal 박스 안: 847 (2.4%)
- Pantanal에서 녹음된 종 수: 119

## 시사점
1. **A+B = 167** 종이 test에 진짜 나타날 가능성 큼. 나머지 ~67 종은 false positive 위험.
2. **D 25개 (Insect sonotype)** 는 labeled soundscape (66 files)만이 학습 소스. 무조건 활용 필수.
3. **S05 9개 파일** 은 test와 정확히 같은 사이트. BirdNET pseudo-label 우선순위.
