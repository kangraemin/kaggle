"""Trial 001: March Mania Baseline (최적화 버전)
- 피처: 시드 차이, 정규시즌 승률, 평균 득실점 차이
- 모델: LightGBM
- val: 2021~2024
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path

DATA = Path(__file__).parent.parent.parent.parent / 'data'


def get_seed_num(seed_str):
    return int(seed_str[1:3])


def build_all_team_stats(regular):
    """전체 시즌 팀 통계를 한번에 계산"""
    wins = regular.groupby(['Season', 'WTeamID']).agg(
        w_games=('WScore', 'count'),
        w_pts=('WScore', 'sum'),
        w_opp_pts=('LScore', 'sum')
    ).reset_index().rename(columns={'WTeamID': 'TeamID'})

    losses = regular.groupby(['Season', 'LTeamID']).agg(
        l_games=('LScore', 'count'),
        l_pts=('LScore', 'sum'),
        l_opp_pts=('WScore', 'sum')
    ).reset_index().rename(columns={'LTeamID': 'TeamID'})

    stats = wins.merge(losses, on=['Season', 'TeamID'], how='outer').fillna(0)
    stats['games'] = stats['w_games'] + stats['l_games']
    stats['win_rate'] = stats['w_games'] / stats['games']
    stats['avg_pts'] = (stats['w_pts'] + stats['l_pts']) / stats['games']
    stats['avg_opp_pts'] = (stats['w_opp_pts'] + stats['l_opp_pts']) / stats['games']
    stats['margin'] = stats['avg_pts'] - stats['avg_opp_pts']
    return stats[['Season', 'TeamID', 'win_rate', 'avg_pts', 'avg_opp_pts', 'margin']]


def main():
    tourney = pd.read_csv(DATA / 'MNCAATourneyCompactResults.csv')
    regular = pd.read_csv(DATA / 'MRegularSeasonCompactResults.csv')
    seeds = pd.read_csv(DATA / 'MNCAATourneySeeds.csv')
    sample = pd.read_csv(DATA / 'SampleSubmissionStage1.csv')

    print("Building team stats...")
    team_stats = build_all_team_stats(regular)
    seeds['seed_num'] = seeds.Seed.apply(get_seed_num)

    # 학습 데이터
    print("Building training data...")
    t = tourney[tourney.Season >= 2003].copy()
    t['T1'] = t[['WTeamID', 'LTeamID']].min(axis=1)
    t['T2'] = t[['WTeamID', 'LTeamID']].max(axis=1)
    t['label'] = (t['WTeamID'] == t['T1']).astype(int)

    # merge features
    for prefix, col in [('t1_', 'T1'), ('t2_', 'T2')]:
        t = t.merge(team_stats.rename(columns={c: f'{prefix}{c}' for c in ['win_rate', 'avg_pts', 'avg_opp_pts', 'margin']}
                                       | {'TeamID': col}),
                     on=['Season', col], how='left')
        t = t.merge(seeds[['Season', 'TeamID', 'seed_num']].rename(columns={'TeamID': col, 'seed_num': f'{prefix}seed'}),
                     on=['Season', col], how='left')

    t['seed_diff'] = t['t1_seed'] - t['t2_seed']
    t['win_rate_diff'] = t['t1_win_rate'] - t['t2_win_rate']
    t['margin_diff'] = t['t1_margin'] - t['t2_margin']

    feat_cols = ['seed_diff', 't1_seed', 't2_seed', 'win_rate_diff',
                 't1_win_rate', 't2_win_rate', 'margin_diff', 't1_margin', 't2_margin']

    val_mask = t.Season >= 2021
    X_tr, y_tr = t[~val_mask][feat_cols], t[~val_mask]['label']
    X_val, y_val = t[val_mask][feat_cols], t[val_mask]['label']

    print(f"Train: {len(X_tr)}, Val: {len(X_val)}")
    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                num_leaves=31, random_state=42, verbose=-1)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])

    val_pred = model.predict_proba(X_val)[:, 1]
    brier = np.mean((val_pred - y_val.values) ** 2)
    print(f"\nVal Brier: {brier:.6f} (baseline 0.25)")

    # test 예측
    print("Predicting test...")
    test = sample.copy()
    test[['Season', 'T1', 'T2']] = test.ID.str.split('_', expand=True).astype(int)

    for prefix, col in [('t1_', 'T1'), ('t2_', 'T2')]:
        test = test.merge(team_stats.rename(columns={c: f'{prefix}{c}' for c in ['win_rate', 'avg_pts', 'avg_opp_pts', 'margin']}
                                             | {'TeamID': col}),
                           on=['Season', col], how='left')
        test = test.merge(seeds[['Season', 'TeamID', 'seed_num']].rename(columns={'TeamID': col, 'seed_num': f'{prefix}seed'}),
                           on=['Season', col], how='left')

    test['seed_diff'] = test['t1_seed'] - test['t2_seed']
    test['win_rate_diff'] = test['t1_win_rate'] - test['t2_win_rate']
    test['margin_diff'] = test['t1_margin'] - test['t2_margin']

    test['Pred'] = model.predict_proba(test[feat_cols].fillna(0))[:, 1]
    test[['ID', 'Pred']].to_csv(Path(__file__).parent / 'submission.csv', index=False)
    print("Saved → submission.csv")


if __name__ == '__main__':
    main()
