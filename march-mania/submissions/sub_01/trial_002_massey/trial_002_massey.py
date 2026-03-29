"""Trial 002: Massey Ordinals 추가
- 197개 전문가 랭킹 시스템의 평균/최고/표준편차
- 시드 + 승률 + 랭킹 조합
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path

DATA = Path(__file__).parent.parent.parent.parent / 'data'


def get_seed_num(s):
    return int(s[1:3])


def build_team_stats(regular):
    wins = regular.groupby(['Season', 'WTeamID']).agg(
        w_games=('WScore', 'count'), w_pts=('WScore', 'sum'), w_opp=('LScore', 'sum')
    ).reset_index().rename(columns={'WTeamID': 'TeamID'})
    losses = regular.groupby(['Season', 'LTeamID']).agg(
        l_games=('LScore', 'count'), l_pts=('LScore', 'sum'), l_opp=('WScore', 'sum')
    ).reset_index().rename(columns={'LTeamID': 'TeamID'})
    s = wins.merge(losses, on=['Season', 'TeamID'], how='outer').fillna(0)
    s['games'] = s.w_games + s.l_games
    s['win_rate'] = s.w_games / s.games
    s['margin'] = (s.w_pts + s.l_pts - s.w_opp - s.l_opp) / s.games
    return s[['Season', 'TeamID', 'win_rate', 'margin']]


def build_massey(massey):
    last_day = massey.groupby(['Season', 'TeamID'])['RankingDayNum'].max().reset_index()
    merged = massey.merge(last_day, on=['Season', 'TeamID', 'RankingDayNum'])
    avg = merged.groupby(['Season', 'TeamID'])['OrdinalRank'].agg(
        avg_rank='mean', rank_std='std', best_rank='min'
    ).reset_index()
    avg['rank_std'] = avg['rank_std'].fillna(0)
    return avg


def add_features(df, team_stats, massey, seeds):
    for prefix, col in [('t1_', 'T1'), ('t2_', 'T2')]:
        renames = lambda cols: {c: f'{prefix}{c}' for c in cols} | {'TeamID': col}
        df = df.merge(team_stats.rename(columns=renames(['win_rate', 'margin'])),
                      on=['Season', col], how='left')
        df = df.merge(massey.rename(columns=renames(['avg_rank', 'rank_std', 'best_rank'])),
                      on=['Season', col], how='left')
        df = df.merge(seeds[['Season', 'TeamID', 'seed_num']].rename(
            columns={'TeamID': col, 'seed_num': f'{prefix}seed'}),
            on=['Season', col], how='left')

    df['seed_diff'] = df.t1_seed - df.t2_seed
    df['win_rate_diff'] = df.t1_win_rate - df.t2_win_rate
    df['margin_diff'] = df.t1_margin - df.t2_margin
    df['rank_diff'] = df.t1_avg_rank - df.t2_avg_rank
    df['best_rank_diff'] = df.t1_best_rank - df.t2_best_rank
    return df


FEAT_COLS = [
    'seed_diff', 't1_seed', 't2_seed',
    'win_rate_diff', 't1_win_rate', 't2_win_rate',
    'margin_diff', 't1_margin', 't2_margin',
    'rank_diff', 't1_avg_rank', 't2_avg_rank',
    'best_rank_diff', 't1_best_rank', 't2_best_rank',
    't1_rank_std', 't2_rank_std',
]


def main():
    tourney = pd.read_csv(DATA / 'MNCAATourneyCompactResults.csv')
    regular = pd.read_csv(DATA / 'MRegularSeasonCompactResults.csv')
    seeds = pd.read_csv(DATA / 'MNCAATourneySeeds.csv')
    massey = pd.read_csv(DATA / 'MMasseyOrdinals.csv')
    sample = pd.read_csv(DATA / 'SampleSubmissionStage1.csv')

    print("Building features...")
    team_stats = build_team_stats(regular)
    massey_stats = build_massey(massey)
    seeds['seed_num'] = seeds.Seed.apply(get_seed_num)

    # train
    t = tourney[tourney.Season >= 2003].copy()
    t['T1'] = t[['WTeamID', 'LTeamID']].min(axis=1)
    t['T2'] = t[['WTeamID', 'LTeamID']].max(axis=1)
    t['label'] = (t.WTeamID == t.T1).astype(int)
    t = add_features(t, team_stats, massey_stats, seeds)

    val_mask = t.Season >= 2021
    X_tr, y_tr = t[~val_mask][FEAT_COLS].fillna(0), t[~val_mask]['label']
    X_val, y_val = t[val_mask][FEAT_COLS].fillna(0), t[val_mask]['label']

    print(f"Train: {len(X_tr)}, Val: {len(X_val)}, Features: {len(FEAT_COLS)}")
    model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03,
                                num_leaves=15, random_state=42, verbose=-1)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])

    val_pred = model.predict_proba(X_val)[:, 1]
    brier = np.mean((val_pred - y_val.values) ** 2)
    print(f"\nVal Brier: {brier:.6f} (baseline 0.25)")

    imp = pd.Series(model.feature_importances_, index=FEAT_COLS).sort_values(ascending=False)
    print("\nTop features:")
    print(imp.head(8))

    # test
    print("\nPredicting test...")
    test = sample.copy()
    test[['Season', 'T1', 'T2']] = test.ID.str.split('_', expand=True).astype(int)
    test = add_features(test, team_stats, massey_stats, seeds)
    test['Pred'] = model.predict_proba(test[FEAT_COLS].fillna(0))[:, 1]
    test[['ID', 'Pred']].to_csv(Path(__file__).parent / 'submission.csv', index=False)
    print("Saved → submission.csv")


if __name__ == '__main__':
    main()
