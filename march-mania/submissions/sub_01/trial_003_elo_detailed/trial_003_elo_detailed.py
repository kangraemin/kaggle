"""Trial 003: Elo + Detailed Stats
- Elo 레이팅 시스템 구현
- 상세 경기 스탯 (FG%, 3PT%, 리바운드, TO 등)
- 남녀 모두 예측
- 시즌별 CV (2018~2025)
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path

DATA = Path(__file__).parent.parent.parent.parent / 'data'


def get_seed_num(s):
    return int(s[1:3])


# === Elo Rating ===
def compute_elo(games, k=32, home_adv=100):
    """시즌별 Elo 레이팅 계산. 매 시즌 초기화 안 함 (연속)."""
    elo = {}
    season_elo = {}  # {(season, team): elo at end of season}

    for _, row in games.sort_values(['Season', 'DayNum']).iterrows():
        w, l = row.WTeamID, row.LTeamID
        if w not in elo: elo[w] = 1500
        if l not in elo: elo[l] = 1500

        # 홈 어드밴티지
        adv = home_adv if row.WLoc == 'H' else (-home_adv if row.WLoc == 'A' else 0)

        ew = 1 / (1 + 10 ** ((elo[l] - elo[w] - adv) / 400))
        elo[w] += k * (1 - ew)
        elo[l] -= k * (1 - ew)

        season_elo[(row.Season, w)] = elo[w]
        season_elo[(row.Season, l)] = elo[l]

    return season_elo


# === Detailed Stats ===
def build_detailed_stats(detailed):
    """시즌별 팀 상세 통계"""
    rows = []
    for _, r in detailed.iterrows():
        for team, prefix, opp_prefix in [(r.WTeamID, 'W', 'L'), (r.LTeamID, 'L', 'W')]:
            rows.append({
                'Season': r.Season, 'TeamID': team,
                'FGM': r[f'{prefix}FGM'], 'FGA': r[f'{prefix}FGA'],
                'FGM3': r[f'{prefix}FGM3'], 'FGA3': r[f'{prefix}FGA3'],
                'FTM': r[f'{prefix}FTM'], 'FTA': r[f'{prefix}FTA'],
                'OR': r[f'{prefix}OR'], 'DR': r[f'{prefix}DR'],
                'Ast': r[f'{prefix}Ast'], 'TO': r[f'{prefix}TO'],
                'Stl': r[f'{prefix}Stl'], 'Blk': r[f'{prefix}Blk'],
                'Opp_FGM': r[f'{opp_prefix}FGM'], 'Opp_FGA': r[f'{opp_prefix}FGA'],
            })
    df = pd.DataFrame(rows)
    agg = df.groupby(['Season', 'TeamID']).mean().reset_index()
    agg['FG_pct'] = agg.FGM / agg.FGA.clip(lower=1)
    agg['FG3_pct'] = agg.FGM3 / agg.FGA3.clip(lower=1)
    agg['FT_pct'] = agg.FTM / agg.FTA.clip(lower=1)
    agg['Opp_FG_pct'] = agg.Opp_FGM / agg.Opp_FGA.clip(lower=1)
    agg['Reb'] = agg.OR + agg.DR
    agg['TO_ratio'] = agg.TO / (agg.FGA + 0.44 * agg.FTA + agg.TO).clip(lower=1)
    return agg[['Season', 'TeamID', 'FG_pct', 'FG3_pct', 'FT_pct', 'Opp_FG_pct',
                 'Reb', 'Ast', 'TO', 'Stl', 'Blk', 'TO_ratio']]


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
    s['avg_pts'] = (s.w_pts + s.l_pts) / s.games
    return s[['Season', 'TeamID', 'win_rate', 'margin', 'avg_pts']]


def build_massey(massey):
    last_day = massey.groupby(['Season', 'TeamID'])['RankingDayNum'].max().reset_index()
    merged = massey.merge(last_day, on=['Season', 'TeamID', 'RankingDayNum'])
    avg = merged.groupby(['Season', 'TeamID'])['OrdinalRank'].agg(
        avg_rank='mean', best_rank='min'
    ).reset_index()
    return avg


def add_features(df, team_stats, detailed, massey, seeds, elo_dict, gender='M'):
    for prefix, col in [('t1_', 'T1'), ('t2_', 'T2')]:
        rn = lambda cols: {c: f'{prefix}{c}' for c in cols} | {'TeamID': col}
        df = df.merge(team_stats.rename(columns=rn(['win_rate', 'margin', 'avg_pts'])),
                      on=['Season', col], how='left')
        df = df.merge(detailed.rename(columns=rn(['FG_pct', 'FG3_pct', 'FT_pct', 'Opp_FG_pct',
                                                    'Reb', 'Ast', 'TO', 'Stl', 'Blk', 'TO_ratio'])),
                      on=['Season', col], how='left')
        if massey is not None:
            df = df.merge(massey.rename(columns=rn(['avg_rank', 'best_rank'])),
                          on=['Season', col], how='left')
        df = df.merge(seeds[['Season', 'TeamID', 'seed_num']].rename(
            columns={'TeamID': col, 'seed_num': f'{prefix}seed'}),
            on=['Season', col], how='left')
        # Elo
        df[f'{prefix}elo'] = df.apply(lambda r: elo_dict.get((r.Season, r[col]), 1500), axis=1)

    # Diff features
    for f in ['seed', 'win_rate', 'margin', 'avg_pts', 'FG_pct', 'FG3_pct', 'FT_pct',
              'Opp_FG_pct', 'Reb', 'Ast', 'TO', 'Stl', 'Blk', 'TO_ratio', 'elo']:
        if f'{prefix}{f}' in ['t2_avg_rank', 't2_best_rank'] and massey is None:
            continue
        if f't1_{f}' in df.columns and f't2_{f}' in df.columns:
            df[f'{f}_diff'] = df[f't1_{f}'] - df[f't2_{f}']
    if massey is not None:
        df['rank_diff'] = df['t1_avg_rank'] - df['t2_avg_rank']
    return df


FEAT_COLS = [
    'seed_diff', 't1_seed', 't2_seed',
    'win_rate_diff', 'margin_diff', 'avg_pts_diff',
    'FG_pct_diff', 'FG3_pct_diff', 'FT_pct_diff', 'Opp_FG_pct_diff',
    'Reb_diff', 'Ast_diff', 'TO_diff', 'Stl_diff', 'Blk_diff', 'TO_ratio_diff',
    'elo_diff', 't1_elo', 't2_elo',
    'rank_diff', 't1_avg_rank', 't2_avg_rank',
]


def process_gender(prefix, tourney, regular, detailed, seeds, massey, sample, elo_dict):
    print(f"\n=== {prefix} ===")
    team_stats = build_team_stats(regular)
    det_stats = build_detailed_stats(detailed)
    seeds_df = seeds.copy()
    seeds_df['seed_num'] = seeds_df.Seed.apply(get_seed_num)
    massey_stats = build_massey(massey) if massey is not None else None

    t = tourney[tourney.Season >= 2003].copy()
    t['T1'] = t[['WTeamID', 'LTeamID']].min(axis=1)
    t['T2'] = t[['WTeamID', 'LTeamID']].max(axis=1)
    t['label'] = (t.WTeamID == t.T1).astype(int)
    t = add_features(t, team_stats, det_stats, massey_stats, seeds_df, elo_dict, prefix)

    feat_cols = [c for c in FEAT_COLS if c in t.columns]

    # 시즌별 CV
    briers = []
    for val_year in range(2018, 2026):
        tr = t[t.Season < val_year]
        va = t[t.Season == val_year]
        if len(va) == 0: continue
        model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03,
                                    num_leaves=15, random_state=42, verbose=-1)
        model.fit(tr[feat_cols].fillna(0), tr.label)
        pred = model.predict_proba(va[feat_cols].fillna(0))[:, 1]
        b = np.mean((pred - va.label.values) ** 2)
        briers.append(b)
    print(f"CV Brier (2018-2025): {np.mean(briers):.6f}")

    # Full model
    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03,
                                num_leaves=15, random_state=42, verbose=-1)
    model.fit(t[feat_cols].fillna(0), t.label)

    imp = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)
    print("Top features:", imp.head(5).index.tolist())

    # Test predictions
    test = sample.copy()
    test[['Season', 'T1', 'T2']] = test.ID.str.split('_', expand=True).astype(int)
    test = add_features(test, team_stats, det_stats, massey_stats, seeds_df, elo_dict, prefix)
    test['Pred'] = model.predict_proba(test[feat_cols].fillna(0))[:, 1]
    return test[['ID', 'Pred']]


def main():
    # Men's
    m_tourney = pd.read_csv(DATA / 'MNCAATourneyCompactResults.csv')
    m_regular = pd.read_csv(DATA / 'MRegularSeasonCompactResults.csv')
    m_detailed = pd.read_csv(DATA / 'MRegularSeasonDetailedResults.csv')
    m_seeds = pd.read_csv(DATA / 'MNCAATourneySeeds.csv')
    m_massey = pd.read_csv(DATA / 'MMasseyOrdinals.csv')
    sample = pd.read_csv(DATA / 'SampleSubmissionStage1.csv')

    # Women's
    w_tourney = pd.read_csv(DATA / 'WNCAATourneyCompactResults.csv')
    w_regular = pd.read_csv(DATA / 'WRegularSeasonCompactResults.csv')
    w_detailed = pd.read_csv(DATA / 'WRegularSeasonDetailedResults.csv')
    w_seeds = pd.read_csv(DATA / 'WNCAATourneySeeds.csv')

    print("Computing Elo ratings...")
    all_m = pd.concat([m_regular, m_tourney[m_regular.columns]], ignore_index=True)
    all_w = pd.concat([w_regular, w_tourney[w_regular.columns]], ignore_index=True)
    m_elo = compute_elo(all_m)
    w_elo = compute_elo(all_w)

    # 남녀 분리 처리
    m_sample = sample[sample.ID.str.split('_').str[1].astype(int) < 3000]
    w_sample = sample[sample.ID.str.split('_').str[1].astype(int) >= 3000]

    m_preds = process_gender('M', m_tourney, m_regular, m_detailed, m_seeds, m_massey, m_sample, m_elo)
    w_preds = process_gender('W', w_tourney, w_regular, w_detailed, w_seeds, None, w_sample, w_elo)

    result = pd.concat([m_preds, w_preds], ignore_index=True)
    # sample 순서 맞추기
    result = sample[['ID']].merge(result, on='ID', how='left')
    result['Pred'] = result.Pred.fillna(0.5)

    result.to_csv(Path(__file__).parent / 'submission.csv', index=False)
    print(f"\nTotal predictions: {len(result)}, NaN: {result.Pred.isna().sum()}")
    print("Saved → submission.csv")


if __name__ == '__main__':
    main()
