"""Trial 004: Point Spread 예측 → 확률 변환
- 승패(0/1) 대신 점수차(spread) 회귀
- 점수차 → 승리 확률로 변환 (시그모이드 + 캘리브레이션)
- raddar 방식 참고
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.optimize import minimize_scalar
from pathlib import Path

DATA = Path(__file__).parent.parent.parent.parent / 'data'


def get_seed_num(s):
    return int(s[1:3])


def compute_elo(games, k=32, home_adv=100):
    elo = {}
    season_elo = {}
    for _, row in games.sort_values(['Season', 'DayNum']).iterrows():
        w, l = row.WTeamID, row.LTeamID
        if w not in elo: elo[w] = 1500
        if l not in elo: elo[l] = 1500
        adv = home_adv if row.WLoc == 'H' else (-home_adv if row.WLoc == 'A' else 0)
        ew = 1 / (1 + 10 ** ((elo[l] - elo[w] - adv) / 400))
        elo[w] += k * (1 - ew)
        elo[l] -= k * (1 - ew)
        season_elo[(row.Season, w)] = elo[w]
        season_elo[(row.Season, l)] = elo[l]
    return season_elo


def build_detailed_stats(detailed):
    rows = []
    for _, r in detailed.iterrows():
        for team, pf, op in [(r.WTeamID, 'W', 'L'), (r.LTeamID, 'L', 'W')]:
            rows.append({
                'Season': r.Season, 'TeamID': team,
                'FGM': r[f'{pf}FGM'], 'FGA': r[f'{pf}FGA'],
                'FGM3': r[f'{pf}FGM3'], 'FGA3': r[f'{pf}FGA3'],
                'FTM': r[f'{pf}FTM'], 'FTA': r[f'{pf}FTA'],
                'OR': r[f'{pf}OR'], 'DR': r[f'{pf}DR'],
                'Ast': r[f'{pf}Ast'], 'TO': r[f'{pf}TO'],
                'Stl': r[f'{pf}Stl'], 'Blk': r[f'{pf}Blk'],
                'Opp_FGM': r[f'{op}FGM'], 'Opp_FGA': r[f'{op}FGA'],
                'Score': r[f'{pf}Score'], 'Opp_Score': r[f'{op}Score'],
            })
    df = pd.DataFrame(rows)
    agg = df.groupby(['Season', 'TeamID']).mean().reset_index()
    agg['FG_pct'] = agg.FGM / agg.FGA.clip(lower=1)
    agg['FG3_pct'] = agg.FGM3 / agg.FGA3.clip(lower=1)
    agg['FT_pct'] = agg.FTM / agg.FTA.clip(lower=1)
    agg['Opp_FG_pct'] = agg.Opp_FGM / agg.Opp_FGA.clip(lower=1)
    agg['Reb'] = agg.OR + agg.DR
    agg['TO_ratio'] = agg.TO / (agg.FGA + 0.44 * agg.FTA + agg.TO).clip(lower=1)
    agg['avg_margin'] = agg.Score - agg.Opp_Score
    return agg


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
    return merged.groupby(['Season', 'TeamID'])['OrdinalRank'].agg(
        avg_rank='mean', best_rank='min'
    ).reset_index()


def add_features(df, team_stats, detailed, massey, seeds, elo_dict):
    for prefix, col in [('t1_', 'T1'), ('t2_', 'T2')]:
        rn = lambda cols: {c: f'{prefix}{c}' for c in cols} | {'TeamID': col}
        df = df.merge(team_stats.rename(columns=rn(['win_rate', 'margin'])),
                      on=['Season', col], how='left')
        det_cols = ['FG_pct', 'FG3_pct', 'FT_pct', 'Opp_FG_pct', 'Reb', 'Ast', 'TO',
                    'Stl', 'Blk', 'TO_ratio', 'avg_margin']
        df = df.merge(detailed[['Season', 'TeamID'] + det_cols].rename(columns=rn(det_cols)),
                      on=['Season', col], how='left')
        if massey is not None:
            df = df.merge(massey.rename(columns=rn(['avg_rank', 'best_rank'])),
                          on=['Season', col], how='left')
        df = df.merge(seeds[['Season', 'TeamID', 'seed_num']].rename(
            columns={'TeamID': col, 'seed_num': f'{prefix}seed'}),
            on=['Season', col], how='left')
        df[f'{prefix}elo'] = df.apply(lambda r: elo_dict.get((r.Season, r[col]), 1500), axis=1)

    for f in ['seed', 'win_rate', 'margin', 'FG_pct', 'FG3_pct', 'FT_pct',
              'Opp_FG_pct', 'Reb', 'Ast', 'TO', 'Stl', 'Blk', 'TO_ratio',
              'avg_margin', 'elo']:
        if f't1_{f}' in df.columns:
            df[f'{f}_diff'] = df[f't1_{f}'] - df[f't2_{f}']
    if massey is not None and 't1_avg_rank' in df.columns:
        df['rank_diff'] = df.t1_avg_rank - df.t2_avg_rank
    return df


def spread_to_prob(spread, scale):
    """점수차 → 승리 확률 (시그모이드)"""
    return 1 / (1 + np.exp(-spread / scale))


def find_best_scale(spreads, labels):
    """최적 scale 파라미터 찾기"""
    def brier(scale):
        probs = spread_to_prob(spreads, scale)
        return np.mean((probs - labels) ** 2)
    result = minimize_scalar(brier, bounds=(1, 30), method='bounded')
    return result.x


FEAT_COLS = [
    'seed_diff', 't1_seed', 't2_seed',
    'win_rate_diff', 'margin_diff', 'avg_margin_diff',
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

    # 토너먼트 데이터 + 점수차 label
    t = tourney[tourney.Season >= 2003].copy()
    t['T1'] = t[['WTeamID', 'LTeamID']].min(axis=1)
    t['T2'] = t[['WTeamID', 'LTeamID']].max(axis=1)
    t['label'] = (t.WTeamID == t.T1).astype(int)
    # 점수차: T1 관점 (양수 = T1 승리)
    t['spread'] = np.where(t.WTeamID == t.T1, t.WScore - t.LScore, t.LScore - t.WScore)
    t = add_features(t, team_stats, det_stats, massey_stats, seeds_df, elo_dict)

    feat_cols = [c for c in FEAT_COLS if c in t.columns]

    # 시즌별 CV
    briers_cls, briers_reg = [], []
    for val_year in range(2018, 2026):
        tr = t[t.Season < val_year]
        va = t[t.Season == val_year]
        if len(va) == 0: continue

        # 분류 모델
        m_cls = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03,
                                    num_leaves=15, random_state=42, verbose=-1)
        m_cls.fit(tr[feat_cols].fillna(0), tr.label)
        p_cls = m_cls.predict_proba(va[feat_cols].fillna(0))[:, 1]
        briers_cls.append(np.mean((p_cls - va.label.values) ** 2))

        # 회귀 모델 (점수차 예측)
        m_reg = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03,
                                   num_leaves=15, random_state=42, verbose=-1)
        m_reg.fit(tr[feat_cols].fillna(0), tr.spread)
        pred_spread = m_reg.predict(va[feat_cols].fillna(0))
        scale = find_best_scale(m_reg.predict(tr[feat_cols].fillna(0)), tr.label.values)
        p_reg = spread_to_prob(pred_spread, scale)
        briers_reg.append(np.mean((p_reg - va.label.values) ** 2))

    print(f"  Classification CV Brier: {np.mean(briers_cls):.6f}")
    print(f"  Spread→Prob CV Brier:    {np.mean(briers_reg):.6f}")

    # 둘 다 full train
    m_cls = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03,
                                num_leaves=15, random_state=42, verbose=-1)
    m_cls.fit(t[feat_cols].fillna(0), t.label)

    m_reg = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03,
                               num_leaves=15, random_state=42, verbose=-1)
    m_reg.fit(t[feat_cols].fillna(0), t.spread)
    scale = find_best_scale(m_reg.predict(t[feat_cols].fillna(0)), t.label.values)
    print(f"  Optimal scale: {scale:.2f}")

    # Test
    test = sample.copy()
    test[['Season', 'T1', 'T2']] = test.ID.str.split('_', expand=True).astype(int)
    test = add_features(test, team_stats, det_stats, massey_stats, seeds_df, elo_dict)

    p_cls = m_cls.predict_proba(test[feat_cols].fillna(0))[:, 1]
    p_reg = spread_to_prob(m_reg.predict(test[feat_cols].fillna(0)), scale)

    # 앙상블 (0.5 * cls + 0.5 * reg)
    test['Pred'] = 0.5 * p_cls + 0.5 * p_reg
    return test[['ID', 'Pred']]


def main():
    m_tourney = pd.read_csv(DATA / 'MNCAATourneyCompactResults.csv')
    m_regular = pd.read_csv(DATA / 'MRegularSeasonCompactResults.csv')
    m_detailed = pd.read_csv(DATA / 'MRegularSeasonDetailedResults.csv')
    m_seeds = pd.read_csv(DATA / 'MNCAATourneySeeds.csv')
    m_massey = pd.read_csv(DATA / 'MMasseyOrdinals.csv')
    sample = pd.read_csv(DATA / 'SampleSubmissionStage1.csv')

    w_tourney = pd.read_csv(DATA / 'WNCAATourneyCompactResults.csv')
    w_regular = pd.read_csv(DATA / 'WRegularSeasonCompactResults.csv')
    w_detailed = pd.read_csv(DATA / 'WRegularSeasonDetailedResults.csv')
    w_seeds = pd.read_csv(DATA / 'WNCAATourneySeeds.csv')

    print("Computing Elo...")
    all_m = pd.concat([m_regular, m_tourney[m_regular.columns]], ignore_index=True)
    all_w = pd.concat([w_regular, w_tourney[w_regular.columns]], ignore_index=True)
    m_elo = compute_elo(all_m)
    w_elo = compute_elo(all_w)

    m_sample = sample[sample.ID.str.split('_').str[1].astype(int) < 3000]
    w_sample = sample[sample.ID.str.split('_').str[1].astype(int) >= 3000]

    m_preds = process_gender('M', m_tourney, m_regular, m_detailed, m_seeds, m_massey, m_sample, m_elo)
    w_preds = process_gender('W', w_tourney, w_regular, w_detailed, w_seeds, None, w_sample, w_elo)

    result = pd.concat([m_preds, w_preds], ignore_index=True)
    result = sample[['ID']].merge(result, on='ID', how='left')
    result['Pred'] = result.Pred.fillna(0.5).clip(0.01, 0.99)

    result.to_csv(Path(__file__).parent / 'submission.csv', index=False)
    print(f"\nSaved → submission.csv ({len(result)} rows)")


if __name__ == '__main__':
    main()
