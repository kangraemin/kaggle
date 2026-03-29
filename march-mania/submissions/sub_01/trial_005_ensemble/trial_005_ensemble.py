"""Trial 005: XGBoost 앙상블 + 최근 폼 + 토너먼트 경험
- LightGBM + XGBoost 앙상블
- 최근 10경기 승률/마진
- 과거 토너먼트 진출 횟수
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
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
    return s[['Season', 'TeamID', 'win_rate', 'margin']]


def build_recent_form(regular, n=10):
    """최근 N경기 승률과 마진"""
    rows = []
    for season in regular.Season.unique():
        sdf = regular[regular.Season == season].sort_values('DayNum')
        team_games = {}
        for _, r in sdf.iterrows():
            for team, won, pts, opp in [(r.WTeamID, 1, r.WScore, r.LScore),
                                         (r.LTeamID, 0, r.LScore, r.WScore)]:
                if team not in team_games:
                    team_games[team] = []
                team_games[team].append((won, pts - opp))
        for team, games in team_games.items():
            last_n = games[-n:]
            rows.append({
                'Season': season, 'TeamID': team,
                'recent_win_rate': np.mean([g[0] for g in last_n]),
                'recent_margin': np.mean([g[1] for g in last_n]),
            })
    return pd.DataFrame(rows)


def build_tourney_exp(tourney):
    """과거 토너먼트 진출 횟수"""
    teams = pd.concat([tourney[['Season', 'WTeamID']].rename(columns={'WTeamID': 'TeamID'}),
                       tourney[['Season', 'LTeamID']].rename(columns={'LTeamID': 'TeamID'})])
    exp = teams.groupby('TeamID')['Season'].apply(lambda x: sorted(x.unique())).reset_index()
    rows = []
    for _, r in exp.iterrows():
        for i, season in enumerate(r.Season):
            past = [s for s in r.Season if s < season]
            rows.append({'Season': season, 'TeamID': r.TeamID,
                        'tourney_apps': len(past),
                        'recent_tourney': 1 if season - 1 in past else 0})
    return pd.DataFrame(rows)


def build_massey(massey):
    last_day = massey.groupby(['Season', 'TeamID'])['RankingDayNum'].max().reset_index()
    merged = massey.merge(last_day, on=['Season', 'TeamID', 'RankingDayNum'])
    return merged.groupby(['Season', 'TeamID'])['OrdinalRank'].agg(
        avg_rank='mean', best_rank='min'
    ).reset_index()


def add_features(df, stats_dict, elo_dict):
    for prefix, col in [('t1_', 'T1'), ('t2_', 'T2')]:
        for name, sdf in stats_dict.items():
            cols_to_rename = [c for c in sdf.columns if c not in ['Season', 'TeamID']]
            rn = {c: f'{prefix}{c}' for c in cols_to_rename} | {'TeamID': col}
            df = df.merge(sdf.rename(columns=rn), on=['Season', col], how='left')
        df[f'{prefix}elo'] = df.apply(lambda r: elo_dict.get((r.Season, r[col]), 1500), axis=1)

    # 모든 diff features 자동 생성
    t1_cols = [c for c in df.columns if c.startswith('t1_') and c != 'T1']
    for c in t1_cols:
        c2 = c.replace('t1_', 't2_')
        if c2 in df.columns:
            df[c.replace('t1_', '') + '_diff'] = df[c] - df[c2]
    return df


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

    all_preds = []
    for prefix, tourney, regular, detailed, seeds, massey, elo in [
        ('M', m_tourney, m_regular, m_detailed, m_seeds, m_massey, m_elo),
        ('W', w_tourney, w_regular, w_detailed, w_seeds, None, w_elo),
    ]:
        print(f"\n=== {prefix} ===")
        print("Building features...")
        team_stats = build_team_stats(regular)
        det_stats = build_detailed_stats(detailed)
        recent = build_recent_form(regular)
        tourney_exp = build_tourney_exp(tourney)
        massey_stats = build_massey(massey) if massey is not None else None
        seeds_df = seeds.copy()
        seeds_df['seed_num'] = seeds_df.Seed.apply(get_seed_num)

        stats_dict = {'basic': team_stats, 'detailed': det_stats, 'recent': recent, 'exp': tourney_exp}
        if massey_stats is not None:
            stats_dict['massey'] = massey_stats
        stats_dict['seed'] = seeds_df[['Season', 'TeamID', 'seed_num']]

        # Train data
        t = tourney[tourney.Season >= 2003].copy()
        t['T1'] = t[['WTeamID', 'LTeamID']].min(axis=1)
        t['T2'] = t[['WTeamID', 'LTeamID']].max(axis=1)
        t['label'] = (t.WTeamID == t.T1).astype(int)
        t = add_features(t, stats_dict, elo)

        feat_cols = [c for c in t.columns if c.endswith('_diff') or c in ['t1_seed_num', 't2_seed_num', 't1_elo', 't2_elo']]
        feat_cols = [c for c in feat_cols if c in t.columns]
        print(f"Features: {len(feat_cols)}")

        # 시즌별 CV
        briers = []
        for val_year in range(2018, 2026):
            tr = t[t.Season < val_year]
            va = t[t.Season == val_year]
            if len(va) == 0: continue

            X_tr, X_va = tr[feat_cols].fillna(0), va[feat_cols].fillna(0)

            m_lgb = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03,
                                        num_leaves=15, random_state=42, verbose=-1)
            m_lgb.fit(X_tr, tr.label)
            p_lgb = m_lgb.predict_proba(X_va)[:, 1]

            m_xgb = xgb.XGBClassifier(n_estimators=300, learning_rate=0.03,
                                       max_depth=4, random_state=42, verbosity=0)
            m_xgb.fit(X_tr, tr.label)
            p_xgb = m_xgb.predict_proba(X_va)[:, 1]

            p_ens = 0.5 * p_lgb + 0.5 * p_xgb
            briers.append(np.mean((p_ens - va.label.values) ** 2))

        print(f"Ensemble CV Brier: {np.mean(briers):.6f}")

        # Full model
        X_all = t[feat_cols].fillna(0)
        m_lgb = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03,
                                    num_leaves=15, random_state=42, verbose=-1)
        m_lgb.fit(X_all, t.label)

        m_xgb = xgb.XGBClassifier(n_estimators=300, learning_rate=0.03,
                                   max_depth=4, random_state=42, verbosity=0)
        m_xgb.fit(X_all, t.label)

        # Test
        sub = sample[sample.ID.str.split('_').str[1].astype(int) >= (3000 if prefix == 'W' else 0)]
        if prefix == 'M':
            sub = sub[sub.ID.str.split('_').str[1].astype(int) < 3000]
        test = sub.copy()
        test[['Season', 'T1', 'T2']] = test.ID.str.split('_', expand=True).astype(int)
        test = add_features(test, stats_dict, elo)

        p_lgb = m_lgb.predict_proba(test[feat_cols].fillna(0))[:, 1]
        p_xgb = m_xgb.predict_proba(test[feat_cols].fillna(0))[:, 1]
        test['Pred'] = (0.5 * p_lgb + 0.5 * p_xgb).clip(0.01, 0.99)
        all_preds.append(test[['ID', 'Pred']])

    result = pd.concat(all_preds, ignore_index=True)
    result = sample[['ID']].merge(result, on='ID', how='left')
    result['Pred'] = result.Pred.fillna(0.5)
    result.to_csv(Path(__file__).parent / 'submission.csv', index=False)
    print(f"\nSaved → submission.csv ({len(result)} rows)")


if __name__ == '__main__':
    main()
