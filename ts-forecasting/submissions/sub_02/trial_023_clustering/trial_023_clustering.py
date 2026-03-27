"""Trial 023: Series Clustering
- train 시리즈를 raw feature 평균벡터로 K-means 클러스터링
- test 신규 시리즈 → 가장 가까운 클러스터 배정 → 클러스터 통계 사용
- cold-start 문제 직접 해결 시도
- cold-start val (시리즈 10% holdout)
"""
import gc
import sys
import psutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from utils import *

MEM_LIMIT_GB = 2.0
EXCLUDE_FEATURES = {'feature_w', 'feature_x', 'feature_y', 'feature_z'}
N_CLUSTERS = 100
PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 255,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'verbose': -1,
    'n_jobs': -1,
}


def check_memory(label=''):
    avail = psutil.virtual_memory().available / 1024**3
    print(f"[MEM] {label}: {avail:.1f} GB free", flush=True)
    if avail < MEM_LIMIT_GB:
        print(f"[MEM] 위험 — 강제 종료", flush=True)
        sys.exit(1)


def get_raw_feat_cols(df):
    return [c for c in df.columns if c.startswith('feature_') and c not in EXCLUDE_FEATURES]


def build_series_vectors(df):
    """시리즈별 raw feature 평균벡터 계산."""
    feat_cols = get_raw_feat_cols(df)
    series_vec = df.groupby(KEY)[feat_cols].mean()
    return series_vec, feat_cols


def fit_clusters(series_vec, feat_cols, n_clusters=N_CLUSTERS):
    """K-means 클러스터링."""
    X = series_vec[feat_cols].values
    # NaN을 컬럼 평균으로 채움
    col_means = np.nanmean(X, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=3)
    series_vec = series_vec.copy()
    series_vec['cluster'] = kmeans.fit_predict(X_scaled)
    print(f"Cluster sizes: min={series_vec.cluster.value_counts().min()}, "
          f"max={series_vec.cluster.value_counts().max()}")
    return kmeans, scaler, series_vec


def assign_clusters(df, kmeans, scaler, feat_cols):
    """각 row에 클러스터 배정."""
    series_vec, _ = build_series_vectors(df)
    X = series_vec[feat_cols].values
    col_means = np.nanmean(X, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    X_scaled = scaler.transform(X)
    series_vec['cluster'] = kmeans.predict(X_scaled)
    key_df = pd.DataFrame(series_vec.index.tolist(), columns=KEY)
    key_df['cluster'] = series_vec['cluster'].values
    df = df.merge(key_df, on=KEY, how='left')
    return df


def compute_cluster_stats(df):
    """클러스터별 y_target 통계."""
    return df.groupby('cluster')['y_target'].agg(
        cluster_mean='mean', cluster_std='std', cluster_median='median'
    ).reset_index()


def compute_group_stats(df):
    s2 = df.groupby(['sub_category', 'horizon'])['y_target'].agg(
        subcat_mean='mean', subcat_std='std'
    ).reset_index()
    s3 = df.groupby(['horizon'])['y_target'].agg(
        horizon_mean='mean', horizon_std='std'
    ).reset_index()
    return s2, s3


def apply_stats(df, cluster_stats, s2, s3):
    df = df.merge(cluster_stats, on='cluster', how='left')
    df = df.merge(s2, on=['sub_category', 'horizon'], how='left')
    df = df.merge(s3, on=['horizon'], how='left')
    return df


def get_feature_cols(df):
    raw = [c for c in df.columns if c.startswith('feature_') and c not in EXCLUDE_FEATURES]
    enc = [c for c in df.columns if c.endswith('_mean') or c.endswith('_std') or c.endswith('_median')]
    return raw + enc


def prepare_X(df):
    feat_cols = get_feature_cols(df)
    cat_cols = ['code', 'sub_code', 'sub_category']
    X = df[feat_cols + cat_cols + ['horizon', 'ts_index', 'cluster']].copy()
    for c in cat_cols:
        X[c] = X[c].astype('category')
    X['cluster'] = X['cluster'].astype('category')
    return X


def main():
    train = load_train()
    check_memory('after load')

    # cold-start val split (10% holdout)
    all_series = train[KEY].drop_duplicates()
    np.random.seed(42)
    val_idx = np.random.choice(len(all_series), size=int(len(all_series) * 0.1), replace=False)
    val_keys = set(map(tuple, all_series.iloc[val_idx].values))
    is_val = train[KEY].apply(tuple, axis=1).isin(val_keys)
    tr  = train[~is_val].copy()
    val = train[is_val].copy()
    del train, is_val, all_series
    gc.collect()

    print(f"Train series: {tr[KEY].drop_duplicates().shape[0]}, Val series: {val[KEY].drop_duplicates().shape[0]}")
    check_memory('after split')

    # tr 기준 클러스터 학습
    print("Building series vectors...")
    series_vec_tr, feat_cols = build_series_vectors(tr)
    print(f"Fitting {N_CLUSTERS} clusters on {len(series_vec_tr)} train series...")
    kmeans, scaler, series_vec_tr = fit_clusters(series_vec_tr, feat_cols)

    # tr/val에 클러스터 배정
    tr  = assign_clusters(tr,  kmeans, scaler, feat_cols)
    val = assign_clusters(val, kmeans, scaler, feat_cols)
    check_memory('after clustering')

    # 통계 계산 (tr 기준)
    cluster_stats = compute_cluster_stats(tr)
    s2, s3 = compute_group_stats(tr)

    tr  = apply_stats(tr,  cluster_stats, s2, s3)
    val = apply_stats(val, cluster_stats, s2, s3)

    # NaN 체크
    for col in ['cluster_mean', 'subcat_mean']:
        print(f"val {col} NaN: {val[col].isna().mean()*100:.1f}%")
    check_memory('after stats')

    X_tr  = prepare_X(tr)
    X_val = prepare_X(val)
    cat = ['code', 'sub_code', 'sub_category', 'cluster']
    dtrain = lgb.Dataset(X_tr, label=tr.y_target.values, weight=tr.weight.values,
                         categorical_feature=cat, free_raw_data=True)
    dval   = lgb.Dataset(X_val, label=val.y_target.values, weight=val.weight.values,
                         categorical_feature=cat, reference=dtrain, free_raw_data=True)
    model = lgb.train(PARAMS, dtrain, num_boost_round=3000,
                      valid_sets=[dval],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])

    val_pred = model.predict(X_val)
    score = weighted_rmse_score(val.y_target.values, val_pred, val.weight.values)
    print(f"\n[Trial 023] Cold-start val score: {score:.6f}")

    imp = pd.Series(model.feature_importance('gain'), index=X_tr.columns).sort_values(ascending=False)
    print("\nTop 15 feature importance:")
    print(imp.head(15))

    best_iter = model.best_iteration
    del tr, val, X_tr, X_val, model
    gc.collect()
    check_memory('after val, before retrain')

    # full retrain
    train_full = load_train()
    series_vec_full, _ = build_series_vectors(train_full)
    _, _, series_vec_full = fit_clusters(series_vec_full, feat_cols, n_clusters=N_CLUSTERS)
    # full train으로 다시 kmeans 학습
    X_full_vec = series_vec_full[feat_cols].values
    col_means = np.nanmean(X_full_vec, axis=0)
    nan_mask = np.isnan(X_full_vec)
    X_full_vec[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    scaler_full = StandardScaler()
    X_full_scaled = scaler_full.fit_transform(X_full_vec)
    kmeans_full = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=3)
    series_vec_full['cluster'] = kmeans_full.fit_predict(X_full_scaled)

    key_df = pd.DataFrame(series_vec_full.index.tolist(), columns=KEY)
    key_df['cluster'] = series_vec_full['cluster'].values
    train_full = train_full.merge(key_df, on=KEY, how='left')

    cluster_stats_full = compute_cluster_stats(train_full)
    s2_full, s3_full = compute_group_stats(train_full)
    train_full = apply_stats(train_full, cluster_stats_full, s2_full, s3_full)

    y_full = train_full.y_target.values
    w_full = train_full.weight.values
    X_full = prepare_X(train_full)
    del train_full
    gc.collect()

    model_full = retrain_full(X_full, y_full, w_full, best_iter, params=PARAMS)
    del X_full, y_full, w_full
    gc.collect()
    check_memory('after retrain')

    # test 예측
    test = load_test()
    # test 시리즈에 클러스터 배정 (full kmeans 기준)
    series_vec_test, _ = build_series_vectors(test.assign(y_target=0))  # dummy y_target
    X_test_vec = series_vec_test[feat_cols].values
    nan_mask = np.isnan(X_test_vec)
    X_test_vec[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    X_test_scaled = scaler_full.transform(X_test_vec)
    series_vec_test['cluster'] = kmeans_full.predict(X_test_scaled)
    key_df_test = pd.DataFrame(series_vec_test.index.tolist(), columns=KEY)
    key_df_test['cluster'] = series_vec_test['cluster'].values
    test = test.merge(key_df_test, on=KEY, how='left')
    test = apply_stats(test, cluster_stats_full, s2_full, s3_full)

    X_te = prepare_X(test)
    for col in X_te.columns:
        nan_pct = X_te[col].isna().mean()
        if nan_pct > 0.05:
            print(f"WARNING test: {col} NaN {nan_pct*100:.1f}%")

    save_submission(test, model_full.predict(X_te), 'trial_023_clustering',
                    output_dir=Path(__file__).parent)


if __name__ == '__main__':
    main()
