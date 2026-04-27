"""XGB max_depth=1 multi-seed — Chris Deotte: 이 데이터는 depth=1이 최적"""
import pandas as pd, numpy as np, xgboost as xgb, json, warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

N_FOLDS=5; SEEDS=[42,0,1,2,3,4,5]; ALPHA=10
DATA_DIR="/Users/ram/programming/vibecoding/kaggle/churn"
OUT_DIR=f"{DATA_DIR}/submissions/sub_07/trial_058_xgb_depth1"
train=pd.read_csv(f"{DATA_DIR}/train.csv"); test=pd.read_csv(f"{DATA_DIR}/test.csv")
train["target"]=(train["Churn"]=="Yes").astype(int)
for df in [train,test]:
    df["TotalCharges"]=pd.to_numeric(df["TotalCharges"],errors="coerce")
    df["TotalCharges"]=df["TotalCharges"].fillna(df["TotalCharges"].median())
    df["AvgMonthlyCharge"]=df["TotalCharges"]/(df["tenure"]+1)
    df["ChargeGap"]=df["MonthlyCharges"]-df["AvgMonthlyCharge"]
    df["ChargeRatio"]=df["MonthlyCharges"]/(df["TotalCharges"]+1)
cat_cols=["gender","Partner","Dependents","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod"]
num_cols=[c for c in train.columns if c not in ["id","Churn","target"]+cat_cols]
y=train["target"]
params={"objective":"binary:logistic","eval_metric":"auc","learning_rate":0.05,"max_depth":1,"subsample":0.8,"colsample_bytree":0.8,"reg_alpha":1.0,"reg_lambda":1.0,"verbosity":0}
all_oof=np.zeros(len(train)); all_test=np.zeros(len(test))
for seed in SEEDS:
    skf=StratifiedKFold(n_splits=N_FOLDS,shuffle=True,random_state=seed)
    seed_oof=np.zeros(len(train)); seed_test=np.zeros(len(test))
    for fold,(tr_idx,val_idx) in enumerate(skf.split(train[num_cols],y)):
        tr_df,val_df=train.iloc[tr_idx],train.iloc[val_idx]
        gm=tr_df["target"].mean()
        te_tr,te_val,te_te={},{},{}
        for col in cat_cols:
            stats=tr_df.groupby(col)["target"].agg(["sum","count"])
            smooth=(stats["sum"]+ALPHA*gm)/(stats["count"]+ALPHA)
            te_tr[f"te_{col}"]=tr_df[col].map(smooth).fillna(gm).values
            te_val[f"te_{col}"]=val_df[col].map(smooth).fillna(gm).values
            te_te[f"te_{col}"]=test[col].map(smooth).fillna(gm).values
        X_tr=pd.concat([tr_df[num_cols].reset_index(drop=True),pd.DataFrame(te_tr)],axis=1)
        X_val=pd.concat([val_df[num_cols].reset_index(drop=True),pd.DataFrame(te_val)],axis=1)
        X_te=pd.concat([test[num_cols].reset_index(drop=True),pd.DataFrame(te_te)],axis=1)
        p={**params,"random_state":seed}
        model=xgb.train(p,xgb.DMatrix(X_tr,label=y.iloc[tr_idx]),num_boost_round=2000,evals=[(xgb.DMatrix(X_val,label=y.iloc[val_idx]),"val")],early_stopping_rounds=50,verbose_eval=False)
        seed_oof[val_idx]=model.predict(xgb.DMatrix(X_val))
        seed_test+=model.predict(xgb.DMatrix(X_te))/N_FOLDS
    print(f"SEED {seed}: {roc_auc_score(y,seed_oof):.5f}")
    all_oof+=seed_oof/len(SEEDS); all_test+=seed_test/len(SEEDS)
oof_auc=roc_auc_score(y,all_oof)
print(f"\nOOF AUC: {oof_auc:.5f}")
np.save(f"{OUT_DIR}/oof_preds.npy",all_oof); np.save(f"{OUT_DIR}/test_preds.npy",all_test)
pd.DataFrame({"id":test["id"],"Churn":all_test}).to_csv(f"{OUT_DIR}/trial_058_xgb_depth1.csv",index=False)
json.dump({"id":"058","status":"done","val_score":round(oof_auc,5)},open(f"{OUT_DIR}/results.json","w"),indent=2)
print("Done.")
