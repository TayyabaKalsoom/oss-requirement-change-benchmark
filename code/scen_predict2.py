import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
e=pd.read_pickle('/home/claude/work/sylius_scen_dataset.pkl')
pc=e.groupby('k').changed.mean(); mass=pc[pc>0.5].index.tolist(); print('mass-restructure pairs dropped:',len(mass))
f=e[~e.k.isin(mass)&(e.status!='Deleted')].copy()      # target: text edited vs not, among surviving scenarios, non-mass releases
f['changed']=(f.status=='Modified').astype(int); print(len(f),'obs; edit rate',round(f.changed.mean(),3))
sets={'old length/steps':['old_len','n_steps','n_ex'],'history only':['prior_obs','prior_changes','since_change','age'],
 'file+project churn':['file_churn_prev','proj_churn_prev'],'all old-release':['old_len','n_steps','n_ex','prior_obs','prior_changes','since_change','age','file_churn_prev','proj_churn_prev']}
rf=lambda:RandomForestClassifier(300,class_weight='balanced',min_samples_leaf=5,random_state=0,n_jobs=-1)
ks=[k for k in sorted(f.k.unique()) if k>=8]; res=[]; rng=np.random.default_rng(0)
for n,c in sets.items():
    A=[]; 
    for k in ks:
        tr=f[f.k<k]; te=f[f.k==k]
        if te.changed.sum()<5 or tr.changed.nunique()<2: continue
        p=rf().fit(tr[c].fillna(-1),tr.changed).predict_proba(te[c].fillna(-1))[:,1]; A.append(roc_auc_score(te.changed,p))
    bs=[np.mean(rng.choice(A,len(A))) for _ in range(2000)]
    res.append(dict(features=n,mean_AUC=np.mean(A),ci_lo=np.percentile(bs,2.5),ci_hi=np.percentile(bs,97.5),pairs=len(A),above_0_5=int(np.sum(np.array(A)>.5))))
r=pd.DataFrame(res).round(3); print(r.to_string(index=False)); r.to_csv('/home/claude/out/scenario_prediction_edit_target.csv',index=False)
