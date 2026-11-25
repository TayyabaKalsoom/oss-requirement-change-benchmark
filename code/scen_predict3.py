import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
E=pd.concat([pd.read_pickle(f'/home/claude/work/{p}_scen_dataset.pkl') for p in ['sylius','diaspora','gitlab']],ignore_index=True)
pc=E.groupby(['project','k']).changed.mean(); mass=set(pc[pc>0.5].index); E=E[[ (p,k) not in mass for p,k in zip(E.project,E.k)]]
f=E[E.status!='Deleted'].copy(); f['changed']=(f.status=='Modified').astype(int)
print('mass pairs dropped',len(mass)); print(f.groupby('project').changed.agg(['size','mean']).round(3))
sets={'old length/steps':['old_len','n_steps','n_ex'],'history only':['prior_obs','prior_changes','since_change','age'],'file+project churn':['file_churn_prev','proj_churn_prev'],
 'all old-release':['old_len','n_steps','n_ex','prior_obs','prior_changes','since_change','age','file_churn_prev','proj_churn_prev']}
rf=lambda:RandomForestClassifier(200,class_weight='balanced',min_samples_leaf=5,random_state=0,n_jobs=-1)
rng=np.random.default_rng(0); res=[]
for design in ['same-project earlier releases','other projects + same-project earlier']:
  for n,c in sets.items():
    for pr in ['sylius','diaspora','gitlab','ALL']:
        A=[]
        for p2 in (['sylius','diaspora','gitlab'] if pr=='ALL' else [pr]):
            g=f[f.project==p2]; ks=[k for k in sorted(g.k.unique()) if k>=8]
            for k in ks[::2 if p2=='gitlab' else 1]:           # gitlab has many pairs: every 2nd to save time
                te=g[g.k==k]
                if te.changed.sum()<5: continue
                tr=g[g.k<k] if design.startswith('same') else pd.concat([g[g.k<k],f[f.project!=p2]])
                if tr.changed.nunique()<2: continue
                p=rf().fit(tr[c].fillna(-1),tr.changed).predict_proba(te[c].fillna(-1))[:,1]; A.append(roc_auc_score(te.changed,p))
        if len(A)<3: continue
        bs=[np.mean(rng.choice(A,len(A))) for _ in range(1500)]
        res.append(dict(design=design,features=n,project=pr,mean_AUC=np.mean(A),lo=np.percentile(bs,2.5),hi=np.percentile(bs,97.5),pairs=len(A),above_half=int(np.sum(np.array(A)>.5))))
r=pd.DataFrame(res).round(3); pd.set_option('display.width',200); print(r.to_string(index=False)); r.to_csv('/home/claude/out/scenario_prediction_3projects.csv',index=False)
