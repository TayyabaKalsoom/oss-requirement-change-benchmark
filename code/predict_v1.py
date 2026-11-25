import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef
d=pd.read_pickle('/home/claude/work/all_records.pkl').reset_index(drop=True)
# order pairs within project
d['k']=d.groupby('project')['version_pair'].transform(lambda s:s.map({v:i for i,v in enumerate(pd.unique(s))}))
d['changed']=d.status.isin(['Modified','Deleted']).astype(int)
H=[]
for i,r in d.iterrows():
    prev=d[(d.project==r.project)&(d.req_id==r.req_id)&(d.k<r.k)]
    pp=d[(d.project==r.project)&(d.k==r.k-1)]
    H.append(dict(prior_pairs=len(prev),prior_changes=int(prev.changed.sum()),
        prev_changed=int(prev.sort_values('k').changed.iloc[-1]) if len(prev) else -1,
        added_prev=int((prev.status=='Added').any()),
        proj_churn_prev=pp.changed.mean() if len(pp) else np.nan))
d=pd.concat([d,pd.DataFrame(H)],axis=1)
e=d[d.status!='Added'].copy()          # only requirements that exist in the old release
e['old_len']=e.old_text.str.len()
P=['Ambiguity','Completeness','Verification','Technological_Needs','Dependency','Specificity','Consistency']
sets={'old length only':['old_len'],'proxies only':P,'history only':['prior_pairs','prior_changes','prev_changed','added_prev','proj_churn_prev'],
 'proxies+length':P+['old_len'],'history+length':['old_len','prior_pairs','prior_changes','prev_changed','added_prev','proj_churn_prev'],
 'all old-release':P+['old_len','prior_pairs','prior_changes','prev_changed','added_prev','proj_churn_prev']}
print('existing reqs',len(e),'changed rate',round(e.changed.mean(),3)); print(e.groupby('project').changed.agg(['size','mean']).round(3))
def run(tr,te,cols,mk):
    X=tr[cols].fillna(-1); m=mk().fit(X,tr.changed); p=m.predict_proba(te[cols].fillna(-1))[:,1]; return p
rf=lambda:RandomForestClassifier(300,class_weight='balanced',min_samples_leaf=3,random_state=0,n_jobs=-1)
def ev(y,p):
    return dict(AUC=roc_auc_score(y,p),F1=f1_score(y,p>=.5),MCC=matthews_corrcoef(y,p>=.5))
rows=[]
# LOPO
for n,c in sets.items():
    ps=np.zeros(len(e))
    for pr in e.project.unique():
        te=e.project==pr; ps[te.values]=run(e[~te],e[te],c,rf)
    rows.append(dict(test='LOPO',features=n,**ev(e.changed,ps)))
# forward in time: train pairs k<last, test last pair of each project (needs history -> skip k=0)
tr=e[e.k<e.groupby('project').k.transform('max')]; te=e[e.k==e.groupby('project').k.transform('max')]
for n,c in sets.items(): rows.append(dict(test='forward(last pair)',features=n,**ev(te.changed,run(tr,te,c,rf))))
# simple rule baselines on forward set
rows.append(dict(test='forward(last pair)',features='RULE changed-last-time',**ev(te.changed,(te.prev_changed==1).astype(float)+1e-6*np.random.rand(len(te)))))
res=pd.DataFrame(rows).round(3); print(res.to_string(index=False))
# permutation test for 'all old-release' forward
rng=np.random.default_rng(0); c=sets['all old-release']; obs=roc_auc_score(te.changed,run(tr,te,c,rf)); null=[]
for _ in range(100):
    t2=tr.copy(); t2['changed']=rng.permutation(t2.changed.values); null.append(roc_auc_score(te.changed,run(t2,te,c,rf)))
print('forward all-features AUC',round(obs,3),'perm p',(np.sum(np.array(null)>=obs)+1)/101, 'null mean',round(np.mean(null),3))
res.to_csv('/home/claude/out/prediction_test_v1.csv',index=False)
