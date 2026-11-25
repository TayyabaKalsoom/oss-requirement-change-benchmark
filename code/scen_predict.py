import pandas as pd, numpy as np, difflib, re, warnings; warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
d=pd.read_pickle('/home/claude/work/sylius_scen_snapshots.pkl')
norm=lambda s:re.sub(r'\s+',' ',s.strip().lower())
d['nt']=d.text.map(norm); K=sorted(d.k.unique())
S={k:g.set_index('key') for k,g in d.groupby('k')}
rows=[]
for i_,k in enumerate(K[:-1]):
    k2=K[i_+1]; a,b=S[k],S[k2]; gone=[x for x in a.index if x not in b.index]; new=[x for x in b.index if x not in a.index]
    ren={}
    byfile={}
    for n in new: byfile.setdefault(n.split('::')[0],[]).append(n)
    used=set()
    for o in gone:
        best=(0,None)
        for n in byfile.get(o.split('::')[0],[]):
            if n in used: continue
            if difflib.SequenceMatcher(None,a.nt[o],b.nt[n],autojunk=False).quick_ratio()<.6: continue
            r=difflib.SequenceMatcher(None,a.nt[o],b.nt[n],autojunk=False).ratio()
            if r>best[0]: best=(r,n)
        if best[0]>=.6: ren[o]=best[1]; used.add(best[1])
    for key in a.index:
        if key in b.index: new_t=b.nt[key]; st='Unchanged' if new_t==a.nt[key] else 'Modified'
        elif key in ren: st='Modified'
        else: st='Deleted'
        rows.append(dict(k=k,pair=f'{d[d.k==k].tag.iloc[0]}->{d[d.k==k2].tag.iloc[0]}',key=key,file=key.split('::')[0],status=st,old_text=a.text[key]))
e=pd.DataFrame(rows); e['changed']=(e.status!='Unchanged').astype(int)
# history (old-release only)
first=e.groupby('key').k.min()
def hist(r,E):
    p=E[(E.key==r.key)&(E.k<r.k)]
    return pd.Series(dict(prior_obs=len(p),prior_changes=p.changed.sum(),since_change=(r.k-p[p.changed==1].k.max()) if p.changed.sum() else -1,age=r.k-first[r.key]))
E=e.copy(); H=E.apply(lambda r:hist(r,E),axis=1); e=pd.concat([e,H],axis=1)
fc=e.groupby(['file','k']).changed.mean().rename('file_churn_now').reset_index(); fc['k']+=1
e=e.merge(fc,on=['file','k'],how='left').rename(columns={'file_churn_now':'file_churn_prev'})
pc=e.groupby('k').changed.mean().rename('proj_churn_prev').reset_index(); pc['k']+=1; e=e.merge(pc,on='k',how='left')
e['old_len']=e.old_text.str.len(); e['n_steps']=e.old_text.str.count(r'(?m)^\s*(Given|When|Then|And|But)\b')
e['n_ex']=e.old_text.str.count(r'\|')
e.to_pickle('/home/claude/work/sylius_scen_dataset.pkl'); e.drop(columns='old_text').to_csv('/home/claude/out/sylius_scenario_change_dataset.csv',index=False)
print(len(e),'scenario-observations;',e.key.nunique(),'distinct scenarios;',e.pair.nunique(),'release pairs'); print(e.status.value_counts()); print(e.groupby('pair',sort=False).changed.agg(['size','mean']).round(2).T.to_string())
sets={'old length/steps':['old_len','n_steps','n_ex'],'history only':['prior_obs','prior_changes','since_change','age'],
 'history+file churn':['prior_obs','prior_changes','since_change','age','file_churn_prev','proj_churn_prev'],'all old-release':['old_len','n_steps','n_ex','prior_obs','prior_changes','since_change','age','file_churn_prev','proj_churn_prev']}
rf=lambda:RandomForestClassifier(300,class_weight='balanced',min_samples_leaf=5,random_state=0,n_jobs=-1)
res=[]; ks=[k for k in sorted(e.k.unique()) if k>=8]
for n,c in sets.items():
    A=[]
    for k in ks:
        tr=e[e.k<k]; te=e[e.k==k]
        if te.changed.nunique()<2 or tr.changed.nunique()<2: continue
        p=rf().fit(tr[c].fillna(-1),tr.changed).predict_proba(te[c].fillna(-1))[:,1]; A.append(roc_auc_score(te.changed,p))
    res.append(dict(features=n,mean_AUC=np.mean(A),median=np.median(A),min=np.min(A),max=np.max(A),pairs=len(A),pairs_above_0_5=int(np.sum(np.array(A)>.5))))
# baselines
for nm,col in [('RULE changed last time','since_change'),('RULE prior change count','prior_changes'),('RULE file churn prev','file_churn_prev')]:
    A=[]
    for k in ks:
        te=e[e.k==k]
        if te.changed.nunique()<2: continue
        s=te[col].fillna(-1); s=(s==1).astype(float)+1e-9*np.arange(len(te)) if col=='since_change' else s
        A.append(roc_auc_score(te.changed,s))
    res.append(dict(features=nm,mean_AUC=np.mean(A),median=np.median(A),min=np.min(A),max=np.max(A),pairs=len(A),pairs_above_0_5=int(np.sum(np.array(A)>.5))))
r=pd.DataFrame(res).round(3); print(r.to_string(index=False)); r.to_csv('/home/claude/out/scenario_prediction_forward_results.csv',index=False)
