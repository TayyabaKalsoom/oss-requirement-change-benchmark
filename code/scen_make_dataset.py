import pandas as pd, numpy as np, difflib, re, warnings; warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import sys
PROJ=sys.argv[1]; d=pd.read_pickle(f'/home/claude/work/{PROJ}_scen_snapshots.pkl')
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
e['project']=PROJ
e.to_pickle(f'/home/claude/work/{PROJ}_scen_dataset.pkl'); e.drop(columns='old_text').to_csv(f'/home/claude/out/{PROJ}_scenario_change_dataset.csv',index=False)
print(PROJ,len(e),'obs;',e.key.nunique(),'scenarios;',e.pair.nunique(),'pairs;',e.status.value_counts().to_dict())
