import pandas as pd, numpy as np, warnings, json, difflib
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score,f1_score,matthews_corrcoef,balanced_accuracy_score,roc_auc_score,confusion_matrix
W='/home/claude/work/'
d=pd.read_pickle(W+'all_records.pkl').reset_index(drop=True); TH3=0.91; d['label']=np.where(d.status=='Unchanged','Unchanged',np.where(d.status.isin(['Added','Deleted']),'Full',np.where(d.similarity_corrected>=TH3,'Partial','Full')))
d['old']=d.old_len; d['new']=d.new_len; d['len_diff']=(d.new-d.old).abs()
mx=d[['old','new']].max(axis=1); d['len_ratio']=np.where(mx>0,d[['old','new']].min(axis=1)/mx.replace(0,np.nan),0); d['len_ratio']=d.len_ratio.fillna(0)
PROX=['Ambiguity','Completeness','Verification','Technological_Needs','Dependency','Specificity','Consistency'];LEN=['old','new','len_diff','len_ratio']
cls=['Unchanged','Partial','Full']; y=d.label.values
def mets(yt,yp): return dict(acc=accuracy_score(yt,yp),macroF1=f1_score(yt,yp,average='macro',labels=cls),bal=balanced_accuracy_score(yt,yp),mcc=matthews_corrcoef(yt,yp),partF1=f1_score(yt,yp,labels=['Partial'],average=None,zero_division=0)[0])
rf=lambda:RandomForestClassifier(300,class_weight='balanced',random_state=0,n_jobs=-1)
lr=lambda:make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,class_weight='balanced'))
class Rule:
    def fit(s,X,y): return s
    def predict(s,X): return np.array(['Full' if (r[0]==0 or r[1]==0 or r[0]!=r[1]) else 'Unchanged' for r in X],dtype=object)
class Maj:
    def fit(s,X,y): s.m=pd.Series(y).value_counts().idxmax(); return s
    def predict(s,X): return np.array([s.m]*len(X),dtype=object)
def run(mk,feats,df=d,labs='label',seeds=5):
    X=df[feats].values.astype(float); yy=df[labs].values; g=df.group.values
    r=[]
    for sd in range(seeds):
        yp=np.empty(len(yy),dtype=object)
        for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=sd).split(X,yy,g):
            m=mk().fit(X[tr],yy[tr]); yp[te]=m.predict(X[te])
        r.append(mets(yy,yp))
    gcv={'grp_'+k:v for k,v in pd.DataFrame(r).mean().items()}
    yp=np.empty(len(yy),dtype=object); per={}
    for p in df.project.unique():
        te=(df.project==p).values; m=mk().fit(X[~te],yy[~te]); yp[te]=m.predict(X[te]); per[p]=mets(yy[te],yp[te])
    lo={'lopo_'+k:v for k,v in mets(yy,yp).items()}
    return {**gcv,**lo},per,yp
rows={};perp={}
for name,mk,ft in [('Majority class',Maj,['old']),('Trivial length rule',Rule,['old','new']),('RF all (7 proxies+length)',rf,PROX+LEN),('RF length-only',rf,LEN),('RF proxies-only',rf,PROX),('LR all',lr,PROX+LEN),('LR length-only',lr,LEN),('LR proxies-only',lr,PROX)]:
    r,p,yp=run(mk,ft); rows[name]=r; perp[name]=p
overall=pd.DataFrame(rows).T.round(3); print(overall.to_string())
per=pd.DataFrame({(n,p):v for n,pp in perp.items() for p,v in pp.items()}).T.round(3); per.index.names=['model','held_out']
print(per.loc[['RF all (7 proxies+length)','RF length-only','RF proxies-only']].to_string())
# bootstrap CI for RF all lopo
r,p,yp=run(rf,PROX+LEN,seeds=1); rng=np.random.default_rng(0); idx=np.arange(len(y)); bs=[]
for _ in range(2000):
    i=rng.choice(idx,len(idx)); a=mets(y[i],yp[i]); bs.append([a['acc'],a['macroF1'],a['mcc'],a['partF1']])
bs=np.array(bs); ci={k:(round(float(np.mean(bs[:,j])),3),round(float(np.percentile(bs[:,j],2.5)),3),round(float(np.percentile(bs[:,j],97.5)),3)) for j,k in enumerate(['acc','macroF1','mcc','partF1'])}
print('LOPO RF CI',ci); print(pd.DataFrame(confusion_matrix(y,yp,labels=cls),index=cls,columns=cls))
# Modified-only
mo=d[d.status=='Modified'].reset_index(drop=True); print('Modified',len(mo),mo.label.value_counts().to_dict(),mo.groupby('project').label.value_counts().unstack(fill_value=0).to_dict('index'))
mod={}
for name,mk,ft in [('RF all',rf,PROX+LEN),('RF length-only',rf,LEN),('RF proxies-only',rf,PROX),('LR all',lr,PROX+LEN),('LR length-only',lr,LEN),('LR proxies-only',lr,PROX),('Majority',Maj,['old'])]:
    X=mo[ft].values.astype(float); yy=mo.label.values; r=[];au=[]
    for sd in range(10):
        yp=np.empty(len(yy),dtype=object); pr=np.zeros(len(yy))
        for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=sd).split(X,yy,mo.group.values):
            m=mk().fit(X[tr],yy[tr]); yp[te]=m.predict(X[te])
            if hasattr(m,'predict_proba'): pr[te]=m.predict_proba(X[te])[:,list(m.classes_).index('Partial')]
        r.append(dict(acc=accuracy_score(yy,yp),macroF1=f1_score(yy,yp,average='macro'),mcc=matthews_corrcoef(yy,yp))); au.append(roc_auc_score(yy=='Partial',pr) if name!='Majority' else .5)
    dd={'grp_'+k:v for k,v in pd.DataFrame(r).mean().items()}; dd['grp_auc']=np.mean(au)
    yp=np.empty(len(yy),dtype=object); pr=np.zeros(len(yy))
    for p in mo.project.unique():
        te=(mo.project==p).values; m=mk().fit(X[~te],yy[~te]); yp[te]=m.predict(X[te])
        if hasattr(m,'predict_proba'): pr[te]=m.predict_proba(X[te])[:,list(m.classes_).index('Partial')]
    dd.update({'lopo_'+k:v for k,v in dict(acc=accuracy_score(yy,yp),macroF1=f1_score(yy,yp,average='macro'),mcc=matthews_corrcoef(yy,yp)).items()}); dd['lopo_auc']=roc_auc_score(yy=='Partial',pr) if name!='Majority' else .5
    mod[name]=dd
modt=pd.DataFrame(mod).T.round(3); print(modt.to_string())
# similarity-measure sensitivity: corrected vs original autojunk label counts (old 3 projects only), theta sweep on all Modified
ths=[0.75,0.8,0.85,0.9,0.91,0.95]
sw=pd.DataFrame({'theta_p':ths,'Partial':[int((mo.similarity_corrected>=t).sum()) for t in ths],'Full':[int((mo.similarity_corrected<t).sum()) for t in ths]}); print(sw.to_string(index=False))
d2=d.copy()
with pd.ExcelWriter(W+'refined_results_v4_independent_raters.xlsx') as w:
    pd.DataFrame({'note':['Data: iTrust2, GitLab, Diaspora*, Sylius (Fat Free CRM dropped: its files are RSpec tests). 699 records.','Similarity: difflib SequenceMatcher with autojunk=False (true Ratcliff-Obershelp). theta_p=0.75, theta_u=1.0; Added/Deleted = Full.','Proxies recomputed (v2): Completeness = type-token ratio; computed on the old text when it exists.','grp = 5-fold StratifiedGroupKFold (group=project+file), mean of 5 seeds; lopo = leave-one-project-out over all 4 projects.']}).to_excel(w,'README',index=False)
    overall.to_excel(w,'Overall_models'); per.to_excel(w,'LOPO_per_project'); modt.to_excel(w,'Modified_only'); sw.to_excel(w,'Threshold_sweep',index=False)
    pd.crosstab(d.project,d.label,margins=True).to_excel(w,'Class_distribution'); pd.crosstab(d.project,d.status,margins=True).to_excel(w,'Status_distribution')
json.dump({'ci':ci},open(W+'ci.json','w'))
