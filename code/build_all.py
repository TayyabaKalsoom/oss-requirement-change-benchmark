import pandas as pd, numpy as np, difflib, re
W='/home/claude/work/'
a=pd.read_pickle('/home/claude/texts/records.pkl')
a=a[a.project!='fat_free_crm'][['project','version_pair','req_id','status','old_text','new_text']]
s=pd.read_pickle(W+'sylius_records.pkl')
d=pd.concat([a,s],ignore_index=True)
# sanity: original Unchanged == identical text
chk=d[(d.status=='Unchanged')]; print('Unchanged identical text:',(chk.old_text==chk.new_text).mean())
def sim(o,n): return difflib.SequenceMatcher(None,o,n,autojunk=False).ratio()
d['similarity_corrected']=[sim(o,n) if st=='Modified' else (1.0 if st=='Unchanged' else np.nan) for o,n,st in zip(d.old_text,d.new_text,d.status)]
TH=0.75
d['label']=np.where(d.status=='Unchanged','Unchanged',np.where(d.status.isin(['Added','Deleted']),'Full',np.where(d.similarity_corrected>=TH,'Partial','Full')))
d['old_len']=d.old_text.fillna('').str.len(); d['new_len']=d.new_text.fillna('').str.len()
# proxies v2: computed on the OLD text when it exists (pre-change), else on the new text (Added)
HEDGE=r"\b(may|might|could|possibly|probably|approximately|etc|tbd|appropriate|adequate|as needed|if possible|usually|generally|sometimes|various|some|several|normally|typically)\b"
VERIF=r"\b(verify|check|assert|should see|expect|validate|confirm|ensure|should|must|then)\b"
TECH=r"\b(database|server|api|http|https|json|xml|browser|javascript|email|ssl|session|cache|queue|token|sql|url|rest|oauth|cookie|html|css|ajax|cron|log|logs)\b"
DEPS=r"(\bUC\d+\b|\bsee\b|\brefer|\bas in\b|\bdepends?\b|\bafter\b|\bbefore\b|\brequires?\b|\bprerequisite|\bprecondition|\bbackground\b|\bgiven\b)"
KW={'g':['feature','scenario','given','when','then','as a','i want'],'u':['preconditions','main flow','sub-flow','alternative','postcondition','logging','data format']}
def proxies(t,style):
    toks=re.findall(r"[A-Za-z0-9_']+",t); n=max(len(toks),1); low=t.lower()
    cap=sum(1 for w in toks if w[0].isupper())
    return dict(Ambiguity=len(re.findall(HEDGE,low))/n, Completeness=len(set(w.lower() for w in toks))/n,
      Verification=len(re.findall(VERIF,low))/n, Technological_Needs=len(re.findall(TECH,low))/n,
      Dependency=len(re.findall(DEPS,low,flags=re.I))/n, Specificity=cap/n,
      Consistency=sum(k in low for k in KW[style])/len(KW[style]))
P=[]
for o,n,pr in zip(d.old_text,d.new_text,d.project):
    t=o if isinstance(o,str) and len(o)>0 else (n if isinstance(n,str) else '')
    P.append(proxies(t,'u' if pr=='itrust2' else 'g'))
d=pd.concat([d,pd.DataFrame(P)],axis=1)
d['group']=d.project+'|'+d.req_id
d.to_pickle(W+'all_records.pkl')
d.drop(columns=['old_text','new_text']).to_csv(W+'dataset_v2.csv',index=False)
print(pd.crosstab(d.project,d.label,margins=True))
print(pd.crosstab(d.project,d.status,margins=True))
print(d.groupby('label')[['Ambiguity','Completeness','Specificity']].mean().round(3))
print(d[['Completeness','old_len','new_len']].corr().round(2))
