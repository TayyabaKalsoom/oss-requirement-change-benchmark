import pandas as pd, numpy as np, re, difflib
f=pd.read_pickle('/home/claude/texts/records.pkl')
m=f[f.status=='Modified'].copy()
def char(a,b): return difflib.SequenceMatcher(None,a,b,autojunk=False).ratio()
m['char_re']=[char(a,b) for a,b in zip(m.old_text,m.new_text)]
print('reproduces csv ratio: max abs diff', (m.char_re-m.similarity_ratio).abs().max(), 'n>0.002:',((m.char_re-m.similarity_ratio).abs()>0.002).sum())
tok=lambda s: re.findall(r"[A-Za-z0-9_']+",s.lower())
def tokr(a,b): return difflib.SequenceMatcher(None,tok(a),tok(b),autojunk=False).ratio()
def jac(A,B):
    A,B=set(A),set(B); return len(A&B)/len(A|B) if A|B else 1.0
def steps(s):
    out=[]
    for l in s.splitlines():
        l=l.strip()
        if re.match(r'^(Given|When|Then|And|But|\*)\s',l): out.append(re.sub(r'\s+',' ',re.sub(r'^(Given|When|Then|And|But|\*)\s+','',l)).lower())
    return out
def lines(s): return [re.sub(r'\s+',' ',l.strip().lower()) for l in s.splitlines() if l.strip()]
def unit(s,proj): 
    st=steps(s); return st if (proj!='itrust2' and len(st)>0) else lines(s)
m['tok_ratio']=[tokr(a,b) for a,b in zip(m.old_text,m.new_text)]
m['unit_jaccard']=[jac(unit(a,p),unit(b,p)) for a,b,p in zip(m.old_text,m.new_text,m.project)]
m['unit_ordered']=[difflib.SequenceMatcher(None,unit(a,p),unit(b,p),autojunk=False).ratio() for a,b,p in zip(m.old_text,m.new_text,m.project)]
# TF-IDF cosine (embedding stand-in)
from sklearn.feature_extraction.text import TfidfVectorizer
allt=pd.concat([f.old_text.dropna(),f.new_text.dropna()]).tolist()
tv=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True).fit(allt)
A=tv.transform(m.old_text); B=tv.transform(m.new_text)
m['tfidf_cos']=np.asarray(A.multiply(B).sum(axis=1)).ravel()
m['label_char']=m.ChangeType
m.to_pickle('/home/claude/texts/modified_sims.pkl')
c=['char_re','tok_ratio','unit_jaccard','unit_ordered','tfidf_cos']
print(m[c].corr(method='spearman').round(2))
print(m.groupby('ChangeType')[c].mean().round(3))
# disagreement: char says Full (<0.75) but unit_jaccard high
d=m[(m.char_re<0.75)&(m.unit_jaccard>=0.75)]
print('char=Full but step-set>=0.75 :',len(d),'of',(m.char_re<0.75).sum())
print(d[['project','version_pair','req_id','char_re','unit_jaccard','tok_ratio']].round(2).to_string())
d2=m[(m.char_re>=0.75)&(m.unit_jaccard<0.5)]
print('char=Partial but step-set<0.5:',len(d2))
