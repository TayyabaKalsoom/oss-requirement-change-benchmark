import subprocess, pandas as pd, difflib
R='/home/claude/repos/Sylius'; TAGS=['v0.10.0','v0.12.0','v0.14.0','v0.16.0']
def listing(tag):
    out=subprocess.run(['git','ls-tree','-r',tag],cwd=R,capture_output=True,text=True).stdout.splitlines()
    return {l.split('\t')[1]:l.split()[2] for l in out if l.endswith('.feature')}
def blobs(shas):
    shas=sorted(set(shas)); p=subprocess.run(['git','cat-file','--batch'],cwd=R,input=('\n'.join(shas)+'\n').encode(),capture_output=True).stdout
    res={};i=0
    while i<len(p):
        j=p.index(b'\n',i); h=p[i:j].split(); i=j+1
        if len(h)<3: continue
        n=int(h[2]); res[h[0].decode()]=p[i:i+n].decode('utf8','replace'); i+=n+1
    return res
L={t:listing(t) for t in TAGS}; print({t:len(v) for t,v in L.items()},flush=True)
allsha=[s for t in TAGS for s in L[t].values()]; B=blobs(allsha); print('blobs',len(B),flush=True)
rows=[]
for a,b in zip(TAGS,TAGS[1:]):
    for path in sorted(set(L[a])|set(L[b])):
        o=B[L[a][path]] if path in L[a] else None; n=B[L[b][path]] if path in L[b] else None
        st='Added' if o is None else 'Deleted' if n is None else ('Unchanged' if o==n else 'Modified')
        rows.append(dict(project='sylius',version_pair=f'{a}->{b}',req_id=path,status=st,old_text=o,new_text=n))
pd.DataFrame(rows).to_pickle('/home/claude/work/sylius_records.pkl')
d=pd.DataFrame(rows); print(d.groupby('version_pair').status.value_counts().unstack(fill_value=0))
