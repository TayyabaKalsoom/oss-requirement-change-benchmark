import subprocess, re, pandas as pd, sys, difflib
R='/home/claude/repos/'+sys.argv[1]; PAT=sys.argv[2]; OUT=sys.argv[3]; MAXN=int(sys.argv[4]) if len(sys.argv)>4 else 10**9
def g(*a,inp=None): return subprocess.run(['git','-C',R,*a],input=inp,capture_output=True).stdout
tags=[t for t in g('tag','--sort=creatordate').decode().split() if re.fullmatch(PAT,t)]
tags=sorted(set(tags),key=lambda t:tuple(map(int,t[1:].split('.'))))
tags=[t for t in tags if int(t[1:].split('.')[0])<=MAXN]
snap={}; blobs={}
for t in tags:
    for l in g('ls-tree','-r',t,'--','features').decode().splitlines():
        meta,path=l.split('\t'); sha=meta.split()[2]
        if path.endswith('.feature'): snap.setdefault(t,{})[path]=sha; blobs[sha]=None
print(len(tags),'tags',len(blobs),'blobs',flush=True)
shas=list(blobs); inp=''.join(s+'\n' for s in shas).encode()
out=g('cat-file','--batch',inp=inp); pos=0; i=0
while pos<len(out):
    nl=out.index(b'\n',pos); hdr=out[pos:nl].split(); size=int(hdr[2]); blobs[hdr[0].decode()]=out[nl+1:nl+1+size].decode('utf8','ignore'); pos=nl+1+size+1
def scen(txt):
    res={}; cur=None; buf=[]
    for line in txt.splitlines():
        m=re.match(r'\s*(Scenario Outline|Scenario|Example):?\s*(.*)',line)
        if m:
            if cur: res[cur]=buf
            cur=re.sub(r'\s+',' ',m.group(2).strip().lower()); buf=[line]
        elif cur is not None: buf.append(line)
    if cur: res[cur]=buf
    return {k:'\n'.join(v).strip() for k,v in res.items() if k}
rows=[]
for ti,t in enumerate(tags):
    for path,sha in snap.get(t,{}).items():
        for k,tx in scen(blobs[sha] or '').items(): rows.append((ti,t,path,k,tx))
d=pd.DataFrame(rows,columns=['k','tag','file','title','text']); d['key']=d.file+'::'+d.title
d=d.drop_duplicates(['k','key'])
d.to_pickle(OUT); print(d.groupby('tag',sort=False).size().to_string())
