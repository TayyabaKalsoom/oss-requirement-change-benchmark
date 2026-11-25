import subprocess, pandas as pd, os, sys
R='/home/claude/repos'; O='/home/claude/texts'
f=pd.read_csv('/home/claude/res/zip/final_dataset_with_proxies.csv')
def sh(*a,cwd=None):
    return subprocess.run(a,cwd=cwd,capture_output=True)
def blob(repo,ref,path):
    p=sh('git','show',f'{ref}:{path}',cwd=f'{R}/{repo}')
    return p.stdout.decode('utf8','replace') if p.returncode==0 else None
rows=[]
# map project -> (repo, ref function, path function)
def refs(proj,pair):
    a,b=pair.split('->')
    if proj=='itrust2': return ('iTrust2','origin/'+a,'origin/'+b)
    if proj=='gitlab': return ('gitlabhq',a,b)
    if proj=='diaspora': return ('diaspora',a,b)
    if proj=='fat_free_crm': return ('fat_free_crm',a,b)
n=0
for i,r in f.iterrows():
    repo,ra,rb=refs(r.project,r.version_pair)
    path=f'docs/{r.req_id}.md' if r.project=='itrust2' else r.req_id
    old=blob(repo,ra,path); new=blob(repo,rb,path)
    rows.append(dict(idx=i,old=old,new=new))
    n+=1
    if n%50==0: print(n,flush=True)
t=pd.DataFrame(rows); t.to_pickle(O+'/texts.pkl')
f2=f.copy(); f2['old_text']=t.old; f2['new_text']=t.new
f2['len_old_fetched']=f2.old_text.str.len(); f2['len_new_fetched']=f2.new_text.str.len()
f2.to_pickle(O+'/records.pkl')
print(f2[['old_len_chars','len_old_fetched','new_len_chars','len_new_fetched']].head())
ok=((f2.old_len_chars.fillna(0)==f2.len_old_fetched.fillna(0))&(f2.new_len_chars.fillna(0)==f2.len_new_fetched.fillna(0)))
print('length match',ok.mean(), ok.sum())
print(f2[~ok][['project','version_pair','req_id','old_len_chars','len_old_fetched','new_len_chars','len_new_fetched']].head(20).to_string())
