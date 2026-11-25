exec(open('predict_v1.py').read().split("rows=[]")[0])
rng=np.random.default_rng(0)
def bauc(y,p,B=500):
    y=np.asarray(y);p=np.asarray(p)
    if len(set(y))<2: return (np.nan,)*3
    a=roc_auc_score(y,p);bs=[]
    for _ in range(B):
        i=rng.integers(0,len(y),len(y))
        if len(set(y[i]))==2: bs.append(roc_auc_score(y[i],p[i]))
    return a,*np.percentile(bs,[2.5,97.5])
out=[]
for n,c in sets.items():
    # per-project LOPO
    for pr in e.project.unique():
        te=e[e.project==pr]; p=run(e[e.project!=pr],te,c,rf); out.append(dict(design='LOPO',features=n,project=pr,n=len(te),**dict(zip(['AUC','lo','hi'],bauc(te.changed,p)))))
    # pooled-within-project (rank-normalised inside project) LOPO
    ps=np.zeros(len(e))
    for pr in e.project.unique():
        te=(e.project==pr).values; ps[te]=pd.Series(run(e[~te],e[te],c,rf)).rank(pct=True).values
    out.append(dict(design='LOPO',features=n,project='ALL(rank within project)',n=len(e),**dict(zip(['AUC','lo','hi'],bauc(e.changed,ps)))))
    # forward-in-time within each project: train on earlier pairs of the SAME project only? too small; train on earlier pairs of all projects + same-project earlier
    ps=[];ys=[]
    for pr in e.project.unique():
        last=e[e.project==pr].k.max(); te=e[(e.project==pr)&(e.k==last)]
        tr=e[(e.k<last)|(e.project!=pr)&(e.k<=last)]; tr=tr[~((tr.project==pr)&(tr.k==last))]
        p=run(tr,te,c,rf); out.append(dict(design='forward',features=n,project=pr,n=len(te),**dict(zip(['AUC','lo','hi'],bauc(te.changed,p)))))
        ps+=list(pd.Series(p).rank(pct=True));ys+=list(te.changed)
    out.append(dict(design='forward',features=n,project='ALL(rank within project)',n=len(ys),**dict(zip(['AUC','lo','hi'],bauc(ys,ps)))))
r=pd.DataFrame(out).round(3); pd.set_option('display.width',200)
print(r[r.project=='ALL(rank within project)'].to_string(index=False))
print(r[r.features.isin(['proxies only','history only','old length only'])&(r.project!='ALL(rank within project)')].to_string(index=False))
r.to_csv('/home/claude/out/prediction_test_v2_within_project.csv',index=False)
