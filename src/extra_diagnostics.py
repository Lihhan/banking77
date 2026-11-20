"""Extra failure-analysis diagnostics for the final report (Sections 6-7): transfer cluster, confident errors,
model agreement, length, support-F1 correlation, calibration/temperature scaling, and the post-hoc ensemble.
Run from src/:  python extra_diagnostics.py"""
import json, numpy as np, pandas as pd
from common import load_splits, RESULTS, core_metrics, selective, word_len
from scipy.stats import spearmanr, binomtest
tr, va, te, labels = load_splits(); L = np.array(labels)
y, yv = te.label.values, va.label.values
B = np.load(RESULTS/"modernbert/final_s42/test_probs.npy"); Bv = np.load(RESULTS/"modernbert/final_s42/val_probs.npy")
T = np.load(RESULTS/"tfidf/test_probs.npy"); Tv = np.load(RESULTS/"tfidf/val_probs.npy")
pb, pt = B.argmax(1), T.argmax(1)
print("== A transfer cluster")
C = ["pending_transfer","balance_not_updated_after_bank_transfer","transfer_not_received_by_recipient","failed_transfer","transfer_timing","declined_transfer"]
ci = np.isin(L[y], C); cp = np.isin(L[pb], C); err = pb != y
print("cluster test n", ci.sum(), "bert acc in cluster", (pb[ci]==y[ci]).mean().round(4), "outside", (pb[~ci]==y[~ci]).mean().round(4))
print("errors total", err.sum(), "gold in C", (err&ci).sum(), "gold&pred in C", (err&ci&cp).sum())
collapsed = np.where(cp & ci, True, pb==y); print("acc if cluster collapsed", collapsed.mean().round(4))
for n,P in [("bert",pb),("tfidf",pt)]:
    e=P!=y; print(n,"errors",e.sum(),"within-cluster",(e&ci&np.isin(L[P],C)).sum())
cm = pd.crosstab(L[y][err], L[pb][err]).stack().sort_values(ascending=False)
top10 = cm.head(10); print(top10); print("top10 pairs share", top10.sum(), "/", err.sum())
# symmetric pairs
pairs = {}
for (g,p),n in cm.items(): k=tuple(sorted((g,p))); pairs[k]=pairs.get(k,0)+n
sp = sorted(pairs.items(), key=lambda x:-x[1])[:8]; print("symmetric", sp)
print("== B confident errors")
conf = B.max(1); ce = err & (conf >= 0.96)
print("conf errors", ce.sum(), "of", err.sum(), "conf>=0.99 errors", (err&(conf>=0.99)).sum())
gr = np.array([int(np.where(np.argsort(-B[i])==y[i])[0][0])+1 for i in range(len(y))])
print("gold rank among conf errors", pd.Series(gr[ce]).value_counts().sort_index().to_dict())
print("gold rank all errors", pd.Series(gr[err]).value_counts().sort_index().to_dict())
d = te[ce].assign(pred=L[pb[ce]], conf=conf[ce].round(3), rank=gr[ce]).sort_values("conf", ascending=False)
print(d[["norm","category","pred","conf","rank"]].head(32).to_string())
print("== C model agreement")
bw, tw = pb!=y, pt!=y
print("both wrong", (bw&tw).sum(), "same wrong label", (bw&tw&(pb==pt)).sum(), "oracle acc", (~(bw&tw)).mean().round(4))
print("== D length")
wl = te.norm.map(word_len).values
for lo,hi in [(1,5),(6,10),(11,20),(21,200)]:
    m=(wl>=lo)&(wl<=hi); print(f"len {lo}-{hi}: n={m.sum()} bert acc {(pb[m]==y[m]).mean():.4f} tfidf {(pt[m]==y[m]).mean():.4f} bert cov {(conf[m]>=0.96).mean():.3f}")
print("== E support vs F1")
pc = pd.read_csv(RESULTS/"per_class_modernbert.csv"); pct = pd.read_csv(RESULTS/"per_class_tfidf.csv")
sup = tr.category.value_counts()
print("spearman bert", spearmanr(sup[pc.intent].values, pc.f1)); print("spearman tfidf", spearmanr(sup[pct.intent].values, pct.f1))
m = pc.merge(pct, on="intent", suffixes=("_b","_t")); m["d"]=m.f1_b-m.f1_t
print("bottom10 bert"); print(pc.sort_values("f1").head(10)[["intent","precision","recall","f1"]].round(3).to_string())
print("bottom10 tfidf"); print(pct.sort_values("f1").head(8)[["intent","f1"]].round(3).to_string())
print("intents where tfidf better", (m.d<0).sum(), m.sort_values("d").head(6)[["intent","f1_b","f1_t"]].round(3).to_string())
print("intents perfect bert", (pc.f1==1).sum())
print("== G calibration")
def ece(P,yy,nb=15):
    c=P.max(1); a=(P.argmax(1)==yy); e=0
    bins=np.linspace(0,1,nb+1)
    for i in range(nb):
        m=(c>bins[i])&(c<=bins[i+1])
        if m.any(): e+=m.mean()*abs(a[m].mean()-c[m].mean())
    return e
def nll(P,yy): return float(-np.log(np.clip(P[np.arange(len(yy)),yy],1e-12,1)).mean())
for n,P,Pv in [("bert",B,Bv),("tfidf",T,Tv)]:
    print(n,"test ECE",round(ece(P,y),4),"NLL",round(nll(P,y),4),"mean conf",P.max(1).mean().round(4),"acc",(P.argmax(1)==y).mean().round(4))
# temperature scaling fitted on val (log probs as logits)
def temp(P,t):
    z=np.log(np.clip(P,1e-12,1))/t; z-=z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)
for n,P,Pv in [("bert",B,Bv),("tfidf",T,Tv)]:
    ts=np.round(np.arange(0.5,5.01,0.05),2); best=min(ts,key=lambda t:nll(temp(Pv,t),yv))
    print(n,"T*",best,"test ECE",round(ece(temp(P,best),y),4),"NLL",round(nll(temp(P,best),y),4))
print("== I ensemble (weight chosen on val macro-F1)")
from sklearn.metrics import f1_score
rows=[]
for w in np.round(np.arange(0,1.01,0.1),1):
    Pv=w*Bv+(1-w)*Tv; rows.append((w,f1_score(yv,Pv.argmax(1),average="macro"),(Pv.argmax(1)==yv).mean()))
for r in rows: print("  w",r[0],"val f1",round(r[1],4),"acc",round(r[2],4))
w=max(rows,key=lambda r:(r[1],r[0]))[0]; P=w*B+(1-w)*T
print("chosen w",w,"test",core_metrics(P,y))
a=pb==y; c=P.argmax(1)==y; n01,n10=int((a&~c).sum()),int((~a&c).sum()); print("vs bert: ens right bert wrong",n10,"bert right ens wrong",n01,"p",binomtest(min(n01,n10),n01+n10,.5).pvalue)
# seeds
print("== seeds val/test")
for s in [42,13,7]:
    r=json.load(open(RESULTS/f"modernbert/final_s{s}/run.json")); print(s, r["threshold"], r["epoch_log"][-1]["val_macro_f1"])
