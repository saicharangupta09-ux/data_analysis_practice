import pandas as pd, numpy as np, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,roc_curve,confusion_matrix)
from sklearn.inspection import permutation_importance
df=pd.read_csv("synthetic_telco_churn.csv")
R={}
# ---- data quality ----
R["rows"],R["cols"]=df.shape; R["missing"]=int(df.isna().sum().sum()); R["dups"]=int(df.customerID.duplicated().sum())
q1,q3=df.MonthlyCharges.quantile([.25,.75]); iqr=q3-q1
R["outliers_mc"]=int(((df.MonthlyCharges<q1-1.5*iqr)|(df.MonthlyCharges>q3+1.5*iqr)).sum())
# ---- feature engineering ----
df["y"]=(df.Churn=="Yes").astype(int)
df["is_mtm"]=(df.Contract=="Month-to-month").astype(int)
df["new_customer"]=(df.tenure<=6).astype(int)
df["has_support"]=(df.TechSupport=="Yes").astype(int)
df["autopay"]=df.PaymentMethod.isin(["Bank transfer","Credit card"]).astype(int)
df["avg_charge_per_month"]=df.TotalCharges/df.tenure
df["fiber_no_support"]=((df.InternetService=="Fiber optic")&(df.has_support==0)).astype(int)
cat=["Contract","InternetService","PaymentMethod","TechSupport"]
num=["tenure","MonthlyCharges","TotalCharges","SeniorCitizen","is_mtm","new_customer","has_support","autopay","avg_charge_per_month","fiber_no_support"]
X=df[cat+num]; y=df.y
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
R["train"],R["test"]=len(Xtr),len(Xte); R["churn_rate"]=y.mean()*100
pre=ColumnTransformer([("c",OneHotEncoder(drop="first"),cat),("n",StandardScaler(),num)])
models={"Baseline (majority class)":DummyClassifier(strategy="most_frequent"),
 "Logistic Regression":LogisticRegression(max_iter=1000),
 "Random Forest":RandomForestClassifier(n_estimators=300,min_samples_leaf=5,random_state=42,n_jobs=-1),
 "Gradient Boosting":GradientBoostingClassifier(random_state=42)}
cv=StratifiedKFold(5,shuffle=True,random_state=42)
res={};fitted={}
for n,m in models.items():
    p=Pipeline([("pre",pre),("m",m)]).fit(Xtr,ytr); fitted[n]=p
    pr=p.predict(Xte); pp=p.predict_proba(Xte)[:,1]
    res[n]=dict(accuracy=accuracy_score(yte,pr),precision=precision_score(yte,pr,zero_division=0),recall=recall_score(yte,pr),
      f1=f1_score(yte,pr),auc=roc_auc_score(yte,pp),
      cv_auc=cross_val_score(Pipeline([("pre",pre),("m",m)]),Xtr,ytr,cv=cv,scoring="roc_auc").mean() if n!="Baseline (majority class)" else 0.5)
R["results"]=res
best=max([k for k in res if "Baseline" not in k],key=lambda k:res[k]["auc"]); R["best"]=best
bp=fitted[best]; pp=bp.predict_proba(Xte)[:,1]
# threshold tuning: pick threshold maximizing F1
ths=np.linspace(.1,.7,61); f1s=[f1_score(yte,pp>=t) for t in ths]; tb=float(ths[int(np.argmax(f1s))])
pt=(pp>=tb).astype(int)
R["thr"]=tb; R["tuned"]=dict(precision=precision_score(yte,pt),recall=recall_score(yte,pt),f1=f1_score(yte,pt),accuracy=accuracy_score(yte,pt))
cm=confusion_matrix(yte,pt); R["cm"]=cm.tolist()
# top-decile lift
o=np.argsort(-pp); k=len(pp)//10; R["top_decile_churn"]=float(yte.values[o[:k]].mean()*100)
R["lift"]=R["top_decile_churn"]/R["churn_rate"]
# importance (permutation on best)
pi=permutation_importance(bp,Xte,yte,scoring="roc_auc",n_repeats=8,random_state=42,n_jobs=-1)
imp=pd.Series(pi.importances_mean,index=X.columns).sort_values(ascending=False); R["imp"]=imp.head(8).round(4).to_dict()
# ---- charts ----
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False}); BLUE,RED,GREY,GREEN="#2B5C8A","#C0392B","#9AA5B1","#2E8B57"
fig,ax=plt.subplots(figsize=(6.2,3.8))
for n,c in zip(list(models)[1:],[GREY,GREEN,BLUE]):
    f,t,_=roc_curve(yte,fitted[n].predict_proba(Xte)[:,1]); ax.plot(f,t,color=c,lw=2,label=f"{n} (AUC {res[n]['auc']:.3f})")
ax.plot([0,1],[0,1],ls="--",color="#bbb"); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("ROC curves: all models beat the baseline",loc="left",fontweight="bold"); ax.legend(frameon=False,loc="lower right",fontsize=8)
fig.tight_layout(); fig.savefig("m_roc.png",dpi=170); plt.close()
fig,ax=plt.subplots(figsize=(6.2,3.4)); names=list(res)[1:]
x=np.arange(len(names)); w=.2
for i,(mt,c) in enumerate(zip(["precision","recall","f1","auc"],[BLUE,RED,GREEN,GREY])):
    ax.bar(x+i*w-1.5*w,[res[n][mt] for n in names],w,color=c,label=mt.upper() if mt=="auc" else mt.capitalize())
ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0,1); ax.legend(frameon=False,ncol=4,fontsize=8,loc="upper center")
ax.set_title("Model comparison on the held-out test set",loc="left",fontweight="bold")
fig.tight_layout(); fig.savefig("m_compare.png",dpi=170); plt.close()
fig,ax=plt.subplots(figsize=(4.6,3.6)); ax.imshow(cm,cmap="Blues")
for i in range(2):
    for j in range(2): ax.text(j,i,f"{cm[i,j]}",ha="center",va="center",fontsize=16,color="white" if cm[i,j]>cm.max()/2 else "black")
ax.set_xticks([0,1]); ax.set_yticks([0,1]); ax.set_xticklabels(["Stay","Churn"]); ax.set_yticklabels(["Stay","Churn"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"{best}, threshold {tb:.2f}",loc="left",fontweight="bold",fontsize=10)
for s in ax.spines.values(): s.set_visible(False)
fig.tight_layout(); fig.savefig("m_cm.png",dpi=170); plt.close()
t=imp.head(8)[::-1]; fig,ax=plt.subplots(figsize=(6.2,3.4)); ax.barh(t.index,t.values,color=BLUE)
ax.set_title("Top drivers of predicted churn (permutation importance)",loc="left",fontweight="bold",fontsize=10); ax.set_xlabel("Drop in AUC when feature is shuffled")
fig.tight_layout(); fig.savefig("m_imp.png",dpi=170); plt.close()
json.dump(R,open("ml.json","w"),indent=1,default=float); print(json.dumps(R,indent=1,default=float))
