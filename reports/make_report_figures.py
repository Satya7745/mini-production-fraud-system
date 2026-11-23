from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT=Path(__file__).parent/'figures'; OUT.mkdir(exist_ok=True)
NAVY='#123B5D'; BLUE='#1F77B4'; CYAN='#22A6B3'; GREEN='#2A9D8F'; ORANGE='#F4A261'; RED='#D1495B'; LIGHT='#EDF4F7'; GRAY='#5B6573'

def box(ax,xy,w,h,text,color=NAVY,fs=10):
    p=FancyBboxPatch(xy,w,h,boxstyle='round,pad=0.02,rounding_size=0.02',facecolor=color,edgecolor='white',lw=1.5)
    ax.add_patch(p); ax.text(xy[0]+w/2,xy[1]+h/2,text,ha='center',va='center',color='white',fontsize=fs,fontweight='bold')

def arrow(ax,a,b,color=GRAY):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,color=color,lw=1.5))

fig,ax=plt.subplots(figsize=(10,5.6)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
box(ax,(.04,.66),.17,.18,'Public dataset\nSchema + hash',BLUE)
box(ax,(.28,.66),.18,.18,'Temporal split\nLeakage audit',CYAN)
box(ax,(.53,.66),.19,.18,'Train candidates\nTune on validation',GREEN)
box(ax,(.79,.66),.17,.18,'Versioned artifact\nModel + metadata',NAVY)
for x in [(.21,.75,.28,.75),(.46,.75,.53,.75),(.72,.75,.79,.75)]: arrow(ax,(x[0],x[1]),(x[2],x[3]))
box(ax,(.18,.25),.22,.18,'FastAPI\n/health + /predict',NAVY)
box(ax,(.50,.25),.20,.18,'Pydantic\nstrict validation',CYAN)
box(ax,(.78,.25),.17,.18,'Fraud decision\n+ request ID',GREEN)
arrow(ax,(.875,.66),(.35,.43)); arrow(ax,(.40,.34),(.50,.34)); arrow(ax,(.70,.34),(.78,.34))
box(ax,(.04,.02),.22,.12,'Fallback + rollback',ORANGE,9)
box(ax,(.38,.02),.24,.12,'Logs, latency, errors',GRAY,9)
box(ax,(.74,.02),.22,.12,'PSI / KS / labels',RED,9)
arrow(ax,(.29,.25),(.17,.14)); arrow(ax,(.29,.25),(.50,.14)); arrow(ax,(.88,.25),(.85,.14))
ax.text(.5,.94,'Training and Online Serving Architecture',ha='center',va='center',fontsize=17,fontweight='bold',color=NAVY)
fig.tight_layout(); fig.savefig(OUT/'architecture.png',dpi=200,bbox_inches='tight'); plt.close(fig)

# Split figure
names=['Train','Validation','Locked test']; rows=np.array([170236,56746,56744]); fraud=np.array([342,57,74]); colors=[NAVY,CYAN,GREEN]
fig,ax=plt.subplots(figsize=(8,3.6)); left=0
for n,r,f,c in zip(names,rows,fraud,colors):
    ax.barh([0],[r],left=left,height=.48,color=c,label=f'{n}: {r:,} rows / {f} frauds')
    ax.text(left+r/2,0,f'{n}\n{r:,}\n{f} frauds',ha='center',va='center',color='white',fontweight='bold',fontsize=10)
    left+=r
for cut,label in [(170236,'120,393 s'),(226982,'145,234 s')]:
    ax.axvline(cut,color='white',lw=2)
ax.set_xlim(0,rows.sum()); ax.set_yticks([]); ax.set_xlabel('Transactions in chronological order (after 1,081 duplicate feature rows removed)')
ax.set_title('Strict Contiguous Temporal Partitioning',color=NAVY,fontweight='bold',fontsize=15)
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.20),ncol=3,frameon=False,fontsize=8)
fig.tight_layout(); fig.savefig(OUT/'temporal_split.png',dpi=200,bbox_inches='tight'); plt.close(fig)

# Latency/quality ablation
ab=json.load(open(Path(__file__).parent/'metrics'/'latency_ablation.json'))['configurations']
labels=[f"{x['trees']} trees" for x in ab]; p95=[x['p95_ms'] for x in ab]; ap=[x['test_pr_auc'] for x in ab]
fig,ax=plt.subplots(figsize=(7.5,4.2)); x=np.arange(2); bars=ax.bar(x,p95,color=[GRAY,GREEN],width=.55)
ax.set_ylabel('p95 latency (ms)',color=NAVY); ax.set_xticks(x,labels); ax.set_title('Matched-load Serving Ablation (1,200 requests, concurrency 8)',color=NAVY,fontweight='bold')
ax.bar_label(bars,fmt='%.1f ms',padding=3,fontweight='bold'); ax.set_ylim(0,max(p95)*1.18); ax.grid(axis='y',alpha=.2)
ax2=ax.twinx(); ax2.plot(x,ap,color=ORANGE,marker='o',ms=9,lw=2.5); ax2.set_ylabel('Locked-test PR-AUC',color=ORANGE); ax2.set_ylim(.74,.83)
for i,v in enumerate(ap): ax2.text(i,v+.006,f'{v:.3f}',ha='center',color=ORANGE,fontweight='bold')
fig.tight_layout(); fig.savefig(OUT/'latency_ablation.png',dpi=200,bbox_inches='tight'); plt.close(fig)

# Monitoring blueprint
fig,ax=plt.subplots(figsize=(9,4.8)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
items=[(.07,.68,.25,.20,'System','p95 > 250 ms\nerrors > 1%\nfallback > 0.1%',NAVY),(.375,.68,.25,.20,'Data','PSI 0.10 warn\nPSI 0.25 critical\nconfirm with KS',CYAN),(.68,.68,.25,.20,'Prediction','score quantiles\nalert-rate shift\namount bands',GREEN),(.22,.25,.25,.20,'Concept','weekly PR-AUC\nrecall < 0.70\ncost trend',ORANGE),(.535,.25,.25,.20,'Response','two windows\nshadow retrain\nrollback artifact',RED)]
for x,y,w,h,title,body,c in items:
    p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.02',facecolor='white',edgecolor=c,lw=2); ax.add_patch(p)
    ax.text(x+w/2,y+h*.72,title,ha='center',va='center',fontweight='bold',color=c,fontsize=12)
    ax.text(x+w/2,y+h*.34,body,ha='center',va='center',color=GRAY,fontsize=9)
for a,b in [((.195,.68),(.33,.45)),((.50,.68),(.47,.45)),((.805,.68),(.67,.45)),((.47,.25),(.535,.35))]: arrow(ax,a,b)
ax.text(.5,.95,'Operational Monitoring and Action Policy',ha='center',fontweight='bold',fontsize=17,color=NAVY)
ax.text(.5,.08,'Alert -> investigate lineage and labels -> shadow retrain -> approve -> canary -> rollback if SLO or quality regresses',ha='center',fontsize=10,color=GRAY)
fig.tight_layout(); fig.savefig(OUT/'monitoring_blueprint.png',dpi=200,bbox_inches='tight'); plt.close(fig)
print('created',sorted(p.name for p in OUT.glob('*.png')))
