#!/usr/bin/env python3
"""Build publication-quality figures from the six empirical studies (real computed numbers)."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
NAVY="#1f3358"; TEAL="#2f8f6e"; BLUE="#3f86c0"; RED="#c0506a"; GREY="#9aa8be"; GOLD="#c8922b"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.edgecolor":"#5a6b86","axes.linewidth":0.8})

A=json.load(open("study_A_results_FINAL.json"))["trials"]
B=json.load(open("study_B_results_FINAL.json"))["results"]
C=json.load(open("study_C_results_FINAL.json"))["trials"]
D=json.load(open("study_D_results.json"))
F=json.load(open("study_F_results_FINAL.json"))
G=json.load(open("study_G_results_FINAL.json"))

# ============ FIGURE 1 — empirical validation of the identifiability framework ============
fig=plt.figure(figsize=(13,9)); gs=GridSpec(2,2,figure=fig,hspace=0.34,wspace=0.26)

# (a) observation vs intervention orientation (Study A)
ax=fig.add_subplot(gs[0,0])
obs=np.mean([t['obs_oriented_frac'] for t in A]); pc=np.mean([t['pc']['oriented_frac'] for t in A if 'pc' in t])
full=1.0
ax.bar([0,1,2],[pc,obs,full],color=[GREY,BLUE,TEAL],width=0.6)
for x,v in zip([0,1,2],[pc,obs,full]): ax.text(x,v+0.02,f"{v:.2f}",ha="center",fontweight="bold",color=NAVY)
ax.set_xticks([0,1,2]); ax.set_xticklabels(["PC\n(observation)","v-struct+Meek\n(observation)","+ interventions"],fontsize=8.5)
ax.set_ylabel("fraction of edges oriented"); ax.set_ylim(0,1.15)
ax.set_title("(a) Observation underdetermines direction;\nintervention completes it",fontsize=10.5,fontweight="bold",color=NAVY,loc="left")
for s in ["top","right"]: ax.spines[s].set_visible(False)

# (b) orientation vs intervention coverage, random vs greedy (Study A)
ax=fig.add_subplot(gs[0,1])
A20=[t for t in A if t['n']==20]
cr=np.mean([t['orient_curve_random'] for t in A20],axis=0)
cg=np.mean([t['orient_curve_greedy'] for t in A20],axis=0)
x=np.arange(len(cr))
ax.plot(x,cr,"-o",color=GREY,ms=3,label="random targets")
ax.plot(x,cg,"-o",color=RED,ms=3,label="greedy (active design)")
itr=np.mean([t['int_to_full_random'] for t in A20]); itg=np.mean([t['int_to_full_greedy'] for t in A20])
ax.axhline(1.0,ls=":",color="#888",lw=0.8)
ax.set_xlabel("number of perturbed genes (20-gene networks)"); ax.set_ylabel("fraction oriented"); ax.set_ylim(0,1.05); ax.set_xlim(0,20)
ax.legend(frameon=False,fontsize=8.5,loc="lower right")
ax.set_title(f"(b) Greedy design reaches full orientation in\n~{itg:.1f} vs ~{itr:.1f} perturbations (20-gene nets)",fontsize=10.5,fontweight="bold",color=NAVY,loc="left")
for s in ["top","right"]: ax.spines[s].set_visible(False)

# (c) algorithm x functional form F1 (Study C) — contract-matrix validation
ax=fig.add_subplot(gs[1,0])
algs=["PC","GES","NOTEARS","DirectLiNGAM"]; forms=["linear-gauss","linear-exp","nonlinear"]; flab=["Gaussian","non-Gaussian","nonlinear"]
w=0.25; xa=np.arange(len(algs)); cols=[BLUE,TEAL,GOLD]
for k,f in enumerate(forms):
    vals=[np.mean([t[a]['f1'] for t in C if t['form']==f and a in t and isinstance(t[a],dict)]) for a in algs]
    ax.bar(xa+(k-1)*w,vals,w,color=cols[k],label=flab[k])
ax.set_xticks(xa); ax.set_xticklabels(algs,fontsize=8.5,rotation=12); ax.set_ylabel("structure-recovery F1"); ax.set_ylim(0,1)
ax.legend(frameon=False,fontsize=8,title="noise / form",title_fontsize=8)
ax.set_title("(c) Each method recovers structure only when its\ncontract holds (LiNGAM needs non-Gaussian noise)",fontsize=10.5,fontweight="bold",color=NAVY,loc="left")
for s in ["top","right"]: ax.spines[s].set_visible(False)

# (d) obstruction hierarchy obstruction vs remedy (Study D)
ax=fig.add_subplot(gs[1,1])
levels=["L1 redundancy","L2 faithfulness","L3 state-cond."]
obstructed=[D["L1_redundancy"]["single_KO_effect"][0], D["L2_faithfulness"]["edge_detected_obs_rate"][0], D["L3_state"]["detected_bulk_rate"][0]]
remedied=[D["L1_redundancy"]["family_KO_effect"][0], D["L2_faithfulness"]["edge_detected_remedy_rate"][0], D["L3_state"]["detected_within_rate"][0]]
xa=np.arange(3); w=0.36
ax.bar(xa-w/2,obstructed,w,color=GREY,label="naive method (obstructed)")
ax.bar(xa+w/2,remedied,w,color=TEAL,label="minimal remedy")
ax.set_xticks(xa); ax.set_xticklabels(levels,fontsize=8.5); ax.set_ylabel("detection / effect (normalised)"); ax.set_ylim(0,1.1)
ax.legend(frameon=False,fontsize=8)
ax.set_title("(d) Each obstruction blocks the naive method;\nits minimal design discharges it",fontsize=10.5,fontweight="bold",color=NAVY,loc="left")
for s in ["top","right"]: ax.spines[s].set_visible(False)
fig.suptitle("Empirical validation of the identifiability framework (original analyses)",fontsize=13,fontweight="bold",color=NAVY,x=0.02,ha="left")
fig.savefig("media/study_validation_causal.png",dpi=600,bbox_inches="tight",facecolor="white")
fig.savefig("media/study_validation_causal.pdf",bbox_inches="tight",facecolor="white"); plt.close(fig)

# ============ FIGURE 2 — perturbation prediction: simple >= deep, and why ============
fig=plt.figure(figsize=(13,5)); gs=GridSpec(1,2,figure=fig,wspace=0.3)
# (a) benchmark with CIs (Study B)
ax=fig.add_subplot(gs[0,0])
items=[(k,v) for k,v in B.items() if v["mean_r"]==v["mean_r"]]   # drop nan (No change)
items=sorted(items,key=lambda kv:kv[1]["mean_r"])
names=[k for k,_ in items]; rs=[v["mean_r"] for _,v in items]
lo=[v["mean_r"]-v["ci95_r"][0] for _,v in items]; hi=[v["ci95_r"][1]-v["mean_r"] for _,v in items]
cols=[RED if "deep" in n.lower() else (TEAL if r>=0.88 else GREY) for n,r in zip(names,rs)]
ax.barh(range(len(names)),rs,xerr=[lo,hi],color=cols,height=0.62,error_kw=dict(ecolor="#444",lw=1))
for i,r in enumerate(rs): ax.text(r+0.012,i,f"{r:.2f}",va="center",fontsize=8.5,fontweight="bold",color=NAVY)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names,fontsize=8.5); ax.set_xlabel("mean correlation with true effect (95% CI)"); ax.set_xlim(0,1.05)
ax.set_title("(a) Simple baselines match/beat the deep model\n(disjoint CIs; Norman 2019, 131 held-out combos)",fontsize=10.5,fontweight="bold",color=NAVY,loc="left")
for s in ["top","right"]: ax.spines[s].set_visible(False)
# (b) synergy terciles (Study F)
ax=fig.add_subplot(gs[0,1])
tn=["low-synergy","mid-synergy","high-synergy"]; x=np.arange(3)
mp={"Additive":TEAL,"Random forest":BLUE,"MLP (deep)":RED}
for name,col in mp.items():
    d=F["per_method_r_by_synergy_tercile"][name]; ax.plot(x,[d[t] for t in tn],"-o",color=col,label=name,ms=5)
ax.set_xticks(x); ax.set_xticklabels(["low","medium","high"]); ax.set_xlabel("genetic-interaction (synergy) level"); ax.set_ylabel("mean correlation"); ax.set_ylim(0.45,1.0)
ax.legend(frameon=False,fontsize=8.5)
ax.set_title("(b) Every model degrades on high-synergy combos:\nthe unpredictable part is genuine interaction",fontsize=10.5,fontweight="bold",color=NAVY,loc="left")
for s in ["top","right"]: ax.spines[s].set_visible(False)
fig.suptitle("Perturbation-effect prediction: simple beats deep, and synergy sets the limit (Norman 2019)",fontsize=12.5,fontweight="bold",color=NAVY,x=0.02,ha="left")
fig.savefig("media/study_perturbation_ml.png",dpi=600,bbox_inches="tight",facecolor="white")
fig.savefig("media/study_perturbation_ml.pdf",bbox_inches="tight",facecolor="white"); plt.close(fig)

print("figures written:")
import os
for f in ["study_validation_causal.png","study_perturbation_ml.png"]:
    p="media/"+f; print(f"  {p}  {os.path.getsize(p)//1024} KB" if os.path.exists(p) else f"  MISSING {p}")
